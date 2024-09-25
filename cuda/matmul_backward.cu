#include "cpu_utils.cuh"
#include "gpu_utils.cuh"

#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>


template<unsigned int BS_DIM, unsigned int WS_DIM, unsigned int D_DIM, 
         unsigned int MMA_M_DIM, unsigned int MMA_N_DIM, unsigned int MMA_K_DIM>
__global__ void matmul_backward_db(half* dH, half* Q, half* K, half* V, half* F, half* I,
      half* M, half* B, half* dB, const unsigned int S_DIM){
    /*
        BS_DIM: Number of rows a block is responsible for
        WS_DIM: Number of rows a warp is responsible for (WS_DIM = MMA_M_DIM)
        D_DIM: Hidden/Head dimension
     */

    constexpr unsigned int WARP_SIZE = 32;
    constexpr unsigned int warp_mma_blocks_d = D_DIM / MMA_K_DIM;
    constexpr unsigned int warp_mma_blocks_bs = BS_DIM / MMA_N_DIM;

    const float NINFINITY = -CUDART_INF_F;
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    // the two rows inside the block the warp is responsible for
    unsigned int block_row_1 = warp_id * WS_DIM + lane_id / 4;
    unsigned int block_row_2 = warp_id * WS_DIM + lane_id / 4 + 8;
    // the two rows inside the whole matrix the warp is responsible for
    unsigned int matrix_row_1 = blockIdx.x * BS_DIM + block_row_1;
    unsigned int matrix_row_2 = blockIdx.x * BS_DIM + block_row_2;

    Q += blockIdx.x * BS_DIM * D_DIM;
    dH += blockIdx.x * BS_DIM * D_DIM;
    M += blockIdx.x * BS_DIM;
    B += blockIdx.x * BS_DIM;
    dB += blockIdx.x * BS_DIM;

    extern __shared__ half shmem[];
    half* qkv_shmem = shmem;
    half* imb_shmem = (half*) &shmem[BS_DIM * D_DIM];
    float* f_shmem = (float*) &shmem[BS_DIM * D_DIM + BS_DIM];

    uint32_t dH_reg[warp_mma_blocks_d][2];
    uint32_t Q_reg[warp_mma_blocks_d][2];
    uint32_t KV_reg[warp_mma_blocks_bs][warp_mma_blocks_d];
    uint32_t QK_reg[warp_mma_blocks_bs][2];
    uint32_t dC_reg[warp_mma_blocks_bs][2];

    float m_reg[2];
    float f_row[2];
    f_row[0] = 0.0f;
    f_row[1] = 0.0f;
    float dn_acc[2];
    dn_acc[0] = 0.0f;
    dn_acc[1] = 0.0f;

    half (&KV_reg_half) [warp_mma_blocks_bs][warp_mma_blocks_d][2] = reinterpret_cast<half(&)[warp_mma_blocks_bs][warp_mma_blocks_d][2]>(KV_reg);
    half (&QK_reg_half) [warp_mma_blocks_bs][4] = reinterpret_cast<half(&)[warp_mma_blocks_bs][4]>(QK_reg);
    half (&dC_reg_half) [warp_mma_blocks_bs][4] = reinterpret_cast<half(&)[warp_mma_blocks_bs][4]>(dC_reg);

    // Load Q, M
    if (warp_id < 2) imb_shmem[threadIdx.x] = M[threadIdx.x];
    loadMatrixX2<BS_DIM, WS_DIM, D_DIM, MMA_K_DIM, warp_mma_blocks_d>(Q, qkv_shmem, Q_reg, warp_id, lane_id);
    m_reg[0] = __half2float(imb_shmem[block_row_1]);
    m_reg[1] = __half2float(imb_shmem[block_row_2]);

    // Load dH
    loadMatrixX2<BS_DIM, WS_DIM, D_DIM, MMA_K_DIM, warp_mma_blocks_d>(dH, qkv_shmem, dH_reg, warp_id, lane_id);

    float scale = rsqrtf(D_DIM);
    // causal
    K += blockIdx.x * BS_DIM * D_DIM;
    V += blockIdx.x * BS_DIM * D_DIM;
    F += blockIdx.x * BS_DIM;
    I += blockIdx.x * BS_DIM;
    unsigned int matrix_col = blockIdx.x * BS_DIM;
    for (unsigned int block_s = 0; block_s < blockIdx.x+1; block_s++) {
        zero_accumulator<half, warp_mma_blocks_bs>(QK_reg_half);
        zero_accumulator<half, warp_mma_blocks_bs>(dC_reg_half);

        // Load K
        blockCpy(K, qkv_shmem, BS_DIM, D_DIM);
        __syncthreads();

        half* k_pointer = qkv_shmem + (lane_id % 8) * D_DIM;
        uint32_t k_offset;
        CVTA_TO_SHARED_U32(k_pointer, k_offset);
        for (unsigned int warp_bs = 0; warp_bs < warp_mma_blocks_bs; warp_bs++) {
            for (unsigned int warp_d = 0; warp_d < warp_mma_blocks_d; warp_d++) {
                const uint32_t thread_offset = k_offset + (warp_bs * MMA_N_DIM * D_DIM + warp_d * MMA_K_DIM) * sizeof(half);
                LDMATRIX_X1(KV_reg[warp_bs][warp_d], thread_offset);
                KV_reg_half[warp_bs][warp_d][0] *= scale;
                KV_reg_half[warp_bs][warp_d][1] *= scale;
            }
        }

        // Compute QK^T
        matrixMultBSD<warp_mma_blocks_bs, warp_mma_blocks_d>(QK_reg, Q_reg, KV_reg);

        // Load V, F, I
        blockCpy(V, qkv_shmem, BS_DIM, D_DIM);
        load_f_i(F, f_shmem, I, imb_shmem, warp_id, lane_id);
        __syncthreads();

        // Compute D_tilde
        f_row[0] = block_s > 0 ? f_row[0] + f_shmem[BS_DIM-1] : f_shmem[block_row_1];
        f_row[1] = block_s > 0 ? f_row[1] + f_shmem[BS_DIM-1] : f_shmem[block_row_2];
        for (unsigned int warp_bs = 0; warp_bs < warp_mma_blocks_bs; warp_bs++) {
            unsigned int tile_col_1 = warp_bs * MMA_N_DIM + (lane_id % 4) * 2;
            unsigned int tile_col_2 = tile_col_1 + 1;
            unsigned int matrix_col_1 = matrix_col + tile_col_1;
            unsigned int matrix_col_2 = matrix_col + tile_col_2;
            float i1 = __half2float(imb_shmem[tile_col_1]);
            float i2 = __half2float(imb_shmem[tile_col_2]);
            float f_col_1 = f_shmem[tile_col_1];
            float f_col_2 = f_shmem[tile_col_2];

            QK_reg_half[warp_bs][0] *= __float2half(expf((matrix_row_1 >= matrix_col_1 ? f_row[0] - f_col_1 + i1 : NINFINITY) - m_reg[0]));
            QK_reg_half[warp_bs][1] *= __float2half(expf((matrix_row_1 >= matrix_col_2 ? f_row[0] - f_col_2 + i2 : NINFINITY) - m_reg[0]));
            QK_reg_half[warp_bs][2] *= __float2half(expf((matrix_row_2 >= matrix_col_1 ? f_row[1] - f_col_1 + i1 : NINFINITY) - m_reg[1]));
            QK_reg_half[warp_bs][3] *= __float2half(expf((matrix_row_2 >= matrix_col_2 ? f_row[1] - f_col_2 + i2 : NINFINITY) - m_reg[1]));
        }

        half* v_pointer = qkv_shmem + (lane_id % 8) * D_DIM;
        uint32_t v_offset;
        CVTA_TO_SHARED_U32(v_pointer, v_offset);
        for (unsigned int warp_bs = 0; warp_bs < warp_mma_blocks_bs; warp_bs++) {
            for (unsigned int warp_d = 0; warp_d < warp_mma_blocks_d; warp_d++) {
                const uint32_t thread_offset = v_offset + (warp_bs * MMA_N_DIM * D_DIM + warp_d * MMA_K_DIM) * sizeof(half);
                LDMATRIX_X1(KV_reg[warp_bs][warp_d], thread_offset);
            }
        }

        // Compute dC = dH*V^T
        matrixMultBSD<warp_mma_blocks_bs, warp_mma_blocks_d>(dC_reg, dH_reg, KV_reg);

        float dn_tmp[2];
        dn_tmp[0] = 0.0f;
        dn_tmp[1] = 0.0f;
        #pragma unroll
        for (unsigned int warp_bs = 0; warp_bs < warp_mma_blocks_bs; warp_bs++) {
            dn_tmp[0] += __half2float(dC_reg_half[warp_bs][0] * QK_reg_half[warp_bs][0] + dC_reg_half[warp_bs][1] * QK_reg_half[warp_bs][1]);
            dn_tmp[1] += __half2float(dC_reg_half[warp_bs][2] * QK_reg_half[warp_bs][2] + dC_reg_half[warp_bs][3] * QK_reg_half[warp_bs][3]);
        }
        sum_reduce_groups_of_4<float>(dn_tmp, lane_id);
        dn_acc[0] += dn_tmp[0];
        dn_acc[1] += dn_tmp[1];

        K -= BS_DIM * D_DIM;
        V -= BS_DIM * D_DIM;
        I -= BS_DIM;
        F -= BS_DIM;
        matrix_col -= BS_DIM;
    }

    if(warp_id < 2) imb_shmem[threadIdx.x] = B[threadIdx.x];
    __syncthreads();
    // 4 threads share a row
    if ((lane_id % 4) == 0) {
        float b_row_1 = __half2float(imb_shmem[block_row_1]);
        float b_row_2 = __half2float(imb_shmem[block_row_2]);
        float n_row_1 = (fmaxf(fabsf(b_row_1), expf(-m_reg[0])) + 1e-6);
        float n_row_2 = (fmaxf(fabsf(b_row_2), expf(-m_reg[1])) + 1e-6);
        dn_acc[0] = -dn_acc[0] / powf(n_row_1, 2.0f);
        dn_acc[1] = -dn_acc[1] / powf(n_row_2, 2.0f);
        float db_row_1 = fabsf(b_row_1) > expf(-m_reg[0]) ? signbit(b_row_1) * dn_acc[0] : 0.0f;
        float db_row_2 = fabsf(b_row_2) > expf(-m_reg[1]) ? signbit(b_row_2) * dn_acc[1] : 0.0f;
        dB[block_row_1] = db_row_1;
        dB[block_row_2] = db_row_2;
    }
}


template<unsigned int BS_DIM, unsigned int WS_DIM, unsigned int D_DIM, 
         unsigned int MMA_M_DIM, unsigned int MMA_N_DIM, unsigned int MMA_K_DIM>
__global__ void matmul_backward(half* dH, half* dB, half* Q, half* K, half* V, half* dQ, half* dK,
                                half* dV, half* F, half* I, half* dI, half* M, half* B, const unsigned int S_DIM){
    /*
        BS_DIM: Number of rows a block is responsible for
        WS_DIM: Number of rows a warp is responsible for (WS_DIM = MMA_M_DIM)
        D_DIM: Hidden/Head dimension
     */

    constexpr unsigned int WARP_SIZE = 32;
    constexpr unsigned int warp_mma_blocks_d = D_DIM / MMA_K_DIM;
    constexpr unsigned int warp_mma_blocks_bs = BS_DIM / MMA_N_DIM;

    const float NINFINITY = -CUDART_INF_F;
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    // the two rows inside the block the warp is responsible for
    unsigned int block_row_1 = warp_id * WS_DIM + lane_id / 4;
    unsigned int block_row_2 = warp_id * WS_DIM + lane_id / 4 + 8;
    // the two rows inside the whole matrix the warp is responsible for
    unsigned int matrix_row_1 = blockIdx.x * BS_DIM + block_row_1;
    unsigned int matrix_row_2 = blockIdx.x * BS_DIM + block_row_2;

    Q += blockIdx.x * BS_DIM * D_DIM;
    dQ += blockIdx.x * BS_DIM * D_DIM;
    dK += blockIdx.x * BS_DIM * D_DIM;
    dH += blockIdx.x * BS_DIM * D_DIM;
    dI += blockIdx.x * BS_DIM;
    M += blockIdx.x * BS_DIM;
    B += blockIdx.x * BS_DIM;
    dB += blockIdx.x * BS_DIM;

    extern __shared__ half shmem[];
    half* qkv_shmem = shmem;
    half* i_shmem = (half*) &shmem[BS_DIM * D_DIM];
    half* m_shmem = (half*) &shmem[BS_DIM * D_DIM + BS_DIM];
    half* b_shmem = (half*) &shmem[BS_DIM * D_DIM + 2 * BS_DIM];
    half* db_shmem = (half*) &shmem[BS_DIM * D_DIM + 3 * BS_DIM];
    float* f_shmem = (float*) &shmem[BS_DIM * D_DIM + 4 * BS_DIM];

    uint32_t dH_reg[warp_mma_blocks_d][2];
    uint32_t Q_reg[warp_mma_blocks_d][2];
    uint32_t KV_reg[warp_mma_blocks_bs][warp_mma_blocks_d];
    uint32_t QK_reg[warp_mma_blocks_bs][2];
    uint32_t dC_reg[warp_mma_blocks_bs][2];
    uint32_t dQdK_acc[warp_mma_blocks_d][2];

    // Used for m when computing dQ and i when computing dK, dV
    float m_i_reg[2];
    float n_reg[2];
    // Used for b when computing dQ and di when computing dK, dV
    float b_di_reg[2];
    float db_reg[2];
    float f_row[2];
    f_row[0] = 0.0f;
    f_row[1] = 0.0f;
    float df_acc[2];
    df_acc[0] = 0.0f;
    df_acc[1] = 0.0f;

    half (&KV_reg_half) [warp_mma_blocks_bs][warp_mma_blocks_d][2] = reinterpret_cast<half(&)[warp_mma_blocks_bs][warp_mma_blocks_d][2]>(KV_reg);
    half (&QK_reg_half) [warp_mma_blocks_bs][4] = reinterpret_cast<half(&)[warp_mma_blocks_bs][4]>(QK_reg);
    half (&dC_reg_half) [warp_mma_blocks_bs][4] = reinterpret_cast<half(&)[warp_mma_blocks_bs][4]>(dC_reg);
    half (&dQdK_acc_half) [warp_mma_blocks_d][4] = reinterpret_cast<half(&)[warp_mma_blocks_d][4]>(dQdK_acc);

    /*
       Compute dQ
     */
    // Load Q, M, B
    if (warp_id < 2) {
        m_shmem[threadIdx.x] = M[threadIdx.x];
    } else {
        b_shmem[threadIdx.x % 64] = B[threadIdx.x % 64];
    }
    loadMatrixX2<BS_DIM, WS_DIM, D_DIM, MMA_K_DIM, warp_mma_blocks_d>(Q, qkv_shmem, Q_reg, warp_id, lane_id);

    m_i_reg[0] = __half2float(m_shmem[block_row_1]);
    m_i_reg[1] = __half2float(m_shmem[block_row_2]);
    b_di_reg[0] = __half2float(b_shmem[block_row_1]);
    b_di_reg[1] = __half2float(b_shmem[block_row_2]);
    n_reg[0] = fmaxf(fabsf(b_di_reg[0]), expf(-m_i_reg[0])) + 1e-6;
    n_reg[1] = fmaxf(fabsf(b_di_reg[1]), expf(-m_i_reg[1])) + 1e-6;

    // Load dH, dB
    if (warp_id < 2) b_shmem[threadIdx.x] = dB[threadIdx.x];
    loadMatrixX2<BS_DIM, WS_DIM, D_DIM, MMA_K_DIM, warp_mma_blocks_d>(dH, qkv_shmem, dH_reg, warp_id, lane_id);
    db_reg[0] = __half2float(b_shmem[block_row_1]);
    db_reg[1] = __half2float(b_shmem[block_row_2]);

    zero_accumulator<half, warp_mma_blocks_d>(dQdK_acc_half);

    float scale = rsqrtf(D_DIM);
    // causal
    K += blockIdx.x * BS_DIM * D_DIM;
    V += blockIdx.x * BS_DIM * D_DIM;
    F += blockIdx.x * BS_DIM;
    I += blockIdx.x * BS_DIM;
    unsigned int matrix_col = blockIdx.x * BS_DIM;
    for (unsigned int block_s = 0; block_s < blockIdx.x+1; block_s++) {
        zero_accumulator<half, warp_mma_blocks_bs>(QK_reg_half);
        zero_accumulator<half, warp_mma_blocks_bs>(dC_reg_half);

        // Load V
        blockCpy(V, qkv_shmem, BS_DIM, D_DIM);
        __syncthreads();

        half* v_pointer = qkv_shmem + (lane_id % 8) * D_DIM;
        uint32_t v_offset;
        CVTA_TO_SHARED_U32(v_pointer, v_offset);
        for (unsigned int warp_bs = 0; warp_bs < warp_mma_blocks_bs; warp_bs++) {
            for (unsigned int warp_d = 0; warp_d < warp_mma_blocks_d; warp_d++) {
                const uint32_t thread_offset = v_offset + (warp_bs * MMA_N_DIM * D_DIM + warp_d * MMA_K_DIM) * sizeof(half);
                LDMATRIX_X1(KV_reg[warp_bs][warp_d], thread_offset);
            }
        }

        // Compute dC = dH*V^T
        matrixMultBSD<warp_mma_blocks_bs, warp_mma_blocks_d>(dC_reg, dH_reg, KV_reg);

        // dC_tilde
        #pragma unroll
        for (unsigned int warp_bs = 0; warp_bs < warp_mma_blocks_bs; warp_bs++) {
            dC_reg_half[warp_bs][0] = __float2half(__half2float(dC_reg_half[warp_bs][0]) / n_reg[0] + db_reg[0]);
            dC_reg_half[warp_bs][1] = __float2half(__half2float(dC_reg_half[warp_bs][1]) / n_reg[0] + db_reg[0]);
            dC_reg_half[warp_bs][2] = __float2half(__half2float(dC_reg_half[warp_bs][2]) / n_reg[1] + db_reg[1]);
            dC_reg_half[warp_bs][3] = __float2half(__half2float(dC_reg_half[warp_bs][3]) / n_reg[1] + db_reg[1]);
        }

        // Load K, F, I
        blockCpy(K, qkv_shmem, BS_DIM, D_DIM);
        load_f_i(F, f_shmem, I, i_shmem, warp_id, lane_id);
        __syncthreads();

        half* k_pointer = qkv_shmem + (lane_id % 8) * D_DIM;
        uint32_t k_offset;
        CVTA_TO_SHARED_U32(k_pointer, k_offset);
        for (unsigned int warp_bs = 0; warp_bs < warp_mma_blocks_bs; warp_bs++) {
            for (unsigned int warp_d = 0; warp_d < warp_mma_blocks_d; warp_d++) {
                const uint32_t thread_offset = k_offset + (warp_bs * MMA_N_DIM * D_DIM + warp_d * MMA_K_DIM) * sizeof(half);
                LDMATRIX_X1(KV_reg[warp_bs][warp_d], thread_offset);
                KV_reg_half[warp_bs][warp_d][0] *= scale;
                KV_reg_half[warp_bs][warp_d][1] *= scale;
            }
        }

        // Compute QK^T
        matrixMultBSD<warp_mma_blocks_bs, warp_mma_blocks_d>(QK_reg, Q_reg, KV_reg);

        // Compute D_tilde
        f_row[0] = block_s > 0 ? f_row[0] + f_shmem[BS_DIM-1] : f_shmem[block_row_1];
        f_row[1] = block_s > 0 ? f_row[1] + f_shmem[BS_DIM-1] : f_shmem[block_row_2];
        float df_tmp[2];
        df_tmp[0] = 0.0f;
        df_tmp[1] = 0.0f;
        for (unsigned int warp_bs = 0; warp_bs < warp_mma_blocks_bs; warp_bs++) {
            unsigned int tile_col_1 = warp_bs * MMA_N_DIM + (lane_id % 4) * 2;
            unsigned int tile_col_2 = tile_col_1 + 1;
            unsigned int matrix_col_1 = matrix_col + tile_col_1;
            unsigned int matrix_col_2 = matrix_col + tile_col_2;
            float i1 = __half2float(i_shmem[tile_col_1]);
            float i2 = __half2float(i_shmem[tile_col_2]);
            float f_col_1 = f_shmem[tile_col_1];
            float f_col_2 = f_shmem[tile_col_2];

            half d1 = __float2half(expf((matrix_row_1 >= matrix_col_1 ? f_row[0] - f_col_1 + i1 : NINFINITY) - m_i_reg[0]));
            half d2 = __float2half(expf((matrix_row_1 >= matrix_col_2 ? f_row[0] - f_col_2 + i2 : NINFINITY) - m_i_reg[0]));
            half d3 = __float2half(expf((matrix_row_2 >= matrix_col_1 ? f_row[1] - f_col_1 + i1 : NINFINITY) - m_i_reg[1]));
            half d4 = __float2half(expf((matrix_row_2 >= matrix_col_2 ? f_row[1] - f_col_2 + i2 : NINFINITY) - m_i_reg[1]));

            QK_reg_half[warp_bs][0] *= d1;
            QK_reg_half[warp_bs][1] *= d2;
            QK_reg_half[warp_bs][2] *= d3;
            QK_reg_half[warp_bs][3] *= d4;
            df_tmp[0] += __half2float(QK_reg_half[warp_bs][0] * dC_reg_half[warp_bs][0] + QK_reg_half[warp_bs][1] * dC_reg_half[warp_bs][1]);
            df_tmp[1] += __half2float(QK_reg_half[warp_bs][2] * dC_reg_half[warp_bs][2] + QK_reg_half[warp_bs][3] * dC_reg_half[warp_bs][3]);

            dC_reg_half[warp_bs][0] *= d1;
            dC_reg_half[warp_bs][1] *= d2;
            dC_reg_half[warp_bs][2] *= d3;
            dC_reg_half[warp_bs][3] *= d4;
        }

        sum_reduce_groups_of_4<float>(df_tmp, lane_id);
        df_acc[0] += df_tmp[0];
        df_acc[1] += df_tmp[1];

        // Transpose K in registers
        transposeMatrix<warp_mma_blocks_bs, warp_mma_blocks_d>(KV_reg);

        // Compute dQ
        matrixMultDBS<warp_mma_blocks_bs, warp_mma_blocks_d>(dQdK_acc, dC_reg, KV_reg);

        K -= BS_DIM * D_DIM;
        V -= BS_DIM * D_DIM;
        I -= BS_DIM;
        F -= BS_DIM;
        matrix_col -= BS_DIM;
    }

    // Save dQ
    saveMatrix<warp_mma_blocks_d, MMA_K_DIM, D_DIM>(dQ, dQdK_acc_half, block_row_1, block_row_2, lane_id);

    /*
       Compute dK, dV, dI
     */
    K += (blockIdx.x + 1) * BS_DIM * D_DIM;
    V += (blockIdx.x + 1) * BS_DIM * D_DIM;
    I += (blockIdx.x + 1) * BS_DIM;
    F += (blockIdx.x + 1) * BS_DIM;

    // Load K, I, F reuse Q_reg for K
    load_f_i(F, f_shmem, I, i_shmem, warp_id, lane_id);
    loadMatrixX2<BS_DIM, WS_DIM, D_DIM, MMA_K_DIM, warp_mma_blocks_d>(K, qkv_shmem, Q_reg, warp_id, lane_id);
    f_row[0] = f_shmem[block_row_1];
    f_row[1] = f_shmem[block_row_2];
    m_i_reg[0] = __half2float(i_shmem[block_row_1]);
    m_i_reg[1] = __half2float(i_shmem[block_row_2]);
    b_di_reg[0] = 0.0f;
    b_di_reg[1] = 0.0f;

    // Load V, reuse dH_reg for V
    loadMatrixX2<BS_DIM, WS_DIM, D_DIM, MMA_K_DIM, warp_mma_blocks_d>(V, qkv_shmem, dH_reg, warp_id, lane_id);

    zero_accumulator<half, warp_mma_blocks_d>(dQdK_acc_half);
    float f_sum = 0.0f;
    matrix_col = blockIdx.x * BS_DIM;
    for (unsigned int block_s = blockIdx.x; block_s < S_DIM / BS_DIM; block_s++) {
        zero_accumulator<half, warp_mma_blocks_bs>(QK_reg_half);
        zero_accumulator<half, warp_mma_blocks_bs>(dC_reg_half);

        // Load dH, M, B
        blockCpy(dH, qkv_shmem, BS_DIM, D_DIM);
        if (warp_id < 2) {
            m_shmem[threadIdx.x] = M[threadIdx.x];
        } else {
            b_shmem[threadIdx.x % 64] = B[threadIdx.x % 64];
        }
        __syncthreads();

        half* dh_pointer = qkv_shmem + (lane_id % 8) * D_DIM;
        uint32_t dh_offset;
        CVTA_TO_SHARED_U32(dh_pointer, dh_offset);
        for (unsigned int warp_bs = 0; warp_bs < warp_mma_blocks_bs; warp_bs++) {
            for (unsigned int warp_d = 0; warp_d < warp_mma_blocks_d; warp_d++) {
                const uint32_t thread_offset = dh_offset + (warp_bs * MMA_N_DIM * D_DIM + warp_d * MMA_K_DIM) * sizeof(half);
                LDMATRIX_X1(KV_reg[warp_bs][warp_d], thread_offset);
            }
        }

        // Compute dC^T = V * dH^T
        matrixMultBSD<warp_mma_blocks_bs, warp_mma_blocks_d>(dC_reg, dH_reg, KV_reg);

        // Load Q, F, dB
        blockCpy(Q, qkv_shmem, BS_DIM, D_DIM);
        load_f_i(F, f_shmem, dB, db_shmem, warp_id, lane_id);
        __syncthreads();

        half* q_pointer = qkv_shmem + (lane_id % 8) * D_DIM;
        uint32_t q_offset;
        CVTA_TO_SHARED_U32(q_pointer, q_offset);
        for (unsigned int warp_bs = 0; warp_bs < warp_mma_blocks_bs; warp_bs++) {
            for (unsigned int warp_d = 0; warp_d < warp_mma_blocks_d; warp_d++) {
                const uint32_t thread_offset = q_offset + (warp_bs * MMA_N_DIM * D_DIM + warp_d * MMA_K_DIM) * sizeof(half);
                LDMATRIX_X1(KV_reg[warp_bs][warp_d], thread_offset);
                KV_reg_half[warp_bs][warp_d][0] *= scale;
                KV_reg_half[warp_bs][warp_d][1] *= scale;
            }
        }

        // Compute K * Q^T
        matrixMultBSD<warp_mma_blocks_bs, warp_mma_blocks_d>(QK_reg, Q_reg, KV_reg);

        // Compute D_tilde
        float di_tmp[2];
        di_tmp[0] = 0.0f;
        di_tmp[1] = 0.0f;
        for (unsigned int warp_bs = 0; warp_bs < warp_mma_blocks_bs; warp_bs++) {
            unsigned int tile_col_1 = warp_bs * MMA_N_DIM + (lane_id % 4) * 2;
            unsigned int tile_col_2 = tile_col_1 + 1;
            unsigned int matrix_col_1 = matrix_col + tile_col_1;
            unsigned int matrix_col_2 = matrix_col + tile_col_2;
            float f_col_1 = f_shmem[tile_col_1] + f_sum;
            float f_col_2 = f_shmem[tile_col_2] + f_sum;
            float m_col_1 = m_shmem[tile_col_1];
            float m_col_2 = m_shmem[tile_col_2];
            float b_col_1 = b_shmem[tile_col_1];
            float b_col_2 = b_shmem[tile_col_2];
            half db_col_1 = __float2half(db_shmem[tile_col_1]);
            half db_col_2 = __float2half(db_shmem[tile_col_2]);
            half n_col_1 = __float2half(fmaxf(fabsf(b_col_1), expf(-m_col_1)) + 1e-6);
            half n_col_2 = __float2half(fmaxf(fabsf(b_col_2), expf(-m_col_2)) + 1e-6);

            half d1 = __float2half(expf((matrix_row_1 <= matrix_col_1 ? f_col_1 - f_row[0] + m_i_reg[0] : NINFINITY) - m_col_1));
            half d2 = __float2half(expf((matrix_row_1 <= matrix_col_2 ? f_col_2 - f_row[0] + m_i_reg[0] : NINFINITY) - m_col_2));
            half d3 = __float2half(expf((matrix_row_2 <= matrix_col_1 ? f_col_1 - f_row[1] + m_i_reg[1] : NINFINITY) - m_col_1));
            half d4 = __float2half(expf((matrix_row_2 <= matrix_col_2 ? f_col_2 - f_row[1] + m_i_reg[1] : NINFINITY) - m_col_2));

            QK_reg_half[warp_bs][0] *= d1;
            QK_reg_half[warp_bs][1] *= d2;
            QK_reg_half[warp_bs][2] *= d3;
            QK_reg_half[warp_bs][3] *= d4;

            dC_reg_half[warp_bs][0] = dC_reg_half[warp_bs][0] / n_col_1 + db_col_1;
            dC_reg_half[warp_bs][1] = dC_reg_half[warp_bs][1] / n_col_2 + db_col_2;
            dC_reg_half[warp_bs][2] = dC_reg_half[warp_bs][2] / n_col_1 + db_col_1;
            dC_reg_half[warp_bs][3] = dC_reg_half[warp_bs][3] / n_col_2 + db_col_2;

            di_tmp[0] += __half2float(QK_reg_half[warp_bs][0] * dC_reg_half[warp_bs][0] + QK_reg_half[warp_bs][1] * dC_reg_half[warp_bs][1]);
            di_tmp[1] += __half2float(QK_reg_half[warp_bs][2] * dC_reg_half[warp_bs][2] + QK_reg_half[warp_bs][3] * dC_reg_half[warp_bs][3]);

            dC_reg_half[warp_bs][0] *= d1;
            dC_reg_half[warp_bs][1] *= d2;
            dC_reg_half[warp_bs][2] *= d3;
            dC_reg_half[warp_bs][3] *= d4;

            QK_reg_half[warp_bs][0] /= n_col_1;
            QK_reg_half[warp_bs][1] /= n_col_2;
            QK_reg_half[warp_bs][2] /= n_col_1;
            QK_reg_half[warp_bs][3] /= n_col_2;
        }
        sum_reduce_groups_of_4<float>(di_tmp, lane_id);
        b_di_reg[0] += di_tmp[0];
        b_di_reg[1] += di_tmp[1];

        // Transpose Q in registers
        transposeMatrix<warp_mma_blocks_bs, warp_mma_blocks_d>(KV_reg);

        // Compute dK
        matrixMultDBS<warp_mma_blocks_bs, warp_mma_blocks_d>(dQdK_acc, dC_reg, KV_reg);

        // TODO: Compute dV
        //
        //

        f_sum += f_shmem[BS_DIM-1];
        Q += BS_DIM * D_DIM;
        dH += BS_DIM * D_DIM;
        M += BS_DIM;
        B += BS_DIM;
        dB += BS_DIM;
        F += BS_DIM;
        matrix_col += BS_DIM;
    }

    // Save dK
    saveMatrix<warp_mma_blocks_d, MMA_K_DIM, D_DIM>(dK, dQdK_acc_half, block_row_1, block_row_2, lane_id);
  }


template<unsigned int NUM_THREADS, unsigned int ITEMS_PER_THREAD>
__global__ void backward_df(half* F, half* dF) {
    using BlockLoadT = cub::BlockLoad<half, NUM_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_VECTORIZE>;
    using BlockStoreT = cub::BlockStore<half, NUM_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_VECTORIZE>;
    using BlockScanT = cub::BlockScan<float, NUM_THREADS>;

    __shared__ union TempStorage
    {
        typename BlockLoadT::TempStorage load;
        typename BlockStoreT::TempStorage store;
        typename BlockScanT::TempStorage scan;
    } temp_storage;

    half thread_data[ITEMS_PER_THREAD];
    float thread_data_float[ITEMS_PER_THREAD];

    BlockLoadT(temp_storage.load).Load(F, thread_data);
    __syncthreads();
    #pragma unroll
    for(size_t i = 0; i < ITEMS_PER_THREAD; i++) thread_data_float[i] = __half2float(thread_data[i]);

    BlockScanT(temp_storage.scan).InclusiveSum(thread_data_float, thread_data_float);
    __syncthreads();

    BlockLoadT(temp_storage.load).Load(dF, thread_data);
    __syncthreads();

    #pragma unroll
    for(size_t i = 0; i < ITEMS_PER_THREAD; i++) {
        thread_data[i] = __float2half(neg_sigmoid(thread_data_float[i]) * __half2float(thread_data[i]));
    }

    BlockStoreT(temp_storage.store).Store(dF, thread_data);
}


int main(){
    constexpr unsigned int THREADS_BLOCK = 128;
    constexpr unsigned int BS_DIM = 64;
    constexpr unsigned int WS_DIM = 16;
    constexpr unsigned int MMA_M_DIM = 16;
    constexpr unsigned int MMA_N_DIM = 8;
    constexpr unsigned int MMA_K_DIM = 8;

    constexpr unsigned int S = 64*20;
    constexpr unsigned int D = 128;

    half *dH, *Q, *K, *V, *F, *I;
    dH = (half*) malloc(S * D * sizeof(half));
    Q = (half*) malloc(S * D * sizeof(half));
    K = (half*) malloc(S * D * sizeof(half));
    V = (half*) malloc(S * D * sizeof(half));
    F = (half*) malloc(S * sizeof(half));
    I = (half*) malloc(S * sizeof(half));

    srand(1678);
    fillMatrix(dH, S*D);
    fillMatrix(Q, S*D);
    fillMatrix(K, S*D);
    fillMatrix(V, S*D);
    fillMatrix(F, S);
    fillMatrix(I, S);

    half *B, *M, *dB, *dQ, *dK;
    dB = (half*) malloc(S * sizeof(half));
    dQ = (half*) malloc(S * D * sizeof(half));
    dK = (half*) malloc(S * D * sizeof(half));
    M = (half*) malloc(S * sizeof(half));
    B = (half*) malloc(S * sizeof(half));
    mlstmBackwardCpu(dB, dH, Q, K, V, dQ, dK, F, I, M, B, S, D);

    half *dev_dH, *dev_Q, *dev_K, *dev_V, *dev_dQ, *dev_dI;
    half *dev_dK, *dev_dV, *dev_F, *dev_I, *dev_dB, *dev_M, *dev_B;
    CUDA_CHECK(cudaMalloc((void**) &dev_dH, S * D * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**) &dev_Q, S * D * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**) &dev_K, S * D * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**) &dev_V, S * D * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**) &dev_dQ, S * D * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**) &dev_dK, S * D * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**) &dev_dV, S * D * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**) &dev_F, S * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**) &dev_I, S * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**) &dev_B, S * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**) &dev_M, S * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**) &dev_dB, S * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**) &dev_dI, S * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(dev_dH, dH, S * D * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_Q, Q, S * D * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_K, K, S * D * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_V, V, S * D * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_F, F, S * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_I, I, S * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_M, M, S * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_B, B, S * sizeof(half), cudaMemcpyHostToDevice));

    unsigned int shmem_size_db = (BS_DIM * D + BS_DIM) * sizeof(half) + BS_DIM * sizeof(float);
    matmul_backward_db<BS_DIM, WS_DIM, D, MMA_M_DIM, MMA_N_DIM, MMA_K_DIM>
    <<<S / BS_DIM, THREADS_BLOCK, shmem_size_db>>>(dev_dH, dev_Q, dev_K, dev_V, dev_F, dev_I, dev_M, dev_B, dev_dB, S);

    unsigned int shmem_size = (BS_DIM * D + 4 * BS_DIM) * sizeof(half) + BS_DIM * sizeof(float);
    matmul_backward<BS_DIM, WS_DIM, D, MMA_M_DIM, MMA_N_DIM, MMA_K_DIM>
    <<<S / BS_DIM, THREADS_BLOCK, shmem_size>>>(dev_dH, dev_dB, dev_Q, dev_K, dev_V, dev_dQ,
                                                dev_dK, dev_dV, dev_F, dev_I, dev_dI, dev_M, dev_B, S);

    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    half *cuda_dB, *cuda_dQ, *cuda_dK;
    cuda_dB = (half*) malloc(S * sizeof(half));
    cuda_dQ = (half*) malloc(S * D * sizeof(half));
    cuda_dK = (half*) malloc(S * D * sizeof(half));
    CUDA_CHECK(cudaMemcpy(cuda_dB, dev_dB, S * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cuda_dQ, dev_dQ, S * D * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cuda_dK, dev_dK, S * D * sizeof(half), cudaMemcpyDeviceToHost));

    printf("Check dB \n");
    check(dB, cuda_dB, S);
    printf("Check dQ \n");
    check(dQ, cuda_dQ, S*D);
    printf("Check dK \n");
    check(dK, cuda_dK, S*D);

    free(dH);
    free(Q);
    free(K);
    free(V);
    free(F);
    free(I);
    free(M);
    free(B);
    free(dB);
    free(dQ);
    free(cuda_dQ);

    CUDA_CHECK(cudaFree(dev_dH));
    CUDA_CHECK(cudaFree(dev_Q));
    CUDA_CHECK(cudaFree(dev_K));
    CUDA_CHECK(cudaFree(dev_V));
    CUDA_CHECK(cudaFree(dev_F));
    CUDA_CHECK(cudaFree(dev_I));
    CUDA_CHECK(cudaFree(dev_M));
    CUDA_CHECK(cudaFree(dev_B));
    CUDA_CHECK(cudaFree(dev_dB));
    CUDA_CHECK(cudaFree(dev_dQ));
}
