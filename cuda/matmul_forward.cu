#include "cpu_utils.cuh"
#include "gpu_utils.cuh"


template<unsigned int BS_DIM, unsigned int WS_DIM, unsigned int D_DIM, 
         unsigned int MMA_M_DIM, unsigned int MMA_N_DIM, unsigned int MMA_K_DIM>
__global__ void matmul_forward(half* Q, half* K, half* V, half* F, half* I,
                               half* H, const unsigned int S_DIM){
    /*
        BS_DIM: Number of rows a block is responsible for
        WS_DIM: Number of rows a warp is responsible for (WS_DIM = MMA_M_DIM)
        D_DIM: Hidden/Head dimension
    */

    constexpr unsigned int WARP_SIZE = 32;
    constexpr unsigned int warp_mma_blocks_d = D_DIM / MMA_K_DIM;
    constexpr unsigned int warp_mma_blocks_bs = BS_DIM / MMA_N_DIM;

    const half NINFINITY = -CUDART_INF_F;
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    // the two rows inside the block the warp is responsible for
    unsigned int block_row_1 = warp_id * WS_DIM + lane_id / 4;
    unsigned int block_row_2 = warp_id * WS_DIM + lane_id / 4 + 8;
    // the two rows inside the whole matrix the warp is responsible for
    unsigned int matrix_row_1 = blockIdx.x * BS_DIM + block_row_1;
    unsigned int matrix_row_2 = blockIdx.x * BS_DIM + block_row_2;

    Q += blockIdx.x * BS_DIM * D_DIM;
    H += blockIdx.x * BS_DIM * D_DIM;

    extern __shared__ half shmem[];
    half* qkv_shmem = shmem;
    half* i_shmem = &shmem[BS_DIM * D_DIM];
    float* f_shmem = (float*) &shmem[BS_DIM * D_DIM + BS_DIM];

    uint32_t Q_reg[warp_mma_blocks_d][2];
    uint32_t KV_reg[warp_mma_blocks_bs][warp_mma_blocks_d];
    uint32_t QK_reg[warp_mma_blocks_bs][2];
    uint32_t D_reg[warp_mma_blocks_bs][2];
    uint32_t H_acc[warp_mma_blocks_d][2];

    float f_row[2];
    f_row[0] = 0.0f;
    f_row[1] = 0.0f;
    half m_acc[2];
    m_acc[0] = NINFINITY;
    m_acc[1] = NINFINITY;
    half m_acc_buffer[2];
    m_acc_buffer[0] = NINFINITY;
    m_acc_buffer[1] = NINFINITY;
    half b_acc[2];
    b_acc[0] = 0.0;
    b_acc[1] = 0.0;
    half b_acc_buffer[2];

    half (&KV_reg_half) [warp_mma_blocks_bs][warp_mma_blocks_d][2] = reinterpret_cast<half(&)[warp_mma_blocks_bs][warp_mma_blocks_d][2]>(KV_reg);
    half (&QK_reg_half) [warp_mma_blocks_bs][4] = reinterpret_cast<half(&)[warp_mma_blocks_bs][4]>(QK_reg);
    half (&H_acc_half) [warp_mma_blocks_d][4] = reinterpret_cast<half(&)[warp_mma_blocks_d][4]>(H_acc);
    half (&D_reg_half) [warp_mma_blocks_bs][4] = reinterpret_cast<half(&)[warp_mma_blocks_bs][4]>(D_reg);
  
    zero_accumulator<half, warp_mma_blocks_d>(H_acc_half);
    // Load Q
    loadMatrixX2<BS_DIM, WS_DIM, D_DIM, MMA_K_DIM, warp_mma_blocks_d>(Q, qkv_shmem, Q_reg, warp_id, lane_id);

    half scale = hrsqrt(D_DIM);
    // causal
    K += blockIdx.x * BS_DIM * D_DIM;
    V += blockIdx.x * BS_DIM * D_DIM;
    F += blockIdx.x * BS_DIM;
    I += blockIdx.x * BS_DIM;
    unsigned int matrix_col = blockIdx.x * BS_DIM;
    for(unsigned int block_s = 0; block_s < blockIdx.x+1; block_s++) {
        zero_accumulator<half, warp_mma_blocks_bs>(QK_reg_half);

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
        load_f_i(F, f_shmem, I, i_shmem, warp_id, lane_id);
        __syncthreads();

        half* v_pointer = qkv_shmem + (lane_id % 8) * D_DIM;
        uint32_t v_offset;
        CVTA_TO_SHARED_U32(v_pointer, v_offset);
        for (unsigned int warp_bs = 0; warp_bs < warp_mma_blocks_bs; warp_bs++) {
            for (unsigned int warp_d = 0; warp_d < warp_mma_blocks_d; warp_d++) {
                const uint32_t thread_offset = v_offset + (warp_bs * MMA_N_DIM * D_DIM + warp_d * MMA_K_DIM) * sizeof(half);
                LDMATRIX_X1_TRANS(KV_reg[warp_bs][warp_d], thread_offset);
            }
        }

        // Compute D_tilde, m
        f_row[0] = block_s > 0 ? f_row[0] + f_shmem[BS_DIM-1] : f_shmem[block_row_1];
        f_row[1] = block_s > 0 ? f_row[1] + f_shmem[BS_DIM-1] : f_shmem[block_row_2];
        for (unsigned int warp_bs = 0; warp_bs < warp_mma_blocks_bs; warp_bs++) {
            unsigned int tile_col_1 = warp_bs * MMA_N_DIM + (lane_id % 4) * 2;
            unsigned int tile_col_2 = tile_col_1 + 1;
            unsigned int matrix_col_1 = matrix_col + tile_col_1;
            unsigned int matrix_col_2 = matrix_col + tile_col_2;
            half i1 = i_shmem[tile_col_1];
            half i2 = i_shmem[tile_col_2];
            half f_col_1 = __float2half(f_shmem[tile_col_1]);
            half f_col_2 = __float2half(f_shmem[tile_col_2]);
            
            D_reg_half[warp_bs][0] = (matrix_row_1 >= matrix_col_1 ? __float2half(f_row[0]) - f_col_1 + i1 : NINFINITY);
            D_reg_half[warp_bs][1] = (matrix_row_1 >= matrix_col_2 ? __float2half(f_row[0]) - f_col_2 + i2 : NINFINITY);
            D_reg_half[warp_bs][2] = (matrix_row_2 >= matrix_col_1 ? __float2half(f_row[1]) - f_col_1 + i1 : NINFINITY);
            D_reg_half[warp_bs][3] = (matrix_row_2 >= matrix_col_2 ? __float2half(f_row[1]) - f_col_2 + i2 : NINFINITY);
            m_acc_buffer[0] = __hmax(m_acc_buffer[0], __hmax(D_reg_half[warp_bs][0], D_reg_half[warp_bs][1]));
            m_acc_buffer[1] = __hmax(m_acc_buffer[1], __hmax(D_reg_half[warp_bs][2], D_reg_half[warp_bs][3]));
        }

        // Compute C_tilde, b
        b_acc_buffer[0] = 0;
        b_acc_buffer[1] = 0;
        max_reduce_groups_of_4(m_acc_buffer, lane_id);
        #pragma unroll
        for (unsigned int warp_bs = 0; warp_bs < warp_mma_blocks_bs; warp_bs++) {
            QK_reg_half[warp_bs][0] *= hexp(D_reg_half[warp_bs][0] - m_acc_buffer[0]);
            QK_reg_half[warp_bs][1] *= hexp(D_reg_half[warp_bs][1] - m_acc_buffer[0]);
            QK_reg_half[warp_bs][2] *= hexp(D_reg_half[warp_bs][2] - m_acc_buffer[1]);
            QK_reg_half[warp_bs][3] *= hexp(D_reg_half[warp_bs][3] - m_acc_buffer[1]);
            b_acc_buffer[0] += QK_reg_half[warp_bs][0] + QK_reg_half[warp_bs][1];
            b_acc_buffer[1] += QK_reg_half[warp_bs][2] + QK_reg_half[warp_bs][3];
        }
        sum_reduce_groups_of_4<half>(b_acc_buffer, lane_id);
        b_acc[0] *= hexp(m_acc[0] - m_acc_buffer[0]);
        b_acc[1] *= hexp(m_acc[1] - m_acc_buffer[1]);
        b_acc[0] += b_acc_buffer[0];
        b_acc[1] += b_acc_buffer[1];

        // Adjust max of H
        #pragma unroll
        for (unsigned int warp_d = 0; warp_d < warp_mma_blocks_d; warp_d++) {
            H_acc_half[warp_d][0] *= hexp(m_acc[0] - m_acc_buffer[0]);
            H_acc_half[warp_d][1] *= hexp(m_acc[0] - m_acc_buffer[0]);
            H_acc_half[warp_d][2] *= hexp(m_acc[1] - m_acc_buffer[1]);
            H_acc_half[warp_d][3] *= hexp(m_acc[1] - m_acc_buffer[1]);
        }

        // Compute H
        matrixMultDBS<warp_mma_blocks_bs, warp_mma_blocks_d>(H_acc, QK_reg, KV_reg);

        K -= BS_DIM * D_DIM;
        V -= BS_DIM * D_DIM;
        I -= BS_DIM;
        F -= BS_DIM;
        matrix_col -= BS_DIM;
        m_acc[0] = m_acc_buffer[0];
        m_acc[1] = m_acc_buffer[1];
    }

    half n_row_1 = __hmax(__habs(b_acc[0]), hexp(-m_acc[0])) + __float2half(1e-6);
    half n_row_2 = __hmax(__habs(b_acc[1]), hexp(-m_acc[1])) + __float2half(1e-6);
    #pragma unroll
    for (unsigned int warp_d = 0; warp_d < warp_mma_blocks_d; warp_d++) {
        H_acc_half[warp_d][0] /= n_row_1;
        H_acc_half[warp_d][1] /= n_row_1;
        H_acc_half[warp_d][2] /= n_row_2;
        H_acc_half[warp_d][3] /= n_row_2;
    } 
    // Save H
    saveMatrix<warp_mma_blocks_d, MMA_K_DIM, D_DIM>(H, H_acc_half, block_row_1, block_row_2, lane_id);
}


int main(){
    constexpr unsigned int THREADS_BLOCK = 128;
    constexpr unsigned int BS_DIM = 64;
    constexpr unsigned int WS_DIM = 16;
    constexpr unsigned int MMA_M_DIM = 16;
    constexpr unsigned int MMA_N_DIM = 8;
    constexpr unsigned int MMA_K_DIM = 8;

    constexpr unsigned int S = 64*10;
    constexpr unsigned int D = 128;

    half *Q, *K, *V, *F, *I;
    Q = (half*) malloc(S * D * sizeof(half));
    K = (half*) malloc(S * D * sizeof(half));
    V = (half*) malloc(S * D * sizeof(half));
    F = (half*) malloc(S * sizeof(half));
    I = (half*) malloc(S * sizeof(half));

    srand(100);
    fillMatrix(Q, S*D);
    fillMatrix(K, S*D);
    fillMatrix(V, S*D);
    fillMatrix(F, S);
    fillMatrix(I, S);

    half *H;
    H = (half*) malloc(S * D * sizeof(half));
    mlstmCpu(Q, K, V, F, I, H, S, D);

    half *dev_Q, *dev_K, *dev_V, *dev_F, *dev_I, *dev_H;
    CUDA_CHECK(cudaMalloc((void**) &dev_Q, S * D * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**) &dev_K, S * D * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**) &dev_V, S * D * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**) &dev_F, S * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**) &dev_I, S * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**) &dev_H, S * D * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(dev_Q, Q, S * D * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_K, K, S * D * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_V, V, S * D * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_F, F, S * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_I, I, S * sizeof(half), cudaMemcpyHostToDevice));

    unsigned int shmem_size = (BS_DIM * D + BS_DIM) * sizeof(half) + BS_DIM * sizeof(float);

    matmul_forward<BS_DIM, WS_DIM, D, MMA_M_DIM, MMA_N_DIM, MMA_K_DIM>
    <<<S / BS_DIM, THREADS_BLOCK, shmem_size>>>(dev_Q, dev_K, dev_V, dev_F, dev_I, dev_H, S);

    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    half *cudaH;
    cudaH = (half*) malloc(S * D * sizeof(half));
    CUDA_CHECK(cudaMemcpy(cudaH, dev_H, S * D * sizeof(half), cudaMemcpyDeviceToHost));
    printf("Check H \n");
    check(H, cudaH, S*D);

    free(Q);
    free(K);
    free(V);
    free(F);
    free(I);
    free(H);
    free(cudaH);

    CUDA_CHECK(cudaFree(dev_Q));
    CUDA_CHECK(cudaFree(dev_K));
    CUDA_CHECK(cudaFree(dev_V));
    CUDA_CHECK(cudaFree(dev_F));
    CUDA_CHECK(cudaFree(dev_I));
    CUDA_CHECK(cudaFree(dev_H));
}
