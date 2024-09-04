#pragma once
#include "cuda_runtime.h"
#include <cuda_fp16.h>
#include "ptx.cuh"


#define CUDART_INF_F __int_as_float(0x7f800000)


__device__ float log_sigmoid(half x) {
    return -logf(1.0f + expf(-__half2float(x)));
}


// Very simple load
// TODO: For small dimensions divide warps over rows
// TODO: Unroll + vectorize + swizzle
__device__ void blockCpy(half* src, half* dst, const unsigned int rows,
                         const unsigned int cols) {
    for (unsigned int r = 0; r < rows; r++) {
        for(unsigned int c = threadIdx.x; c < cols; c+=blockDim.x) {
            dst[r * cols + c] = src[r * cols + c];
        }
    }
}


template<unsigned int BS_DIM, unsigned int WS_DIM, unsigned int D_DIM, 
         unsigned int MMA_K_DIM, unsigned int WARP_MMA_BLOCKS_D>
__device__ void loadMatrixX2(half* gmem, half* shmem, uint32_t (&reg)[WARP_MMA_BLOCKS_D][2],
                             int warp_id, int lane_id) {
    blockCpy(gmem, shmem, BS_DIM, D_DIM);
    __syncthreads();

    half* q_pointer = shmem + (warp_id * WS_DIM + lane_id % 16) * D_DIM;
    uint32_t q_offset;
    CVTA_TO_SHARED_U32(q_pointer, q_offset);
    for (unsigned int warp_d = 0; warp_d < WARP_MMA_BLOCKS_D; warp_d++) {
        const uint32_t thread_offset = q_offset + warp_d * MMA_K_DIM * sizeof(half);
        LDMATRIX_X2(reg[warp_d][0], reg[warp_d][1], thread_offset); 
    }
}


template<unsigned int WARP_MMA_BLOCKS_BS, unsigned int WARP_MMA_BLOCKS_D>
__device__ void transposeMatrix(uint32_t (&matrix)[WARP_MMA_BLOCKS_BS][WARP_MMA_BLOCKS_D]) {
    for (unsigned int warp_bs = 0; warp_bs < WARP_MMA_BLOCKS_BS; warp_bs++) {
        for (unsigned int warp_d = 0; warp_d < WARP_MMA_BLOCKS_D; warp_d++) {
            MOVMATRIX(matrix[warp_bs][warp_d], matrix[warp_bs][warp_d]);
        }
    }
}


// Accumulator dimension is BS
template<unsigned int WARP_MMA_BLOCKS_BS, unsigned int WARP_MMA_BLOCKS_D>
__device__ void matrixMultBSD(uint32_t (&ACC)[WARP_MMA_BLOCKS_BS][2],
                           uint32_t (&A)[WARP_MMA_BLOCKS_D][2],
                           uint32_t (&B)[WARP_MMA_BLOCKS_BS][WARP_MMA_BLOCKS_D]) {
    for (unsigned int warp_d = 0; warp_d < WARP_MMA_BLOCKS_D; warp_d++) {
        for (unsigned int warp_bs = 0; warp_bs < WARP_MMA_BLOCKS_BS; warp_bs++) {
            MMA_16_8_8_FP16ACC(ACC[warp_bs][0], ACC[warp_bs][1],
                               A[warp_d][0], A[warp_d][1], B[warp_bs][warp_d]);
        }
    }
}


// Accumulator dimension is D 
template<unsigned int WARP_MMA_BLOCKS_BS, unsigned int WARP_MMA_BLOCKS_D>
__device__ void matrixMultDBS(uint32_t (&ACC)[WARP_MMA_BLOCKS_D][2],
                           uint32_t (&A)[WARP_MMA_BLOCKS_BS][2],
                           uint32_t (&B)[WARP_MMA_BLOCKS_BS][WARP_MMA_BLOCKS_D]) {
    for (unsigned int warp_d = 0; warp_d < WARP_MMA_BLOCKS_D; warp_d++) {
        for (unsigned int warp_bs = 0; warp_bs < WARP_MMA_BLOCKS_BS; warp_bs++) {
            MMA_16_8_8_FP16ACC(ACC[warp_d][0], ACC[warp_d][1],
                               A[warp_bs][0], A[warp_bs][1], B[warp_bs][warp_d]);
        }
    }
}


template<typename T, unsigned int SIZE>
__device__ void zero_accumulator(T (&acc)[SIZE][4]) {
    #pragma unroll
    for(unsigned int i = 0; i < SIZE; i++) {
        acc[i][0] = 0;
        acc[i][1] = 0;
        acc[i][2] = 0;
        acc[i][3] = 0;
    }
}


// TODO: Use stmatrix for >= sm_90 
template<unsigned int WARP_MMA_BLOCKS_D, unsigned int MMA_K_DIM, unsigned int D_DIM>
__device__ void saveMatrix(half* dst, half (&src)[WARP_MMA_BLOCKS_D][4], 
                           unsigned int block_row_1, unsigned int block_row_2,
                           unsigned int lane_id) {
    #pragma unroll
    for (unsigned int warp_d = 0; warp_d < WARP_MMA_BLOCKS_D; warp_d++) {
        unsigned int tile_col_1 = warp_d * MMA_K_DIM + (lane_id % 4) * 2;
        unsigned int tile_col_2 = tile_col_1 + 1;
        dst[block_row_1 * D_DIM + tile_col_1] = src[warp_d][0];
        dst[block_row_1 * D_DIM + tile_col_2] = src[warp_d][1];
        dst[block_row_2 * D_DIM + tile_col_1] = src[warp_d][2];
        dst[block_row_2 * D_DIM + tile_col_2] = src[warp_d][3];
    }
}


__device__ void printMatrixDevice(half* A, int M, int N) {
    if (threadIdx.x == 0) {
        for(int i = 0; i < M; i++) {
            for(int j = 0; j < N; j++) {
                printf("%f ", __half2float(A[i*N + j]));
            }
            printf("\n");
        }
    }
    __syncthreads();
}


__device__ void load_f_i(half *F, float* f_shmem, half* I, half* i_shmem,
                         unsigned int warp_id, unsigned int lane_id) {
    if (warp_id < 2) {
        float val = log_sigmoid(F[threadIdx.x]);
        for (int offset = 1; offset < 32; offset *= 2) {
            float n = __shfl_up_sync(0xFFFFFFFF, val, offset);
            if (lane_id >= offset) val += n;
        }
        f_shmem[threadIdx.x] = val;
    } else {
        i_shmem[threadIdx.x % 64] = I[threadIdx.x % 64];
    }

    __syncthreads();

    if (warp_id == 1) {
        float add_val = f_shmem[31];
        f_shmem[threadIdx.x] += add_val;
    }
}


__device__ void max_reduce_groups_of_4(half* m_acc_buffer, unsigned int lane_id) {
    int group_id = lane_id / 4;
    unsigned int group_mask = 0xF << (group_id * 4);

    m_acc_buffer[0] = __hmax(m_acc_buffer[0], __shfl_xor_sync(group_mask, m_acc_buffer[0], 1));
    m_acc_buffer[0] = __hmax(m_acc_buffer[0], __shfl_xor_sync(group_mask, m_acc_buffer[0], 2));
    m_acc_buffer[1] = __hmax(m_acc_buffer[1], __shfl_xor_sync(group_mask, m_acc_buffer[1], 1));
    m_acc_buffer[1] = __hmax(m_acc_buffer[1], __shfl_xor_sync(group_mask, m_acc_buffer[1], 2));
}


template<typename T>
__device__ void sum_reduce_groups_of_4(T* b_acc_buffer, unsigned int lane_id) {
    int group_id = lane_id / 4;
    unsigned int group_mask = 0xF << (group_id * 4);

    b_acc_buffer[0] += __shfl_xor_sync(group_mask, b_acc_buffer[0], 1);
    b_acc_buffer[0] += __shfl_xor_sync(group_mask, b_acc_buffer[0], 2);
    b_acc_buffer[1] += __shfl_xor_sync(group_mask, b_acc_buffer[1], 1);
    b_acc_buffer[1] += __shfl_xor_sync(group_mask, b_acc_buffer[1], 2);
}