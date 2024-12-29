#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/cluster_launch.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include "cutlass/numeric_conversion.h"

#include "cutlass/arch/mma_sm90.h"
#include "cutlass/device_kernel.h"

#include "cuda-src/cpu_utils.cuh"


#define CUDART_INF_F __int_as_float(0x7f800000)

namespace mlstm {

using namespace cute;

template <class ElementQ,
          class ElementK,
          class ElementV,
          class ElementF,
          class ElementI,
          class SmemLayoutQ,
          class SmemLayoutK,
          class SmemLayoutV,
          class SmemLayoutF,
          class SmemLayoutI>
struct SharedStorage {
  array_aligned<ElementQ, cosize_v<SmemLayoutQ>> smem_Q;
  array_aligned<ElementK, cosize_v<SmemLayoutK>> smem_K;
  array_aligned<ElementV, cosize_v<SmemLayoutV>> smem_V;
  array_aligned<ElementF, cosize_v<SmemLayoutF>> smem_F;
  array_aligned<ElementI, cosize_v<SmemLayoutI>> smem_I;

  cutlass::arch::ClusterTransactionBarrier tma_barrier;
};


// TODO: Consolidate into utils, this is duplicated in multiple files
CUTE_DEVICE float log_sigmoid(half_t x) {
  return -logf(1.0f + expf(-float(x)));
}


CUTE_DEVICE void sum_reduce_groups_of_4(half_t* b_acc_buffer, unsigned int lane_id) {
  int group_id = lane_id / 4;
  unsigned int group_mask = 0xF << (group_id * 4);

  b_acc_buffer[0] += half_t(__shfl_xor_sync(group_mask, b_acc_buffer[0], 1));
  b_acc_buffer[0] += half_t(__shfl_xor_sync(group_mask, b_acc_buffer[0], 2));
  b_acc_buffer[1] += half_t(__shfl_xor_sync(group_mask, b_acc_buffer[1], 1));
  b_acc_buffer[1] += half_t(__shfl_xor_sync(group_mask, b_acc_buffer[1], 2));
}


CUTE_DEVICE void max_reduce_groups_of_4(half_t* m_acc_buffer, unsigned int lane_id) {
  int group_id = lane_id / 4;
  unsigned int group_mask = 0xF << (group_id * 4);

  m_acc_buffer[0] = fmaxf(m_acc_buffer[0], __shfl_xor_sync(group_mask, m_acc_buffer[0], 1));
  m_acc_buffer[0] = fmaxf(m_acc_buffer[0], __shfl_xor_sync(group_mask, m_acc_buffer[0], 2));
  m_acc_buffer[1] = fmaxf(m_acc_buffer[1], __shfl_xor_sync(group_mask, m_acc_buffer[1], 1));
  m_acc_buffer[1] = fmaxf(m_acc_buffer[1], __shfl_xor_sync(group_mask, m_acc_buffer[1], 2));
}


//TODO: Use a config class similar to the sm80 version to get rid of all these template parameters
template <class TQ, class SmemLayoutQ, class TmaQ,
          class TK, class SmemLayoutK, class TmaK,
          class TV, class SmemLayoutV, class SmemLayoutV_T, class TmaV,
          class TF, class SmemLayoutF, class TmaF,
          class TI, class SmemLayoutI, class TmaI,
          class TH, class HStride, class TiledMma_T, class TiledMma,
int bS, int D>
__global__ static
void
mlstm_kernel(int S, TQ const* Q, CUTLASS_GRID_CONSTANT TmaQ const tma_Q,
             TK const* K, CUTLASS_GRID_CONSTANT TmaK const tma_K,
             TV const* V, CUTLASS_GRID_CONSTANT TmaV const tma_V,
             TF const* F, CUTLASS_GRID_CONSTANT TmaF const tma_F,
             TI const* I, CUTLASS_GRID_CONSTANT TmaI const tma_I,
             TH      * H, HStride dH, TiledMma_T mma_T, TiledMma mma) {
  Tensor mQ = tma_Q.get_tma_tensor(make_shape(S, D));
  Tensor mK = tma_K.get_tma_tensor(make_shape(S, D));
  Tensor mV = tma_V.get_tma_tensor(make_shape(S, D));
  Tensor mF = tma_F.get_tma_tensor(make_shape(S));
  Tensor mI = tma_I.get_tma_tensor(make_shape(S));
  Tensor mH = make_tensor(make_gmem_ptr(H), make_shape(S, D), dH);

  auto block_shape_QKV = Shape<Int<bS>, Int<D>>{};
  auto block_shape_FI = Shape<Int<bS>>{};
  Tensor gQ = local_tile(mQ, block_shape_QKV, make_coord(blockIdx.x, 0));
  Tensor gK = local_tile(mK, block_shape_QKV, make_coord(_, 0));
  Tensor gV = local_tile(mV, block_shape_QKV, make_coord(_, 0));
  Tensor gF = local_tile(mF, block_shape_FI, make_coord(_));
  Tensor gI = local_tile(mI, block_shape_FI, make_coord(_));
  Tensor gH = local_tile(mH, block_shape_QKV, make_coord(blockIdx.x, 0));

  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<TQ, TK, TV, TF, TI, SmemLayoutQ, SmemLayoutK,
              SmemLayoutV, SmemLayoutF, SmemLayoutI>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sQ = make_tensor(make_smem_ptr(smem.smem_Q.data()), SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(smem.smem_K.data()), SmemLayoutK{});
  Tensor sV = make_tensor(make_smem_ptr(smem.smem_V.data()), SmemLayoutV{});
  Tensor sV_T = make_tensor(make_smem_ptr(smem.smem_V.data()), SmemLayoutV_T{});
  Tensor sF = make_tensor(make_smem_ptr(smem.smem_F.data()), SmemLayoutF{});
  Tensor sI = make_tensor(make_smem_ptr(smem.smem_I.data()), SmemLayoutI{});

  auto [tQgQ, tQsQ] = tma_partition(tma_Q, Int<0>{}, Layout<_1>{},
                                    group_modes<0,2>(sQ), group_modes<0,2>(gQ));
  auto [tKgK, tKsK] = tma_partition(tma_K, Int<0>{}, Layout<_1>{},
                                    group_modes<0,2>(sK), group_modes<0,2>(gK));
  auto [tVgV, tVsV] = tma_partition(tma_V, Int<0>{}, Layout<_1>{},
                                    group_modes<0,2>(sV), group_modes<0,2>(gV));
  auto [tFgF, tFsF] = tma_partition(tma_F, Int<0>{}, Layout<_1>{},
                                    group_modes<0,1>(sF), group_modes<0,1>(gF));
  auto [tIgI, tIsI] = tma_partition(tma_I, Int<0>{}, Layout<_1>{},
                                    group_modes<0,1>(sI), group_modes<0,1>(gI));

  constexpr int bytes_k = CUTE_STATIC_V(size<0>(tKsK)) * sizeof(TK);
  constexpr int bytes_f_i = CUTE_STATIC_V(size<0>(tFsF)) * sizeof(TF) +
                            CUTE_STATIC_V(size<0>(tIsI)) * sizeof(TI);

  using BarrierType = cutlass::arch::ClusterTransactionBarrier::ValueType;
  auto &mbarrier = smem.tma_barrier;
  unsigned int phase = 0;
  if (threadIdx.x == 0) {
    mbarrier.init(1);
    mbarrier.arrive_and_expect_tx(bytes_k + bytes_k);
    copy(tma_Q.with(reinterpret_cast<BarrierType &>(mbarrier)), tQgQ, tQsQ(_,0));
    copy(tma_K.with(reinterpret_cast<BarrierType &>(mbarrier)), tKgK(_,blockIdx.x), tKsK(_,0));
  }

  ThrMMA thr_mma_T = mma_T.get_thread_slice(threadIdx.x);
  Tensor tCsQ = thr_mma_T.partition_A(sQ);
  Tensor tCsK = thr_mma_T.partition_B(sK);

  Tensor tCrQ = thr_mma_T.make_fragment_A(tCsQ);
  Tensor tCrK = thr_mma_T.make_fragment_B(tCsK);

  ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
  Tensor tCgH = thr_mma.partition_C(gH);
  Tensor tCrH = thr_mma.make_fragment_C(tCgH);
  clear(tCrH);

  const unsigned int warp_id = threadIdx.x / 32;
  const unsigned int lane_id = threadIdx.x % 32;
  unsigned int block_row_1 = warp_id * 16 + lane_id / 4;
  unsigned int block_row_2 = warp_id * 16 + lane_id / 4 + 8;
  unsigned int matrix_row_1 = blockIdx.x * bS + block_row_1;
  unsigned int matrix_row_2 = blockIdx.x * bS + block_row_2;
  unsigned int matrix_col = blockIdx.x * bS;
  half_t scale = half_t(hrsqrt((half)D));

  Tensor f_row = make_tensor<half_t>(Shape<_2>{});
  Tensor m_acc = make_tensor<half_t>(Shape<_2>{});
  Tensor m_acc_buffer = make_tensor<half_t>(Shape<_2>{});
  Tensor b_acc = make_tensor<half_t>(Shape<_2>{});
  Tensor b_acc_buffer = make_tensor<half_t>(Shape<_2>{});
  fill(m_acc, -CUDART_INF_F);
  fill(m_acc_buffer, -CUDART_INF_F);
  clear(b_acc);

  __syncthreads();
  mbarrier.wait(phase++);

  //TODO: Use producers/consumers
  CUTE_NO_UNROLL
  for(int k_tile = blockIdx.x; k_tile >= 0; k_tile--) {
    auto shape_qk = Shape<Int<bS>, Int<bS>>{};
    auto tCrQK = partition_fragment_C(mma_T, shape_qk);
    clear(tCrQK);

    if (threadIdx.x == 0) {
      mbarrier.arrive_and_expect_tx(bytes_f_i);
      copy(tma_F.with(reinterpret_cast<BarrierType &>(mbarrier)), tFgF(_,k_tile), tFsF(_,0));
      copy(tma_I.with(reinterpret_cast<BarrierType &>(mbarrier)), tIgI(_,k_tile), tIsI(_,0));
    }
    __syncthreads();
    mbarrier.wait(phase++);

    if (threadIdx.x == 0) {
      mbarrier.arrive_and_expect_tx(bytes_k);
      copy(tma_V.with(reinterpret_cast<BarrierType &>(mbarrier)), tVgV(_,k_tile), tVsV(_,0));
    }

    warpgroup_arrive();
    gemm(mma_T, tCrQ(_,_,_,0), tCrK(_,_,_,0), tCrQK);
    warpgroup_commit_batch();

    // Remove this and fuse the cumulative sum in the projection before?
    if (threadIdx.x < bS) {
      float val = log_sigmoid(sF(threadIdx.x)); 
      int lane_id = threadIdx.x % 32;
      for (int offset = 1; offset < 32; offset *= 2) {
        float n = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if (lane_id >= offset) val += n;
      }
      sF(threadIdx.x) = half_t(val);
    }
    __syncthreads();
    if ((threadIdx.x / 32) == 1) sF(threadIdx.x) += sF(31);
    __syncthreads();
    mbarrier.wait(phase++);

    f_row(0) = k_tile < blockIdx.x ? f_row(0) + sF(bS-1) : sF(block_row_1);
    f_row(1) = k_tile < blockIdx.x ? f_row(1) + sF(bS-1) : sF(block_row_2);
    Tensor D_reg = make_tensor<half_t>(Shape<_4, Int<size<0,2>(tCrQK)>>{});
    CUTE_UNROLL
    for (size_t block_s = 0; block_s < size<0,2>(tCrQK); block_s++) {
      unsigned int tile_col_1 = block_s * 8 + (lane_id % 4) * 2;
      unsigned int tile_col_2 = tile_col_1 + 1;
      unsigned int matrix_col_1 = matrix_col + tile_col_1;
      unsigned int matrix_col_2 = matrix_col + tile_col_2;
      half_t i1 = sI(tile_col_1);
      half_t i2 = sI(tile_col_2);
      half_t f_col_1 = sF(tile_col_1);
      half_t f_col_2 = sF(tile_col_2);

      D_reg(0, block_s) = (matrix_row_1 >= matrix_col_1 ? f_row(0) - f_col_1 + i1 : -CUDART_INF_F);
      D_reg(1, block_s) = (matrix_row_1 >= matrix_col_2 ? f_row(0) - f_col_2 + i2 : -CUDART_INF_F);
      D_reg(2, block_s) = (matrix_row_2 >= matrix_col_1 ? f_row(1) - f_col_1 + i1 : -CUDART_INF_F);
      D_reg(3, block_s) = (matrix_row_2 >= matrix_col_2 ? f_row(1) - f_col_2 + i2 : -CUDART_INF_F);
      m_acc_buffer(0) = fmaxf(m_acc_buffer(0), fmaxf(D_reg(0, block_s), D_reg(1, block_s)));
      m_acc_buffer(1) = fmaxf(m_acc_buffer(1), fmaxf(D_reg(2, block_s), D_reg(3, block_s)));
    }
    max_reduce_groups_of_4(m_acc_buffer.data(), lane_id);
    clear(b_acc_buffer);

    warpgroup_wait<0>();
    if (k_tile > 0 && threadIdx.x == 0) {
      mbarrier.arrive_and_expect_tx(bytes_k);
      copy(tma_K.with(reinterpret_cast<BarrierType &>(mbarrier)), tKgK(_,k_tile-1), tKsK(_,0));
    }
    CUTE_UNROLL
    for (int i = 0; i < size(tCrQK); i++) tCrQK(i) = tCrQK(i) * scale;

    CUTE_UNROLL
    for (size_t block_s = 0; block_s < size<0,2>(tCrQK); block_s++) {
      tCrQK(make_coord(0, 0, block_s), 0, 0) *= half_t(exp(D_reg(0, block_s) - m_acc_buffer(0)));
      tCrQK(make_coord(1, 0, block_s), 0, 0) *= half_t(exp(D_reg(1, block_s) - m_acc_buffer(0)));
      tCrQK(make_coord(0, 1, block_s), 0, 0) *= half_t(exp(D_reg(2, block_s) - m_acc_buffer(1)));
      tCrQK(make_coord(1, 1, block_s), 0, 0) *= half_t(exp(D_reg(3, block_s) - m_acc_buffer(1)));
      b_acc_buffer(0) += tCrQK(make_coord(0, 0, block_s), 0, 0) + tCrQK(make_coord(1, 0, block_s), 0, 0);
      b_acc_buffer(1) += tCrQK(make_coord(0, 1, block_s), 0, 0) + tCrQK(make_coord(1, 1, block_s), 0, 0);
    }
    sum_reduce_groups_of_4(b_acc_buffer.data(), lane_id);
    b_acc(0) *= half_t(exp(m_acc(0) - m_acc_buffer(0)));
    b_acc(1) *= half_t(exp(m_acc(1) - m_acc_buffer(1)));
    b_acc(0) += b_acc_buffer(0);
    b_acc(1) += b_acc_buffer(1);

    CUTE_UNROLL
    for (unsigned int mma_n = 0; mma_n < size<2>(tCrH); mma_n++) {
      CUTE_UNROLL
      for (unsigned int block_d = 0; block_d < size<0,2>(tCrH); block_d++) {
        tCrH(make_coord(0, 0, block_d), 0, mma_n) *= half_t(exp(m_acc(0) - m_acc_buffer(0)));
        tCrH(make_coord(1, 0, block_d), 0, mma_n) *= half_t(exp(m_acc(0) - m_acc_buffer(0)));
        tCrH(make_coord(0, 1, block_d), 0, mma_n) *= half_t(exp(m_acc(1) - m_acc_buffer(1)));
        tCrH(make_coord(1, 1, block_d), 0, mma_n) *= half_t(exp(m_acc(1) - m_acc_buffer(1)));
      }
    }

    auto acc_layout = tCrQK.layout();
    auto modes = logical_divide(get<0>(acc_layout), Shape<X, X, _2>{});
    auto a_layout = make_layout(make_layout(get<0>(modes), get<1>(modes), get<2, 0>(modes)),
                                get<1>(acc_layout), make_layout(get<2, 1>(modes), get<2>(acc_layout)));
    auto tCrQK_T = make_tensor(tCrQK.data(), a_layout);

    Tensor tCsV = thr_mma.partition_B(sV_T);
    Tensor tCrV = thr_mma.make_fragment_B(tCsV);

    warpgroup_arrive();
    gemm(mma, tCrQK_T, tCrV(_,_,_,0), tCrH);
    warpgroup_commit_batch();
    warpgroup_wait<0>();

    matrix_col -= bS;
    m_acc(0) = m_acc_buffer(0);
    m_acc(1) = m_acc_buffer(1);
    if (k_tile > 0) {
      mbarrier.wait(phase++);
    }
  }

  half_t n_row_1 = half_t(fmaxf(abs(b_acc(0)), exp(-m_acc(0))) + 1e-6);
  half_t n_row_2 = half_t(fmaxf(abs(b_acc(1)), exp(-m_acc(1))) + 1e-6);
  CUTE_UNROLL
  for (unsigned int mma_n = 0; mma_n < size<2>(tCrH); mma_n++) {
    CUTE_UNROLL
    for(unsigned int block_d = 0; block_d < size<0,2>(tCrH); block_d++) {
      tCrH(make_coord(0, 0, block_d), 0, mma_n) /= n_row_1;
      tCrH(make_coord(1, 0, block_d), 0, mma_n) /= n_row_1;
      tCrH(make_coord(0, 1, block_d), 0, mma_n) /= n_row_2;
      tCrH(make_coord(1, 1, block_d), 0, mma_n) /= n_row_2;
    }
  }

  // TODO: Vectorize Store
  CUTE_UNROLL
  for (int i = 0; i < size(tCrH); ++i) {
    tCgH(i) = tCrH(i);
  }
}


template <int D = 64, class TQ, class TK, class TV, class TF, class TI, class TH>
void
mlstm(int S, TQ const* Q, TK const* K, TV const* V,
      TF const* F, TI const* I, TH *H, cudaStream_t stream = 0) {

  auto dQ = make_stride(D, Int<1>{});
  auto dK = make_stride(D, Int<1>{});
  auto dV = make_stride(D, Int<1>{});
  auto dF = make_stride(Int<1>{});
  auto dI = make_stride(Int<1>{});
  auto dH = make_stride(D, Int<1>{});

  auto bS = Int<64>{};
  auto bD = Int<D>{};
  auto bP = Int<1>{};

  auto sQ = tile_to_shape(GMMA::Layout_K_SW128_Atom<TQ>{}, make_shape(bS, bD, Int<1>{}));
  auto sK = tile_to_shape(GMMA::Layout_K_SW128_Atom<TK>{}, make_shape(bS, bD, bP));
  auto sV = tile_to_shape(GMMA::Layout_K_SW128_Atom<TV>{}, make_shape(bS, bD, bP));
  auto sV_T = composition(sV, make_ordered_layout(make_shape(Int<bD>{}, Int<bS>{}, Int<bP>{}), Step<_2, _1, _3>{}));
  // swizzle
  auto sF = tile_to_shape(Layout<Shape<Int<bS>>, Stride<Int<1>>>{}, make_shape(bS, bP));
  auto sI = tile_to_shape(Layout<Shape<Int<bS>>, Stride<Int<1>>>{}, make_shape(bS, bP));

  TiledMMA tiled_mma_T = make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});
  TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F16F16F16_RS<GMMA::Major::K, GMMA::Major::MN>{});

  Tensor mQ = make_tensor(Q, make_shape(S, D), dQ);
  Tensor mK = make_tensor(K, make_shape(S, D), dK);
  Tensor mV = make_tensor(V, make_shape(S, D), dV);
  Tensor mF = make_tensor(F, make_shape(S), dF);
  Tensor mI = make_tensor(I, make_shape(S), dI);

  Copy_Atom tmaQ = make_tma_atom(SM90_TMA_LOAD{}, mQ, sQ(_,_,0), make_shape(bS, bD));
  Copy_Atom tmaK = make_tma_atom(SM90_TMA_LOAD{}, mK, sK(_,_,0), make_shape(bS, bD));
  Copy_Atom tmaV = make_tma_atom(SM90_TMA_LOAD{}, mV, sV(_,_,0), make_shape(bS, bD));
  Copy_Atom tmaF = make_tma_atom(SM90_TMA_LOAD{}, mF, sF(_, 0), make_shape(bS));
  Copy_Atom tmaI = make_tma_atom(SM90_TMA_LOAD{}, mI, sI(_, 0), make_shape(bS));

  int smem_size = int(sizeof(SharedStorage<TQ, TK, TV, TF, TI, decltype(sQ), decltype(sK), 
            decltype(sV), decltype(sF), decltype(sI)>));
  dim3 dimBlock(size(tiled_mma_T));
  // Use 2 producer groups per cluster to maximize register usage
  dim3 dimCluster(1, 1, 1);
  dim3 dimGrid(round_up(size(ceil_div(S, bS)), dimCluster.x));
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

  void const* kernel_ptr = reinterpret_cast<void const*>(
                                              &mlstm_kernel<TQ, decltype(sQ), decltype(tmaQ),
                                              TK, decltype(sK), decltype(tmaK),
                                              TV, decltype(sV), decltype(sV_T), decltype(tmaV),
                                              TF, decltype(sF), decltype(tmaF),
                                              TI, decltype(sI), decltype(tmaI),
                                              TH, decltype(dH), decltype(tiled_mma_T),
                                              decltype(tiled_mma), bS, bD>);

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
    kernel_ptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smem_size));

  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr,
                                                             S, Q, tmaQ,
                                                             K, tmaK,
                                                             V, tmaV,
                                                             F, tmaF,
                                                             I, tmaI,
                                                             H, dH, tiled_mma_T, tiled_mma);
  CUTE_CHECK_LAST();

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Error: Failed at kernel Launch" << std::endl;
  }
}

} // namespace mlstm


int main(int argc, char** argv) {
  cudaDeviceProp props;
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (props.major != 9) {
    std::cout << "This example requires NVIDIA's Hopper Architecture GPU with compute capability 90a\n" << std::endl;
    return 0;
  }

  int S = 64;
  int D = 64;
  if (argc >= 2) sscanf(argv[1], "%d", &S);
  if (argc >= 3) sscanf(argv[2], "%d", &D);


  using TQ = cute::half_t;
  using TK = cute::half_t;
  using TV = cute::half_t;
  using TH = cute::half_t;
  using TF = cute::half_t;
  using TI = cute::half_t;
  using TI = cute::half_t;

  thrust::host_vector<TQ> h_Q(S*D);
  thrust::host_vector<TK> h_K(S*D);
  thrust::host_vector<TV> h_V(S*D);
  thrust::host_vector<TF> h_F(S);
  thrust::host_vector<TI> h_I(S);
  thrust::host_vector<TH> h_H(S*D);

  fillMatrix((half*) h_Q.data(), S*D);
  fillMatrix((half*) h_K.data(), S*D);
  fillMatrix((half*) h_V.data(), S*D);
  fillMatrix((half*) h_F.data(), S);
  fillMatrix((half*) h_I.data(), S);
  for (int j = 0; j < S*D; ++j) h_H[j] = TH(0);

  thrust::device_vector<TQ> d_Q = h_Q;
  thrust::device_vector<TK> d_K = h_K;
  thrust::device_vector<TV> d_V = h_V;
  thrust::device_vector<TF> d_F = h_F;
  thrust::device_vector<TI> d_I = h_I;
  thrust::device_vector<TH> d_H = h_H;

  // TODO: Use define switch
  if (D == 64) {
    mlstm::mlstm<64>(S, d_Q.data().get(), d_K.data().get(), d_V.data().get(),
                     d_F.data().get(), d_I.data().get(), d_H.data().get());
  } else if (D == 128) {
    mlstm::mlstm<128>(S, d_Q.data().get(), d_K.data().get(), d_V.data().get(),
                      d_F.data().get(), d_I.data().get(), d_H.data().get());
  } else if (D == 256) {
    mlstm::mlstm<256>(S, d_Q.data().get(), d_K.data().get(), d_V.data().get(),
                      d_F.data().get(), d_I.data().get(), d_H.data().get());
  } else if (D == 512) {
    mlstm::mlstm<512>(S, d_Q.data().get(), d_K.data().get(), d_V.data().get(),
                      d_F.data().get(), d_I.data().get(), d_H.data().get());
  } else {
    std::cout << "Only supports head dimensions of 64, 128, 256, 512." << std::endl;
    return 1;
  }
  thrust::host_vector<TH> cute_result = d_H;

  CUTE_CHECK_LAST();
  cudaDeviceSynchronize();
  half* H;
  H = (half*) malloc(S * D * sizeof(half));

  mlstmCpu((half*) h_Q.data(), (half*) h_K.data(), (half*) h_V.data(),
           (half*) h_F.data(), (half*) h_I.data(), H, S, D);  
  check(H, (half*) cute_result.data(), S*D);

  return 0;
}
