#pragma once
#include <cute/tensor.hpp>
#include <cutlass/numeric_conversion.h>
#include <cuda_fp16.h>
#include "cpu_utils.cu"

#define CUDART_INF_F __int_as_float(0x7f800000)

namespace mlstm {

using namespace cutlass;
using namespace cute;

template <typename T, typename Layout>
using GmemTensor = Tensor<ViewEngine<gmem_ptr<T*>>, Layout>;
template <typename T, typename Layout>
using SmemTensor = Tensor<ViewEngine<smem_ptr<T*>>, Layout>;


template<int32_t DIM = 128>
struct Config {
    //constants
    using T = half_t;

    // Sequence block size
    static constexpr int64_t BLK_S = 64;
    // Head DIM
    static constexpr int64_t H_DIM = DIM;
    static constexpr int64_t MMA_M_DIM = 16;
    static constexpr int64_t MMA_N_DIM = 8;
    // 4 Warps per Block, each warp computes a (16 x Head DIM) output tile
    static constexpr int64_t NumThreads = 128;

    static constexpr int AccessSizeBits = 128;
    static constexpr int ElemsPerLoad = AccessSizeBits / sizeof_bits_v<T>;
    static constexpr int SmemAtomInner = std::min(64, static_cast<int>(H_DIM));
    static constexpr int SmemAtomOuter = ElemsPerLoad;
    static constexpr int ThreadsPerRow = SmemAtomInner / ElemsPerLoad;

    using BlockShapeQKV = Shape<Int<BLK_S>, Int<H_DIM>>;
    using BlockShapeFI = Shape<Int<BLK_S>>;

    // GMEM Layouts + Copy Atoms
    using GmemCopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, T>;
    using GmemThreadLayoutQKV = Layout<Shape<Int<NumThreads / ThreadsPerRow>, Int<ThreadsPerRow>>,
                                                Stride<Int<ThreadsPerRow>, Int<1>>>;
    using GmemValLayoutQKV = Layout<Shape<Int<1>, Int<ElemsPerLoad>>>;

    using GmemThreadLayoutF = Layout<Shape<Int<ThreadsPerRow>>,
                                                Stride<Int<1>>>;
    using GmemValLayoutF = Layout<Shape<Int<ElemsPerLoad>>>;

    using GmemCopyQKV = decltype(make_tiled_copy(GmemCopyAtom{},
                                                 GmemThreadLayoutQKV{},
                                                 GmemValLayoutQKV{}));

    using GmemCopyFI = decltype(make_tiled_copy(GmemCopyAtom{},
                                                GmemThreadLayoutF{},
                                                GmemValLayoutF{}));

    // SMEM Layouts + Copy Atoms
    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, T>;
    using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, T>;
    using SmemLayoutQKVtom = decltype(composition(Swizzle<3, 3, 3>{},
                                                    Layout<Shape<Int<SmemAtomOuter>, Int<SmemAtomInner>>,
                                                        Stride<Int<SmemAtomInner>, Int<1>>>{}));

    using SmemLayoutQKV = decltype(tile_to_shape(SmemLayoutQKVtom{}, BlockShapeQKV{}));
    using SmemLayoutQKVtransposed = decltype(
        composition(SmemLayoutQKV{}, make_layout(Shape<Int<H_DIM>, Int<BLK_S>>{}, GenRowMajor{})));
    using SmemLayoutQKVtransposedNoSwizzle = decltype(cute::detail::get_nonswizzle_portion(SmemLayoutQKVtransposed{}));

    // TODO: Swizzle
    using SmemLayoutFI = decltype(make_layout(BlockShapeFI{}, make_stride(Int<1>{})));

    // MMA Atoms
    using MmaAtom = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
    using MmaAtomLayout = Layout<Shape<Int<BLK_S / MMA_M_DIM>, Int<1>, Int<1>>>;
    using MmaPermutedShape = Tile<Int<BLK_S>, Int<16>, Int<16>>;

    using TiledMMA = TiledMMA<MmaAtom, MmaAtomLayout, MmaPermutedShape>;
    using SmemCopyQ = decltype(make_tiled_copy_A(SmemCopyAtom{}, TiledMMA{}));
    using SmemCopyK = decltype(make_tiled_copy_B(SmemCopyAtom{}, TiledMMA{}));
    using SmemCopyV = decltype(make_tiled_copy_B(SmemCopyAtomTransposed{}, TiledMMA{}));
};


template <typename To_type, typename Engine, typename Layout>
CUTE_DEVICE auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}


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


template <int size = 128, typename T, typename SrcLayout, typename DstLayout, typename TiledCopy>
CUTE_DEVICE void gmem_to_smem(
        const GmemTensor<T, SrcLayout> &src,
        const SmemTensor<T, DstLayout> &dst,
        TiledCopy tiled_copy) {
    auto thread_copy = tiled_copy.get_thread_slice(threadIdx.x % size);
    auto src_frag = thread_copy.partition_S(src);
    auto dst_frag = thread_copy.partition_D(dst);
    copy(tiled_copy, src_frag, dst_frag);
}


// TODO: Modularize
template <typename GemmConfig, typename LayoutQ, typename LayoutK, typename LayoutV, typename LayoutH,
          typename LayoutF, typename LayoutI>
__global__ void mlstm_kernel(
        GmemTensor<typename GemmConfig::T, LayoutQ> Q,
        GmemTensor<typename GemmConfig::T, LayoutK> K,
        GmemTensor<typename GemmConfig::T, LayoutV> V,
        GmemTensor<typename GemmConfig::T, LayoutH> H,
        GmemTensor<typename GemmConfig::T, LayoutF> F,
        GmemTensor<typename GemmConfig::T, LayoutI> I) {

    auto block_shape_QKV = Shape<Int<GemmConfig::BLK_S>, Int<GemmConfig::H_DIM>>{};
    auto block_shape_FI = Shape<Int<GemmConfig::BLK_S>>{};
    auto Q_block = local_tile(Q, block_shape_QKV, make_coord(blockIdx.x, 0));
    auto K_blocks = local_tile(K, block_shape_QKV, make_coord(_, 0));        
    auto V_blocks = local_tile(V, block_shape_QKV, make_coord(_, 0));
    auto F_blocks = local_tile(F, block_shape_FI, make_coord(_));
    auto I_blocks = local_tile(I, block_shape_FI, make_coord(_));
    auto H_block = local_tile(H, block_shape_QKV, make_coord(blockIdx.x, 0));

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    unsigned int block_row_1 = warp_id * 16 + lane_id / 4;
    unsigned int block_row_2 = warp_id * 16 + lane_id / 4 + 8;
    unsigned int matrix_row_1 = blockIdx.x * GemmConfig::BLK_S + block_row_1;
    unsigned int matrix_row_2 = blockIdx.x * GemmConfig::BLK_S + block_row_2;

    typename GemmConfig::SmemLayoutQKV smem_layout_QKV;
    typename GemmConfig::SmemLayoutFI smem_layout_FI;
    extern __shared__ __align__(sizeof(uint128_t)) half_t s1_data[];
    __align__(sizeof(uint128_t)) half_t* s2_data = (half_t*) &s1_data[cosize_v<decltype(smem_layout_QKV)>];
    __align__(sizeof(uint128_t)) half_t* sF_data = (half_t*) &s2_data[cosize_v<decltype(smem_layout_QKV)>];
    __align__(sizeof(uint128_t)) half_t* sI_data = (half_t*) &sF_data[cosize_v<decltype(smem_layout_FI)>];
    auto s1 = make_tensor(make_smem_ptr(s1_data), smem_layout_QKV);
    auto s2 = make_tensor(make_smem_ptr(s2_data), smem_layout_QKV);
    auto sF = make_tensor(make_smem_ptr(sF_data), smem_layout_FI);
    auto sI = make_tensor(make_smem_ptr(sI_data), smem_layout_FI);

    typename GemmConfig::GmemCopyQKV gmem_copy_QKV;
    typename GemmConfig::GmemCopyFI gmem_copy_FI;
    typename GemmConfig::SmemCopyQ smem_tiled_copy_Q;
    typename GemmConfig::SmemCopyK smem_tiled_copy_K;
    gmem_to_smem(Q_block, s1, gmem_copy_QKV);
    gmem_to_smem(K_blocks(_, _, blockIdx.x), s2, gmem_copy_QKV);

    typename GemmConfig::TiledMMA tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(threadIdx.x);
    auto H_frag = thread_mma.partition_fragment_C(H_block);
    clear(H_frag);

    Tensor f_row = make_tensor<half_t>(Shape<_2>{});
    Tensor m_acc = make_tensor<half_t>(Shape<_2>{});
    Tensor m_acc_buffer = make_tensor<half_t>(Shape<_2>{});
    Tensor b_acc = make_tensor<half_t>(Shape<_2>{});
    Tensor b_acc_buffer = make_tensor<half_t>(Shape<_2>{});
    fill(m_acc, -CUDART_INF_F);
    fill(m_acc_buffer, -CUDART_INF_F);
    clear(b_acc);

    cp_async_wait<0>();
    __syncthreads();

    auto Q_frag = thread_mma.partition_fragment_A(s1);
    auto thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(threadIdx.x);
    auto sQrQ_src = thr_copy_Q.partition_S(s1);
    auto sQrQ_dst = thr_copy_Q.retile_D(Q_frag);
    copy(smem_tiled_copy_Q, sQrQ_src, sQrQ_dst);
    __syncthreads();

    unsigned int matrix_col = blockIdx.x * GemmConfig::BLK_S;
    half_t scale = half_t(hrsqrt((half)GemmConfig::H_DIM));
    for (int k = blockIdx.x; k >= 0; k--) {
        auto shape_qk = Shape<Int<GemmConfig::BLK_S>, Int<GemmConfig::BLK_S>>{};
        auto QK_frag = partition_fragment_C(tiled_mma, shape_qk);
        clear(QK_frag);

        if (threadIdx.x < 8) {
            gmem_to_smem(F_blocks(_, k), sF, gmem_copy_FI);
        } else if (threadIdx.x < 16) {
            gmem_to_smem<8>(I_blocks(_, k), sI, gmem_copy_FI);
        }
        cp_async_fence();
        gmem_to_smem(V_blocks(_, _, k), s1, gmem_copy_QKV);
        cp_async_fence();

        auto K_frag = thread_mma.partition_fragment_B(s2);
        auto thr_copy_K = smem_tiled_copy_K.get_thread_slice(threadIdx.x);
        auto sKrK_src = thr_copy_K.partition_S(s2);
        auto sKrK_dst = thr_copy_K.retile_D(K_frag);
        copy(smem_tiled_copy_K, sKrK_src, sKrK_dst);

        gemm(tiled_mma, Q_frag, K_frag, QK_frag);
        cp_async_wait<1>();
        __syncthreads();

        if (threadIdx.x < GemmConfig::BLK_S) {
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
        cp_async_wait<0>();
        __syncthreads();

        if (k > 0) gmem_to_smem(K_blocks(_, _, k-1), s2, gmem_copy_QKV);

        Tensor QK_frag_half = convert_type<half_t>(QK_frag);
        for(int i = 0; i < size(QK_frag_half); i++) QK_frag_half(i) *= scale;

        f_row(0) = k < blockIdx.x ? f_row(0) + sF(GemmConfig::BLK_S-1) : sF(block_row_1);
        f_row(1) = k < blockIdx.x ? f_row(1) + sF(GemmConfig::BLK_S-1) : sF(block_row_2);
        constexpr unsigned int dim = GemmConfig::BLK_S / GemmConfig::MMA_N_DIM;
        Tensor D_reg = make_tensor<half_t>(Shape<_4, Int<dim>>{});
        for (size_t block_s = 0; block_s < size<2>(QK_frag_half); block_s++) {
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
        #pragma unroll
        for (size_t block_s = 0; block_s < size<2>(QK_frag_half); block_s++) {
            QK_frag_half(0, 0, block_s) *= half_t(exp(D_reg(0, block_s) - m_acc_buffer(0)));
            QK_frag_half(1, 0, block_s) *= half_t(exp(D_reg(1, block_s) - m_acc_buffer(0)));
            QK_frag_half(2, 0, block_s) *= half_t(exp(D_reg(2, block_s) - m_acc_buffer(1)));
            QK_frag_half(3, 0, block_s) *= half_t(exp(D_reg(3, block_s) - m_acc_buffer(1)));
            b_acc_buffer(0) += QK_frag_half(0, 0, block_s) + QK_frag_half(1, 0, block_s);
            b_acc_buffer(1) += QK_frag_half(2, 0, block_s) + QK_frag_half(3, 0, block_s);
        }
        sum_reduce_groups_of_4(b_acc_buffer.data(), lane_id);
        b_acc(0) *= half_t(exp(m_acc(0) - m_acc_buffer(0)));
        b_acc(1) *= half_t(exp(m_acc(1) - m_acc_buffer(1)));
        b_acc(0) += b_acc_buffer(0);
        b_acc(1) += b_acc_buffer(1);

        #pragma unroll
        for (unsigned int block_d = 0; block_d < size<2>(H_frag); block_d++) {
            H_frag(0, 0, block_d) *= exp(m_acc(0) - m_acc_buffer(0));
            H_frag(1, 0, block_d) *= exp(m_acc(0) - m_acc_buffer(0));
            H_frag(2, 0, block_d) *= exp(m_acc(1) - m_acc_buffer(1));
            H_frag(3, 0, block_d) *= exp(m_acc(1) - m_acc_buffer(1));
        }

        typename GemmConfig::SmemCopyV smem_tiled_copy_V;
        Tensor s1_T = make_tensor(s1.data(), typename GemmConfig::SmemLayoutQKVtransposed{});
        Tensor s1_T_no_swizzle = make_tensor(s1.data(), typename GemmConfig::SmemLayoutQKVtransposedNoSwizzle{});

        auto V_frag = thread_mma.partition_fragment_B(s1_T_no_swizzle);
        auto thr_copy_V = smem_tiled_copy_V.get_thread_slice(threadIdx.x);
        auto s1rV_src = thr_copy_V.partition_S(s1_T);
        auto s1rV_dst = thr_copy_V.retile_D(V_frag);
        copy(smem_tiled_copy_V, s1rV_src, s1rV_dst);

        // Reshape from accumulator layout C to layout of A
        using X = Underscore;
        auto modes = logical_divide(QK_frag_half.layout(), Shape<X, X, _2>{});
        auto a_layout = make_layout(make_layout(get<0>(modes), get<2, 0>(modes)),
                                                get<1>(modes), get<2, 1>(modes));
        auto QK_frag_half_A = make_tensor(QK_frag_half.data(), a_layout);

        gemm(tiled_mma, QK_frag_half_A, V_frag, H_frag);

        matrix_col -= GemmConfig::BLK_S;
        m_acc(0) = m_acc_buffer(0);
        m_acc(1) = m_acc_buffer(1);
        cp_async_wait<0>();
        __syncthreads();
    }

    float n_row_1 = fmaxf(abs(b_acc(0)), exp(-m_acc(0))) + 1e-6;
    float n_row_2 = fmaxf(abs(b_acc(1)), exp(-m_acc(1))) + 1e-6;
    #pragma unroll
    for(unsigned int block_d = 0; block_d < size<2>(H_frag); block_d++) {
        H_frag(0, 0, block_d) /= n_row_1;
        H_frag(1, 0, block_d) /= n_row_1;
        H_frag(2, 0, block_d) /= n_row_2;
        H_frag(3, 0, block_d) /= n_row_2;
    }

    auto H_frag_gmem = thread_mma.partition_C(H_block);
    copy(H_frag, H_frag_gmem);
    cp_async_wait<0>();
}


template <typename GemmConfig, typename LayoutQ, typename LayoutK, typename LayoutV, typename LayoutH,
          typename LayoutF, typename LayoutI>
void mlstm(GmemTensor<typename GemmConfig::T, LayoutQ> &Q,
    GmemTensor<typename GemmConfig::T, LayoutK> &K,
    GmemTensor<typename GemmConfig::T, LayoutV> &V,
    GmemTensor<typename GemmConfig::T, LayoutH> &H,
    GmemTensor<typename GemmConfig::T, LayoutF> &F,
    GmemTensor<typename GemmConfig::T, LayoutI> &I) {
    int64_t S = size<0>(Q);

    dim3 block_dim(S / GemmConfig::BLK_S);
    dim3 thread_dim(GemmConfig::NumThreads);

    constexpr int64_t shmem_size = (GemmConfig::BLK_S * GemmConfig::H_DIM + GemmConfig::BLK_S) * sizeof(typename GemmConfig::T) * 2;
    auto kernel = &mlstm_kernel<GemmConfig, LayoutQ, LayoutK, LayoutV, LayoutH, LayoutF, LayoutI>;
    if (shmem_size > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));
    }

    kernel<<<block_dim, thread_dim, shmem_size>>>(Q, K, V, H, F, I);
}

} // namespace mlstm