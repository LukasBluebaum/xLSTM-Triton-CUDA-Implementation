#include <cute/tensor.hpp>

#include "gemm.cu"
#include "cpu_utils.cuh"


int main(int argc, char const *argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " S D" << std::endl;
        return 1;
    }

    int64_t S = atoi(argv[1]);
    int64_t D = atoi(argv[2]);
    if (D != 64 && D != 128 && D != 256 && D!= 512) {
        std::cout << "Only support head dimensions of 64, 128, 256, 512." << std::endl;
        return 1;
    }

    half *Q, *K, *V, *F, *I, *H;
    Q = (half*) malloc(S * D * sizeof(half));
    K = (half*) malloc(S * D * sizeof(half));
    V = (half*) malloc(S * D * sizeof(half));
    F = (half*) malloc(S * sizeof(half));
    I = (half*) malloc(S * sizeof(half));
    H = (half*) malloc(S * D * sizeof(half));

    srand(6);
    fillMatrix(Q, S*D);
    fillMatrix(K, S*D);
    fillMatrix(V, S*D);
    fillMatrix(F, S);
    fillMatrix(I, S);

    mlstmCpu(Q, K, V, F, I, H, S, D);

    cute::half_t *dev_Q, *dev_K, *dev_V, *dev_F, *dev_I, *dev_H;
    CUDA_CHECK(cudaMalloc((void**) &dev_Q, S * D * sizeof(cute::half_t)));
    CUDA_CHECK(cudaMalloc((void**) &dev_K, S * D * sizeof(cute::half_t)));
    CUDA_CHECK(cudaMalloc((void**) &dev_V, S * D * sizeof(cute::half_t)));
    CUDA_CHECK(cudaMalloc((void**) &dev_H, S * D * sizeof(cute::half_t)));
    CUDA_CHECK(cudaMalloc((void**) &dev_F, S * sizeof(cute::half_t)));
    CUDA_CHECK(cudaMalloc((void**) &dev_I, S * sizeof(cute::half_t)));

    CUDA_CHECK(cudaMemcpy(dev_Q, Q, S * D * sizeof(cute::half_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_K, K, S * D * sizeof(cute::half_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_V, V, S * D * sizeof(cute::half_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_F, F, S * sizeof(cute::half_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_I, I, S * sizeof(cute::half_t), cudaMemcpyHostToDevice));

    auto layout_QKV = cute::make_layout(cute::make_shape(S, D), cute::GenRowMajor{});
    auto layout_FI = cute::make_layout(cute::make_shape(S), cute::GenRowMajor{});
    auto tQ = cute::make_tensor(cute::make_gmem_ptr(dev_Q), layout_QKV);
    auto tK = cute::make_tensor(cute::make_gmem_ptr(dev_K), layout_QKV);
    auto tV = cute::make_tensor(cute::make_gmem_ptr(dev_V), layout_QKV);
    auto tH = cute::make_tensor(cute::make_gmem_ptr(dev_H), layout_QKV);
    auto tF = cute::make_tensor(cute::make_gmem_ptr(dev_F), layout_FI);
    auto tI = cute::make_tensor(cute::make_gmem_ptr(dev_I), layout_FI);

    // TODO: use define switch
    if (D == 64) {
        mlstm::mlstm<mlstm::Config<64>>(tQ, tK, tV, tH, tF, tI);
    } else if (D == 128) {
        mlstm::mlstm<mlstm::Config<128>>(tQ, tK, tV, tH, tF, tI);
    } else if (D == 256) {
        mlstm::mlstm<mlstm::Config<256>>(tQ, tK, tV, tH, tF, tI);
    } else if (D == 512) {
        mlstm::mlstm<mlstm::Config<512>>(tQ, tK, tV, tH, tF, tI);
    } else {
        std::cout << "Only support head dimensions of 64, 128, 256, 512." << std::endl;
        return 1;
    }
    cudaDeviceSynchronize();

    half* cuda_H;
    cuda_H = (half*) malloc(S * D * sizeof(half));
    CUDA_CHECK(cudaMemcpy(cuda_H, dev_H, S * D * sizeof(half), cudaMemcpyDeviceToHost));
    check(cuda_H, H, S*D);

    free(Q);
    free(K);
    free(V);
    free(F);
    free(I);
    free(H);
    free(cuda_H);

    CUDA_CHECK(cudaFree(dev_Q));
    CUDA_CHECK(cudaFree(dev_K));
    CUDA_CHECK(cudaFree(dev_V));
    CUDA_CHECK(cudaFree(dev_H));
    CUDA_CHECK(cudaFree(dev_F));
    CUDA_CHECK(cudaFree(dev_I));

    return 0;
}