# xLSTM Triton/CUDA Implementation

- [x] mLSTM scan implementation, inspired by https://srush.github.io/annotated-mamba/hard.html
- [x] mLSTM matmul implementation
- [x] mLSTM matmul backward pass
- [x] simple mLSTM matmul CUDA forward pass
- [x] mLSTM matmul forward pass with Cutlass
- [ ] Implement the algorithm from Mamba 2? https://arxiv.org/abs/2405.21060
- [ ] sLSTM


## Triton Usage

The scan implementation requires Triton 3.0, the matmul implementation should also work with 2.3.

```
pip install triton
```


### [Matmul](https://github.com/LukasBluebaum/xLSTM-Triton-Implementation/blob/3a0a350fc569f78515a2e6543eff33dd4a4362d7/mlstm_matmul.py#L408) based

```python
from mlstm_matmul import Triton_mLSTM
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH = 1
HEADS = 4
S = 2048
D = 128
SB = 16
NUM_WARPS = 8

q = torch.randn((BATCH, HEADS, S, D), device=DEVICE, dtype=torch.float32, requires_grad=True)
k = torch.randn((BATCH, HEADS, S, D), device=DEVICE, dtype=torch.float32, requires_grad=True)
v = torch.randn((BATCH, HEADS, S, D), device=DEVICE, dtype=torch.float32, requires_grad=True)
f = torch.randn((BATCH, HEADS, S), device=DEVICE, dtype=torch.float32, requires_grad=True)
i = torch.randn((BATCH, HEADS, S), device=DEVICE, dtype=torch.float32, requires_grad=True)
dh = torch.randn((BATCH, HEADS, S, D), device=DEVICE, dtype=torch.float32)

h_triton = Triton_mLSTM.apply(q, k, v, f, i, SB, NUM_WARPS)
h_triton.backward(dh)
```

### [Scan](https://github.com/LukasBluebaum/xLSTM-Triton-Implementation/blob/3a0a350fc569f78515a2e6543eff33dd4a4362d7/mlstm_scan.py#L375) based (only forward pass currently)

```python
from mlstm_scan import mlstm_scan
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH = 1
HEADS = 4
S = 2048
D = 1024
VALUE_BLOCK_SIZE = 4
REDUCE_BLOCK_SIZE = 4

q = torch.randn((BATCH, HEADS, S, D), device=DEVICE, dtype=torch.float32)
k = torch.randn((BATCH, HEADS, S, D), device=DEVICE, dtype=torch.float32)
v = torch.randn((BATCH, HEADS, S, D), device=DEVICE, dtype=torch.float32)
f = torch.randn((BATCH, HEADS, S), device=DEVICE, dtype=torch.float32)
i = torch.randn((BATCH, HEADS, S), device=DEVICE, dtype=torch.float32)
o = torch.randn((BATCH, HEADS, S, D), device=DEVICE, dtype=torch.float32)

h_triton = mlstm_scan(q, k, v, f, i, o,
                      reduce_block_size=REDUCE_BLOCK_SIZE,
                      value_block_size=VALUE_BLOCK_SIZE)
```


## CUDA Usage

### [Hopper](https://github.com/LukasBluebaum/xLSTM-Triton-CUDA-Implementation/blob/main/cutlass-src-hopper/mlstm_forward.cu) sm_90a

```
git clone --recurse-submodules https://github.com/LukasBluebaum/xLSTM-Triton-CUDA-Implementation.git
cd cutlass-src-hopper
nvcc --include-path ../. --include-path ../cutlass/include \
     --generate-code=arch=compute_90a,code=[compute_90a,sm_90a] --expt-relaxed-constexpr \
     -forward-unknown-to-host-compiler -std=c++17 -O3 \
     -o mlstm_forward mlstm_forward.cu
chmod +x mlstm_forward
./mlstm_forward
```

### [Ampere](https://github.com/LukasBluebaum/xLSTM-Triton-CUDA-Implementation/blob/main/cutlass-src-ampere/mlstm_forward.cu) sm_80

```
git clone --recurse-submodules https://github.com/LukasBluebaum/xLSTM-Triton-CUDA-Implementation.git
cd cutlass-src-ampere
nvcc --include-path ../. --include-path ../cutlass/include \
     --generate-code=arch=compute_80,code=[compute_80,sm_80] \
     --expt-relaxed-constexpr -forward-unknown-to-host-compiler \
     -std=c++17 -O3 -o main main.cu
chmod +x main
./main
```

### [Turing](https://github.com/LukasBluebaum/xLSTM-Triton-CUDA-Implementation/blob/main/cuda-src/matmul_forward.cu) sm_75

```
cd cuda-src
nvcc -arch=compute_75 -code=sm_75 mlstm_forward.cu -o mlstm_forward
chmod +x mlstm_forward
./mlstm_forward
```

```cuda
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

fillMatrix(Q, S*D);
fillMatrix(K, S*D);
fillMatrix(V, S*D);
fillMatrix(F, S);
fillMatrix(I, S);

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

mlstm_forward<BS_DIM, WS_DIM, D, MMA_M_DIM, MMA_N_DIM, MMA_K_DIM>
<<<S / BS_DIM, THREADS_BLOCK, shmem_size>>>(dev_Q, dev_K, dev_V, dev_F, dev_I, dev_H, S);
```
