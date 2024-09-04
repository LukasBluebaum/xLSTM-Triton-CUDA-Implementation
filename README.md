# xLSTM Triton Implementation

- [x] mLSTM scan implementation, inspired by https://srush.github.io/annotated-mamba/hard.html
- [x] mLSTM matmul implementation
- [x] mLSTM matmul backward pass
- [ ] Implement the algorithm from Mamba 2? https://arxiv.org/abs/2405.21060
- [ ] sLSTM
- [ ] CUDA version


## Usage

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
