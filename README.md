# xLSTM Triton Implementation

- [x] mLSTM scan implementation, inspired by https://srush.github.io/annotated-mamba/hard.html
- [x] mLSTM matmul implementation
- [x] mLSTM matmul backward pass
- [ ] Implement the algorithm from Mamba 2? https://arxiv.org/abs/2405.21060
- [ ] sLSTM
- [ ] CUDA version


## Installation

The scan implementation requires Triton 3.0, the matmul implementation should also work with 2.3.

Install Triton 3.0 from [source](https://github.com/triton-lang/triton) or use the instructions [here](https://srush.github.io/annotated-mamba/hard.html).