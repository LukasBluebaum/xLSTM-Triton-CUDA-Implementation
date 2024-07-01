#!/usr/bin/env python
# import os
# os.environ['TRITON_INTERPRET'] = '1'

import torch
import triton
import triton.language as tl
import math
from utils import check


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@triton.jit
def matrix_mult(x, y, B):
    return tl.dot(x, y) if B >= 16 else tl.sum(x[:, :, None] * y, 1)


# Parallelized over Batch, Head, Sequence Dimension
# TODO: Change block order, i.e., https://triton-lang.org/main/getting-started/tutorials/09-persistent-fp8-matmul.html
#       Especially the first blocks do very little work at the moment.
#       FP 16
#       Integrate o
@triton.jit
def mlstm_matmul_kernel(Q, K, V, F, I, H, NH: tl.constexpr, S: tl.constexpr, D: tl.constexpr, SB: tl.constexpr):
    # NH: Num Heads
    # S: Sequence Length
    # D: (Head) Dimension
    # SB: Sequence Block Size
    bh_id = tl.program_id(0)
    sb_id = tl.program_id(1)

    batch_id = bh_id // NH
    head_id = bh_id % NH

    batch_offset_q = batch_id * NH * S * D + head_id * S * D
    batch_offset_f = batch_id * NH * S + head_id * S
    offset_q = tl.arange(0, SB) + sb_id * SB
    offset_k = tl.arange(0, SB) + sb_id * SB
    d_range = tl.arange(0, D)

    q_range = batch_offset_q + offset_q[:, None] * D + d_range[None, :]
    q_mask = (offset_q[:, None] < S) & (d_range[None, :] < D)
    q = tl.load(Q + q_range, q_mask)
    f = tl.load(F + batch_offset_f + offset_q, offset_q < S)
    f = tl.cumsum(tl.log(tl.sigmoid(f)))

    c_acc = tl.zeros((SB, D), dtype=tl.float32)
    b_acc = tl.zeros((SB,), dtype=tl.float32)
    m_acc = tl.zeros((SB,), dtype=tl.float32) - float("inf")
    # Iterate from right to left so we can build the D matrix in the correct order
    for j in range(sb_id, -1, -1):
        kv_range = batch_offset_q + offset_k[:, None] * D + d_range[None, :]
        kv_mask = (offset_k[:, None] < S) & (d_range[None, :] < D)
        k = tl.load(K + kv_range, kv_mask) / tl.sqrt(tl.full((1,), D, dtype=tl.float32))
        v = tl.load(V + kv_range, kv_mask)
        f_next = tl.load(F + batch_offset_f + offset_k, offset_k < S)
        i = tl.load(I + batch_offset_f + offset_k, offset_k < S)

        f_next = tl.log(tl.sigmoid(f_next))
        if j == sb_id:
            f_next = tl.cumsum(f_next)
            d = f[:, None] - f_next[None, :] + i[None, :]
            mask = offset_q[:, None] >= offset_k[None, :]
            d = tl.where(mask, d, -float("inf"))
        else:
            f += tl.sum(f_next)
            f_next = tl.cumsum(f_next)
            d = f[:, None] - f_next[None, :] + i[None, :]

        m = tl.maximum(tl.max(d, 1), m_acc)
        d = tl.exp(d - m[:, None])

        c = matrix_mult(q, tl.trans(k), SB) * d
        b_acc = b_acc * tl.exp(m_acc - m) + tl.sum(c, 1)
        c = matrix_mult(c, v, SB)
        c_acc = c_acc * tl.exp(m_acc - m)[:, None] + c

        m_acc = m
        offset_k -= SB

    n = tl.maximum(tl.abs(b_acc), tl.exp(-m_acc)) + 1e-6
    h = c_acc / n[:, None]

    # TODO: Save m, b for the backward pass? Or just recompute them
    tl.store(H + q_range, h, q_mask)


def mlstm_matmul_pytorch(q, k, v, f, i):
    _, _, S, D = q.shape
    fc = torch.cumsum(torch.nn.functional.logsigmoid(f), 2)

    mask = torch.tril(torch.ones((S, S), dtype=torch.bool, device=q.device))
    d_tilde = fc[..., None] - fc[:, :, None, :]
    d_tilde = torch.where(mask[None, None, :, :,], d_tilde, -float('inf'))
    d_tilde += i[:, :, None, :]
    max_ = torch.max(d_tilde, -1)[0]
    d_prime = torch.exp(d_tilde - max_[..., None])

    c = (q @ (k.transpose(-2, -1) / math.sqrt(D))) * d_prime
    n = torch.maximum(torch.abs(torch.sum(c, -1)), torch.exp(-max_)) + 1e-6
    h = (c / n[..., None]) @ v
    return h


def mlstm_matmul(q, k, v, f, i, SB=16, num_warps=8):
    B, NH, S, D = q.shape
    h = torch.zeros((B, NH, S, D), device=q.device)

    grid = (B * NH, triton.cdiv(S, SB))
    mlstm_matmul_kernel[grid](q, k, v, f, i, h, NH, S, D, SB, num_warps=num_warps)
    return h


if __name__ == '__main__':
    BATCH = 1
    HEADS = 4
    S = 8192
    D = 1024
    SB = 8

    q = torch.randn((BATCH, HEADS, S, D), device=DEVICE, dtype=torch.float32)
    k = torch.randn((BATCH, HEADS, S, D), device=DEVICE, dtype=torch.float32)
    v = torch.randn((BATCH, HEADS, S, D), device=DEVICE, dtype=torch.float32)
    f = torch.randn((BATCH, HEADS, S), device=DEVICE, dtype=torch.float32)
    i = torch.randn((BATCH, HEADS, S), device=DEVICE, dtype=torch.float32)

    h_triton = mlstm_matmul(q, k, v, f, i, SB=SB)
    h_pytorch = mlstm_matmul_pytorch(q, k, v, f, i)
    check(h_triton, h_pytorch)
