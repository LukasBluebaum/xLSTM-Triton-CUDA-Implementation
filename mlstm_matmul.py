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


@triton.jit
def sign(x):
    return (x > 0).to(tl.float32) - (x < 0).to(tl.float32)


@triton.jit
def scan_add_op(x1, x2):
    return x1 + x2


# Parallelized over Batch, Head, Sequence Dimension
# TODO: Change block order, i.e., https://triton-lang.org/main/getting-started/tutorials/09-persistent-fp8-matmul.html
#       Especially the first blocks do very little work at the moment.
#       FP 16
#       Integrate o
@triton.jit
def mlstm_matmul_kernel(Q, K, V, F, I, M, B, H, NH: tl.constexpr, S: tl.constexpr, D: tl.constexpr, SB: tl.constexpr):
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
    tl.store(B + batch_offset_f + offset_q, b_acc, offset_q < S)
    tl.store(M + batch_offset_f + offset_q, m_acc, offset_q < S)


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


@triton.jit
def mlstm_matmul_kernel_backward_db(dH, Q, K, V, F, I, M, B, dB,
                                    NH: tl.constexpr,
                                    S: tl.constexpr,
                                    D: tl.constexpr,
                                    SB: tl.constexpr):
    # NH: Num Heads
    # S: Sequence Length
    # D: (Head) Dimension
    # SB: Sequence Block Size
    bh_id = tl.program_id(0)
    sb_id = tl.program_id(1)

    batch_id = bh_id // NH
    head_id = bh_id % NH

    batch_offset_dh = batch_id * NH * S * D + head_id * S * D
    batch_offset_f = batch_id * NH * S + head_id * S
    offset_dh = tl.arange(0, SB) + sb_id * SB
    offset_vk = tl.arange(0, SB) + sb_id * SB
    d_range = tl.arange(0, D)

    dh_range = batch_offset_dh + offset_dh[:, None] * D + d_range[None, :]
    dh_mask = (offset_dh[:, None] < S) & (d_range[None, :] < D)
    dh = tl.load(dH + dh_range, dh_mask)
    q = tl.load(Q + dh_range, dh_mask)
    m = tl.load(M + batch_offset_f + offset_dh, offset_dh < S)
    f = tl.load(F + batch_offset_f + offset_dh, offset_dh < S)
    f = tl.cumsum(tl.log(tl.sigmoid(f)))
    scale = tl.sqrt(tl.full((1,), D, dtype=tl.float32))

    dn_acc = tl.zeros((SB,), dtype=tl.float32)
    for j in range(sb_id, -1, -1):
        vk_range = batch_offset_dh + offset_vk[:, None] * D + d_range[None, :]
        vk_mask = (offset_vk[:, None] < S) & (d_range[None, :] < D)
        v = tl.load(V + vk_range, vk_mask)
        f_next = tl.load(F + batch_offset_f + offset_vk, offset_vk < S)
        i = tl.load(I + batch_offset_f + offset_vk, offset_vk < S)

        # TODO: Move into a function, used in 3 kernels
        f_next = tl.log(tl.sigmoid(f_next))
        if j == sb_id:
            f_next = tl.cumsum(f_next)
            d = f[:, None] - f_next[None, :] + i[None, :]
            mask = offset_dh[:, None] >= offset_vk[None, :]
            d = tl.where(mask, d, -float('inf'))
        else:
            f += tl.sum(f_next)
            f_next = tl.cumsum(f_next)
            d = f[:, None] - f_next[None, :] + i[None, :]

        d = tl.exp(d - m[:, None])
        dc = matrix_mult(dh, tl.trans(v), SB)

        k = tl.load(K + vk_range, vk_mask) / scale
        c_tilde = matrix_mult(q, tl.trans(k), SB) * d
        dn_acc += tl.sum(c_tilde * dc, 1)

        offset_vk -= SB

    b = tl.load(B + batch_offset_f + offset_dh, offset_dh < S)
    n = tl.maximum(tl.abs(b), tl.exp(-m)) + 1e-6
    dn = -dn_acc * (1 / tl.exp(tl.log(n) * 2.0))
    # Small mistake in the paper (page 27), sign(b) not sign(n)
    db = sign(b) * dn * tl.where(tl.abs(b) > tl.exp(-m), 1.0, 0.0)
    tl.store(dB + batch_offset_f + offset_dh, db, offset_dh < S)


@triton.jit
def mlstm_matmul_kernel_backward(dH, dB, Q, K, V, dQ, dK, dV, F, dF, I, dI, M, B,
                                 NH: tl.constexpr,
                                 S: tl.constexpr,
                                 D: tl.constexpr,
                                 SB: tl.constexpr):
    # NH: Num Heads
    # S: Sequence Length
    # D: (Head) Dimension
    # SB: Sequence Block Size
    bh_id = tl.program_id(0)
    sb_id = tl.program_id(1)

    batch_id = bh_id // NH
    head_id = bh_id % NH

    batch_offset_dh = batch_id * NH * S * D + head_id * S * D
    batch_offset_f = batch_id * NH * S + head_id * S
    offset_dh = tl.arange(0, SB) + sb_id * SB
    offset_vk = tl.arange(0, SB) + sb_id * SB
    d_range = tl.arange(0, D)

    dh_range = batch_offset_dh + offset_dh[:, None] * D + d_range[None, :]
    dh_mask = (offset_dh[:, None] < S) & (d_range[None, :] < D)
    dh = tl.load(dH + dh_range, dh_mask)
    m = tl.load(M + batch_offset_f + offset_dh, offset_dh < S)
    b = tl.load(B + batch_offset_f + offset_dh, offset_dh < S)
    f = tl.load(F + batch_offset_f + offset_dh, offset_dh < S)
    db = tl.load(dB + batch_offset_f + offset_dh, offset_dh < S)

    # dQ, dF
    q = tl.load(Q + dh_range, dh_mask)
    scale = tl.sqrt(tl.full((1,), D, dtype=tl.float32))
    n = tl.maximum(tl.abs(b), tl.exp(-m)) + 1e-6
    f = tl.cumsum(tl.log(tl.sigmoid(f)))
    f_low = f

    df_acc = tl.zeros((SB,), dtype=tl.float32)
    dq_acc = tl.zeros((SB, D), dtype=tl.float32)
    for j in range(sb_id, -1, -1):
        vk_range = batch_offset_dh + offset_vk[:, None] * D + d_range[None, :]
        vk_mask = (offset_vk[:, None] < S) & (d_range[None, :] < D)
        f_next = tl.load(F + batch_offset_f + offset_vk, offset_vk < S)
        i = tl.load(I + batch_offset_f + offset_vk, offset_vk < S)

        f_next = tl.log(tl.sigmoid(f_next))
        if j == sb_id:
            f_next = tl.cumsum(f_next)
            d = f[:, None] - f_next[None, :] + i[None, :]
            mask = offset_dh[:, None] >= offset_vk[None, :]
            d = tl.where(mask, d, -float('inf'))
        else:
            f += tl.sum(f_next)
            f_next = tl.cumsum(f_next)
            d = f[:, None] - f_next[None, :] + i[None, :]

        d = tl.exp(d - m[:, None])
        v = tl.load(V + vk_range, vk_mask)
        dc_tilde = matrix_mult(dh, tl.trans(v), SB) * (1 / n)[:, None] + db[:, None]

        k = tl.load(K + vk_range, vk_mask) / scale
        dq_acc += matrix_mult(dc_tilde * d, k, SB)
        c_tilde = matrix_mult(q, tl.trans(k), SB) * d
        df_acc += tl.sum(c_tilde * dc_tilde, 1)

        offset_vk -= SB

    tl.store(dQ + dh_range, dq_acc, dh_mask)

    # dK, dV, dI
    offset_q = tl.arange(0, SB) + sb_id * SB
    f = tl.zeros((1,), dtype=tl.float32)

    v = tl.load(V + dh_range, dh_mask)
    k = tl.load(K + dh_range, dh_mask)
    i = tl.load(I + batch_offset_f + offset_dh, offset_dh < S)

    dk_acc = tl.zeros((SB, D), dtype=tl.float32)
    dv_acc = tl.zeros((SB, D), dtype=tl.float32)
    di_acc = tl.zeros((SB,), dtype=tl.float32)
    for j in range(sb_id, tl.cdiv(S, SB)):
        q_range = batch_offset_dh + offset_q[:, None] * D + d_range[None, :]
        q_mask = (offset_q[:, None] < S) & (d_range[None, :] < D)
        f_next = tl.load(F + batch_offset_f + offset_q, offset_q < S)

        f_next = tl.log(tl.sigmoid(f_next))
        f_next_sum = tl.sum(f_next)
        f_next = f + tl.cumsum(f_next)
        d = f_next[None, :] - f_low[:, None] + i[:, None]
        f += f_next_sum

        if j == sb_id:
            mask = offset_dh[:, None] <= offset_q[None, :]
            d = tl.where(mask, d, -float('inf'))

        dh = tl.load(dH + q_range, q_mask)
        m = tl.load(M + batch_offset_f + offset_q, offset_q < S)
        b = tl.load(B + batch_offset_f + offset_q, offset_q < S)
        db = tl.load(dB + batch_offset_f + offset_q, offset_q < S)

        d = tl.exp(d - m[None, :])
        n = tl.maximum(tl.abs(b), tl.exp(-m)) + 1e-6
        dc_tilde_T = matrix_mult(v, tl.trans(dh), SB) * (1 / n)[None, :] + db[None, :]

        q = tl.load(Q + q_range, q_mask) / scale
        dk_acc += matrix_mult(dc_tilde_T * d, q, SB)

        c_tilde_T = matrix_mult(k, tl.trans(q), SB) * d
        dv_acc += matrix_mult(c_tilde_T / n[None, :], dh, SB)
        di_acc += tl.sum(c_tilde_T * dc_tilde_T, 1)

        offset_q += SB

    tl.store(dK + dh_range, dk_acc, dh_mask)
    tl.store(dV + dh_range, dv_acc, dh_mask)
    tl.store(dI + batch_offset_f + offset_dh, di_acc, offset_dh < S)
    tl.store(dF + batch_offset_f + offset_dh + 1, di_acc - df_acc, (offset_dh + 1) < S)


@triton.jit
def mlstm_matmul_kernel_df(dF, F, NH: tl.constexpr, S: tl.constexpr):
    bh_id = tl.program_id(0)
    batch_id = bh_id // NH
    head_id = bh_id % NH

    batch_offset_f = batch_id * NH * S + head_id * S
    offset_f = tl.arange(0, S)

    df = tl.load(dF + batch_offset_f + offset_f, offset_f < S)
    df = tl.associative_scan(df, 0, scan_add_op)

    f = tl.load(F + batch_offset_f + offset_f, offset_f < S)
    df = tl.sigmoid(-f) * df
    tl.store(dF + batch_offset_f + offset_f, df, offset_f < S)


def mlstm_matmul_backward_pytorch(dh, q, k, v, f, i):
    _, _, S, D = q.shape

    fc = torch.cumsum(torch.nn.functional.logsigmoid(f), 2)
    mask = torch.tril(torch.ones((S, S), dtype=torch.bool, device=DEVICE))
    d_tilde = fc[..., None] - fc[:, :, None, :]
    d_tilde = torch.where(mask[None, None, :, :], d_tilde, -float('inf'))
    d_tilde += i[:, :, None, :]
    m, _ = torch.max(d_tilde, dim=-1)  # (B, NH, S, 1)
    d_prime = torch.exp(d_tilde - m[..., None])

    dC = (dh @ v.transpose(-2, -1))

    c_tilde = (q @ (k.transpose(-2, -1) / math.sqrt(D))) * d_prime
    b = torch.sum(c_tilde, -1)
    n = torch.maximum(torch.abs(b), torch.exp(-m)) + 1e-6
    dn = -torch.sum((c_tilde * dC), -1) * (1 / torch.pow(n, 2))
    db = torch.sign(b) * dn * torch.where(torch.abs(b) > torch.exp(-m), 1.0, 0.0)

    c = c_tilde / n[..., None]
    dc_tilde = dC / n[..., None] + db[..., None]

    dD = (q @ (k.transpose(-2, -1) / math.sqrt(D))) * d_prime * dc_tilde

    dQ = (dc_tilde * d_prime) @ (k / math.sqrt(D))
    dK = (dc_tilde * d_prime).transpose(-2, -1) @ (q / math.sqrt(D))
    dV = c.transpose(-2, -1) @ dh
    dI = torch.sum(dD, -2)
    dF = torch.sigmoid(-f) * torch.roll(torch.cumsum(dD.sum(-2) - dD.sum(-1), -1), 1, -1)
    return dQ, dK, dV, dI, dF


class Triton_mLSTM(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, f, i, SB=16, num_warps=8):
        B, NH, S, D = q.shape
        h = torch.zeros((B, NH, S, D), device=q.device)
        m = torch.zeros((B, NH, S), device=q.device)
        b = torch.zeros((B, NH, S), device=q.device)

        grid = (B * NH, triton.cdiv(S, SB))
        mlstm_matmul_kernel[grid](q, k, v, f, i, m, b, h, NH, S, D, SB, num_warps=num_warps)
        ctx.save_for_backward(q, k, v, f, i, m, b)
        ctx.sb = SB
        return h

    @staticmethod
    def backward(ctx, dh):
        assert dh.is_contiguous()
        q, k, v, f, i, m, b = ctx.saved_tensors
        SB = ctx.sb

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        df = torch.empty_like(f)
        di = torch.empty_like(i)
        db = torch.empty_like(b)

        B, NH, S, D = q.shape

        batches = B * NH
        grid = (batches, triton.cdiv(S, SB))
        num_warps = 8
        mlstm_matmul_kernel_backward_db[grid](dh, q, k, v, f, i, m, b, db, HEADS, S, D, B, num_warps=num_warps)
        mlstm_matmul_kernel_backward[grid](dh, db, q, k, v, dq, dk, dv, f, df, i, di, m, b, HEADS, S, D, B, num_warps=num_warps)
        mlstm_matmul_kernel_df[(batches,)](df, f, HEADS, S, num_warps=num_warps)

        return dq, dk, dv, di, df, None, None


if __name__ == '__main__':
    BATCH = 1
    HEADS = 4
    S = 8192
    D = 1024
    SB = 8
    NUM_WARPS = 8

    q = torch.randn((BATCH, HEADS, S, D), device=DEVICE, dtype=torch.float32, requires_grad=True)
    k = torch.randn((BATCH, HEADS, S, D), device=DEVICE, dtype=torch.float32, requires_grad=True)
    v = torch.randn((BATCH, HEADS, S, D), device=DEVICE, dtype=torch.float32, requires_grad=True)
    f = torch.randn((BATCH, HEADS, S), device=DEVICE, dtype=torch.float32, requires_grad=True)
    i = torch.randn((BATCH, HEADS, S), device=DEVICE, dtype=torch.float32, requires_grad=True)
    dh = torch.ones((BATCH, HEADS, S, D), device=DEVICE, dtype=torch.float32)

    h_triton = Triton_mLSTM.apply(q, k, v, f, i, SB, NUM_WARPS)
    h_pytorch = mlstm_matmul_pytorch(q, k, v, f, i)
    check(h_triton, h_pytorch)

    dq, dk, dv, di, df = mlstm_matmul_backward_pytorch(dh, q, k, v, f, i)

    h_triton.backward(dh)
    check(dq, q.grad)
    check(dk, k.grad)
    check(dv, v.grad)
    check(di, i.grad)
    check(df, f.grad)
