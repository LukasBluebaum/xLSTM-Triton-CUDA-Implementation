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
def scan_op(x1, y1, x2, y2):
    z1 = x2 * x1
    z2 = x2 * y1 + y2
    return z1, z2


@triton.jit
def stabilization_scan_op(x1, y1, x2, y2):
    z1 = x2 + x1
    z2 = tl.maximum(x2 + y1, y2)
    return z1, z2


# from https://github.com/srush/annotated-mamba
@triton.jit
def roll_op(a1, b1_last, b1_cur, a2, b2_last, b2_cur):
    return a1 + a2, tl.where(a2 == 1, b1_cur, 0) + b2_last, b2_cur


@triton.jit
def roll(y, dim=0):
    _, rh2, _ = tl.associative_scan((1 + 0 * y, 0.0 * y, y), dim, roll_op)
    return rh2


# SILU seems to introduce some numerical differences compared to the pytorch implementation
@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def stabilize_fi(F, I, M, F_STABILIZED, I_STABILIZED, NH: tl.constexpr, S: tl.constexpr):
    # NH: Number of Heads
    # S: Sequence Length
    bh_id = tl.program_id(0)
    batch_id = bh_id // NH
    head_id = bh_id % NH

    s_range = tl.arange(0, S)
    batch_offset_fi = batch_id * NH * S + head_id * S

    f = tl.load(F + batch_offset_fi + s_range, s_range < S)
    i = tl.load(I + batch_offset_fi + s_range, s_range < S)

    f = tl.log(tl.sigmoid(f))
    _, m = tl.associative_scan((f, i), 0, stabilization_scan_op)
    m_shifted = roll(m, 0)
    i = tl.exp(i - m)
    f = tl.exp(f - m + m_shifted)

    tl.store(F_STABILIZED + batch_offset_fi + s_range, f, s_range < S)
    tl.store(I_STABILIZED + batch_offset_fi + s_range, i, s_range < S)
    tl.store(M + batch_offset_fi + s_range, m, s_range < S)


# Parallelized over Batch, Head, Sequence dimensions and value vectors
# Each thread block iterates over the key vectors in blocks of size VB
@triton.jit
def precompute_mlstm_triton_scan(K, V, F, I, F_REDUCED, C, N,
                                 NH: tl.constexpr,
                                 S: tl.constexpr,
                                 D: tl.constexpr,
                                 SB: tl.constexpr,
                                 VB: tl.constexpr):
    # NH: Number of Heads
    # S: Sequence Length
    # D: (Head) Dimension
    # SB: Sequence Block Size
    # VB: Value Block Size
    bh_id = tl.program_id(0)
    sb_id = tl.program_id(1)
    vb_id = tl.program_id(2)
    batch_id = bh_id // NH
    head_id = bh_id % NH
    num_sequence_blocks = tl.num_programs(1)

    v_range = tl.arange(0, VB) + vb_id * VB
    v_range_2d = v_range[None, :]
    v_range_3d = v_range[None, :, None]
    k_range = tl.arange(0, VB)
    sb_range_2d = tl.arange(0, SB)[:, None]
    sb_range_3d = tl.arange(0, SB)[:, None, None]
    sb_range_offset = tl.arange(0, SB) + sb_id * SB
    sb_range_offset_2d = sb_range_offset[:, None]

    batch_offset_fi = batch_id * NH * S + head_id * S
    batch_offset_qkv = batch_id * NH * S * D + head_id * S * D
    batch_offset_n = batch_id * NH * num_sequence_blocks * D + head_id * num_sequence_blocks * D
    batch_offset_c = batch_id * NH * num_sequence_blocks * D * D + head_id * num_sequence_blocks * D * D

    f = tl.load(F + sb_range_offset + batch_offset_fi, sb_range_offset < S)
    i = tl.load(I + sb_range_offset + batch_offset_fi, sb_range_offset < S)

    v_range_ = batch_offset_qkv + sb_range_offset_2d * D + v_range_2d
    v_mask = (sb_range_offset_2d < S) & (v_range_2d < D)
    v = tl.load(V + v_range_, v_mask)

    k_scale_factor = tl.sqrt(tl.full((1,), D, dtype=tl.float32))
    for j in tl.range(tl.cdiv(D, VB)):
        k_range_ = batch_offset_qkv + sb_range_offset_2d * D + k_range[None, :]
        k_mask = (sb_range_offset_2d < S) & (k_range[None, :] < D)
        k = tl.load(K + k_range_, k_mask) / k_scale_factor

        vk = v[:, :, None] * k[:, None, :] * i[:, None, None]
        _, c = tl.associative_scan((tl.broadcast_to(f[:, None, None], (SB, VB, VB)), vk), 0, scan_op)

        c_range = batch_offset_c + sb_range_3d * 0 + sb_id * D * D + v_range_3d * D + k_range[None, None, :]
        c_mask = (sb_range_3d == SB-1) & (v_range_3d < D) & (k_range[None, None, :] < D)
        tl.store(C + c_range, c, c_mask)

        f_reduced, n = tl.associative_scan((tl.broadcast_to(f[:, None], (SB, VB)), i[:, None] * k), 0, scan_op)
        n_range = batch_offset_n + sb_range_2d * 0 + sb_id * D + k_range[None, :]
        n_mask = (sb_range_2d == SB-1) & (k_range[None, :] < D)
        tl.store(N + n_range, n, n_mask)

        if j == 0:
            f_range = batch_offset_fi + sb_range_2d * 0 + sb_id + tl.arange(0, VB)[None, :]
            f_mask = (sb_range_2d == SB-1) & (tl.arange(0, VB)[None, :] == 0)
            tl.store(F_REDUCED + f_range, f_reduced, f_mask)

        k_range += VB


# Parallelized over Batch, Head and both dimensions of C
# TODO: Put N into its own kernel? Currently a lot of blocks are idle, because we need a smaller
#       number of blocks for N
@triton.jit
def reduce_mlstm_triton(F_REDUCED_IN, F_REDUCED_OUT, C, N,
                        NH: tl.constexpr,
                        D: tl.constexpr,
                        NSB: tl.constexpr,
                        BLOCK_SIZE: tl.constexpr):
    # NH: Number of heads
    # D: (Head) Dimension
    # NSB: Number of sequence blocks
    # BLOCK_SIZE: Block Size for C
    bh_id = tl.program_id(0)
    x_id = tl.program_id(1)
    y_id = tl.program_id(2)

    batch_id = bh_id // NH
    head_id = bh_id % NH

    nsb_range = tl.arange(0, NSB)
    nsb_range_2d = tl.arange(0, NSB)[:, None]
    nsb_range_3d = tl.arange(0, NSB)[:, None, None]
    block_range = tl.arange(0, BLOCK_SIZE)
    block_range_2d = block_range[None, :]
    x_range = block_range + x_id * BLOCK_SIZE
    x_range_3d = x_range[None, :, None]
    y_range = block_range + y_id * BLOCK_SIZE
    y_range_3d = y_range[None, None, :]

    batch_offset_f = batch_id * NH * NSB + head_id * NSB
    batch_offset_n = batch_id * NH * NSB * D + head_id * NSB * D
    batch_offset_c = batch_id * NH * NSB * D * D + head_id * NSB * D * D

    f = tl.load(F_REDUCED_IN + batch_offset_f + nsb_range)

    c_range = batch_offset_c + nsb_range_3d * D * D + x_range_3d * D + y_range_3d
    c_mask = (nsb_range_3d < NSB) & (x_range_3d < D) & (y_range_3d < D)
    c = tl.load(C + c_range, c_mask)

    f_reduced, c = tl.associative_scan((tl.broadcast_to(f[:, None, None], (NSB, BLOCK_SIZE, BLOCK_SIZE)), c),
                                       0, scan_op)

    tl.store(C + c_range, c, c_mask)
    # Only one block per batch and head needs to save f
    if x_id == 0 and y_id == 0:
        f_range = batch_offset_f + nsb_range_3d + block_range[None, :, None] * 0 + block_range[None, None, :] * 0
        f_mask = (nsb_range_3d < NSB) & (block_range[None, :, None] == 0) & (block_range[None, None, :] == 0)
        tl.store(F_REDUCED_OUT + f_range, f_reduced, f_mask)

    if x_id == 0:
        n_range = batch_offset_n + nsb_range_2d * D + block_range_2d + y_id * BLOCK_SIZE
        n_mask = (nsb_range_2d < NSB) & ((block_range_2d + y_id * BLOCK_SIZE) < D)
        n = tl.load(N + n_range, n_mask)
        _, n = tl.associative_scan((tl.broadcast_to(f[:, None], (NSB, BLOCK_SIZE)), n), 0, scan_op)
        tl.store(N + n_range, n, n_mask)


# Parallelized over Batch, Head, Sequence dimensions and value vectors
# Each thread block iterates over the key vectors in blocks of size VB
@triton.jit
def mlstm_triton_scan(Q, K, V, F, I, O, F_REDUCED, C, N, M, H,
                      NH: tl.constexpr,
                      S: tl.constexpr,
                      D: tl.constexpr,
                      SB: tl.constexpr,
                      VB: tl.constexpr):
    # NH: Number of heads
    # S: Sequence Length
    # D: (Head) Dimension
    # SB: Sequence Block Size
    # VB: Value Block Size
    bh_id = tl.program_id(0)
    sb_id = tl.program_id(1)
    vb_id = tl.program_id(2)
    num_sequence_blocks = tl.num_programs(1)
    batch_id = bh_id // NH
    head_id = bh_id % NH

    v_range = tl.arange(0, VB) + vb_id * VB
    v_range_3d = v_range[None, :, None]
    k_range = tl.arange(0, VB)
    sb_range = tl.arange(0, SB)
    sb_range_2d = tl.arange(0, SB)[:, None]
    sb_range_3d = tl.arange(0, SB)[:, None, None]
    sb_range_offset = tl.arange(0, SB) + sb_id * SB
    sb_range_offset_2d = sb_range_offset[:, None]

    batch_offset_fi = batch_id * NH * S + head_id * S
    batch_offset_f_reduced = batch_id * NH * num_sequence_blocks + head_id * num_sequence_blocks
    batch_offset_qkv = batch_id * NH * S * D + head_id * S * D
    batch_offset_n = batch_id * NH * num_sequence_blocks * D + head_id * num_sequence_blocks * D
    batch_offset_c = batch_id * NH * num_sequence_blocks * D * D + head_id * num_sequence_blocks * D * D

    f = tl.load(F + batch_offset_fi + sb_range_offset, sb_range_offset < S)
    i = tl.load(I + batch_offset_fi + sb_range_offset, sb_range_offset < S)
    f_reduced_range = batch_offset_f_reduced + sb_range * 0 + sb_id - 1
    f_reduced_mask = (sb_range == 0) & (sb_id != 0)
    f_reduced = tl.load(F_REDUCED + f_reduced_range,  f_reduced_mask, other=1.0)

    vo_range = batch_offset_qkv + sb_range_offset_2d * D + v_range[None, :]
    vo_mask = (sb_range_offset_2d < S) & (v_range[None, :] < D)
    v = tl.load(V + vo_range, vo_mask)

    normalizer = tl.zeros((SB,), dtype=tl.float32)
    h = tl.zeros((SB, VB), dtype=tl.float32)
    k_scale_factor = tl.sqrt(tl.full((1,), D, dtype=tl.float32))
    for j in tl.range(tl.cdiv(D, VB)):
        k_range_ = batch_offset_qkv + sb_range_offset_2d * D + k_range[None, :]
        k_mask = (sb_range_offset_2d < S) & (k_range[None, :] < D)
        k = tl.load(K + k_range_, k_mask) / k_scale_factor
        q = tl.load(Q + k_range_, k_mask)

        c_range = batch_offset_c + sb_range_3d + (sb_id - 1) * D * D + v_range_3d * D + k_range[None, None, :]
        c_mask = (sb_range_3d == 0) & (v_range_3d < D) & (k_range[None, None, :] < D) & (sb_id != 0)
        c_reduced = tl.load(C + c_range, c_mask, other=0)

        vk = v[:, :, None] * k[:, None, :] * i[:, None, None]
        f_tmp, vk_tmp = scan_op(f_reduced[:, None, None], c_reduced, f[:, None, None], vk)
        f_tmp = tl.broadcast_to(f_tmp, (SB, VB, VB))
        _, c = tl.associative_scan((f_tmp, vk_tmp), 0, scan_op)
        h += tl.sum(c * q[:, None, :], -1)

        n_range = batch_offset_n + sb_range_2d + (sb_id-1) * D + k_range[None, :]
        n_mask = (sb_range_2d == 0) & (k_range[None, :] < D) & (sb_id != 0)
        n = tl.load(N + n_range, n_mask, other=0.0)

        f_tmp, n_tmp = scan_op(f_reduced[:, None], n, f[:, None], k * i[:, None])
        _, n = tl.associative_scan((tl.broadcast_to(f_tmp, (SB, VB)), n_tmp), 0, scan_op)
        normalizer += tl.sum(n * q, -1)

        k_range += VB

    m = tl.load(M + batch_offset_fi + sb_range_offset, sb_range_offset < S)
    o = tl.load(O + vo_range, vo_mask)
    normalizer = tl.maximum(tl.abs(normalizer), tl.exp(-m))[:, None] + 1e-6

    h = (h / normalizer) * silu(o)
    tl.store(H + vo_range, h, vo_mask)


def mlstm_scan(q, k, v, f, i, o,
               reduce_block_size=4,
               value_block_size=4,
               num_warps=8):
    B, NH, S, D = q.shape

    if S <= 128:
        sequence_block_size = 16
    elif S <= 512:
        sequence_block_size = 64
    elif S <= 1024:
        sequence_block_size = 128
    else:
        sequence_block_size = 256

    sequence_blocks = S // sequence_block_size
    reduce_blocks = D // reduce_block_size

    f_reduced_in = torch.zeros((B, NH, sequence_blocks), device=q.device)
    f_reduced_out = torch.zeros((B, NH, sequence_blocks), device=q.device)
    n = torch.zeros((B, NH, sequence_blocks, D), device=q.device)
    c = torch.zeros((B, NH, sequence_blocks, D, D), device=q.device)
    h = torch.zeros((B, NH, S, D), device=q.device)
    # Buffers not needed, could also override f, i
    f_buffer = torch.zeros((B, NH, S), device=q.device)
    i_buffer = torch.zeros((B, NH, S), device=q.device)
    m = torch.zeros((B, NH, S), device=q.device)

    grid_reduce = (B * NH, reduce_blocks, reduce_blocks)
    grid_mlstm = (B * NH, sequence_blocks, triton.cdiv(D, value_block_size))

    # Integrate into the other kernels?
    stabilize_fi[(B * NH,)](f, i, m, f_buffer, i_buffer, NH=NH, S=S)
    precompute_mlstm_triton_scan[grid_mlstm](k, v, f_buffer, i_buffer, f_reduced_in, c, n, NH=NH, S=S, D=D,
                                             SB=sequence_block_size,
                                             VB=value_block_size,
                                             num_warps=num_warps)
    reduce_mlstm_triton[grid_reduce](f_reduced_in, f_reduced_out, c, n, NH=NH, D=D,
                                     NSB=sequence_blocks,
                                     BLOCK_SIZE=reduce_block_size,
                                     num_warps=num_warps)
    mlstm_triton_scan[grid_mlstm](q, k, v, f_buffer, i_buffer, o, f_reduced_out,
                                  c, n, m, h, NH=NH, S=S, D=D,
                                  SB=sequence_block_size,
                                  VB=value_block_size,
                                  num_warps=num_warps)
    return h


def scan_op_python(a, b):
    z1 = b[0] * a[0]
    z2 = b[0] * a[1] + b[1]
    return z1, z2


def stabilization_scan_op_python(a, b):
    z1 = b[0] + a[0]
    z2 = torch.maximum(b[0] + a[1], b[1])
    return z1, z2


# Fully unrolled pytorch implementation
def mlstm_scan_pytorch(q, k, v, f, i, o, eps=1e-6):
    B, NH, _, D = q.shape

    h = []
    for b in range(B):
        h_batch = []
        for head in range(NH):
            h_head = []
            c = (1, 0)
            n = (1, 0)
            m = (0, 0)
            for j in range(f.shape[2]):
                logf = torch.nn.functional.logsigmoid(f[b][head][j])
                m_new = (logf, i[b][head][j])
                m_new = stabilization_scan_op_python(m, m_new)
                i_stable = torch.exp(i[b][head][j] - m_new[1])
                f_stable = torch.exp(logf - m_new[1] + m[1])
                k_scaled = k[b][head][j] / math.sqrt(D)
                m = m_new

                c_new = (f_stable, i_stable * torch.outer(v[b][head][j], k_scaled))
                n_new = (f_stable, i_stable * k_scaled)
                c = scan_op_python(c, c_new)
                n = scan_op_python(n, n_new)
                max_val = torch.exp(-m[1])
                normalizer = torch.maximum(torch.abs(torch.dot(n[1], q[b][head][j])), max_val) + eps
                h_head.append(torch.nn.functional.silu(o[b][head][j]) * (c[1] @ q[b][head][j]) / normalizer)
            h_batch.append(torch.stack(h_head))
        h.append(torch.stack(h_batch))
    return torch.stack(h)


if __name__ == '__main__':
    BATCH = 1
    HEADS = 4
    S = 8192
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
    h_pytorch = mlstm_scan_pytorch(q, k, v, f, i, o)
    check(h_triton, h_pytorch, name='H')
