#pragma once


#define CVTA_TO_SHARED_U32(pointer, addr) \
    asm volatile("{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n" \
        : "=r"(addr) \
        : "l"(pointer))


#define MOVMATRIX(R0, R1) \
    asm volatile ( \
        "movmatrix.sync.aligned.m8n8.trans.b16" \
        "%0, %1;" \
        : "+r"(R0) \
        : "r"(R1) \
    )


#define LDMATRIX_X2(R0, R1, thread_offset) \
    asm volatile ( \
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 " \
        "{%0, %1}, [%2];" \
        : "=r"(R0), "=r"(R1) \
        : "r"(thread_offset) \
    )


#define LDMATRIX_X1(R0, thread_offset) \
    asm volatile ( \
        "ldmatrix.sync.aligned.m8n8.x1.shared.b16 " \
        "{%0}, [%1];" \
        : "=r"(R0) \
        : "r"(thread_offset) \
    )


#define LDMATRIX_X1_TRANS(R0, thread_offset) \
    asm volatile ( \
        "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 " \
        "{%0}, [%1];" \
        : "=r"(R0) \
        : "r"(thread_offset) \
    )


#define MMA_16_8_8_FP32ACC(RD0, RD1, RD2, RD3, RA0, RA1, RB0) \
    asm volatile ( \
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 " \
        "{%0, %1, %2, %3}, " \
        "{%4, %5}, " \
        "{%6}, " \
        "{%7, %8, %9, %10};" \
        : "+f"(RD0), "+f"(RD1), "+f"(RD2), "+f"(RD3) \
        : "r"(RA0), "r"(RA1), \
        "r"(RB0) \
        "f"(RD0), "f"(RD1), "f"(RD2), "f"(RD3) \
    )


#define MMA_16_8_8_FP16ACC(RD0, RD1, RA0, RA1, RB0) \
    asm volatile ( \
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 " \
        "{%0, %1}, " \
        "{%2, %3}, " \
        "{%4}, " \
        "{%5, %6};" \
        : "+r"(RD0), "+r"(RD1) \
        : "r"(RA0), "r"(RA1), \
        "r"(RB0) \
        "r"(RD0), "r"(RD1) \
    )
  