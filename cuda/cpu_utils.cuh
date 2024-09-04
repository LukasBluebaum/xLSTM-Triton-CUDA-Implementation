#pragma once
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <limits>
#include "cuda_runtime.h"
#include <cuda_fp16.h>


#define CUDA_CHECK(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file,
           const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    }
}


#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line) {
               cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
    }
}


void check(half* a, half* b, int size, float atol = 0.01) {
    unsigned int count = 0;
    for (int i = 0; i < size; i++) {
        if (std::abs((float) a[i] - (float) b[i]) > atol) {
            //std::cout << "Non-matching at " << i << ": " << __half2float(a[i]) << " != " << __half2float(b[i]) << std::endl;
            count++;
        }
    }
    std::cout << "Non_matching count: " << count << std::endl;
    std::cout << "Non_matching percent: " << count / (float) size << std::endl;
}


half rand_half(float a = -1.0f, float b = 1.0f) {
    float random = ((float) rand()) / (float) RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return __float2half(a + r);
}


void fillMatrix(half* A, int size) {
    for (int i = 0; i < size; i++) {
        A[i] = rand_half();
    }
}


void printMatrix(half* A, int M, int N) {
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            std::cout << __half2float(A[i*N + j]) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


void matmul(float* A, half* B, half* C, int M, int N, int K, float scale = 1.0) {
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            float acc = 0;
            for(int k = 0; k < K; k++) {
                acc += A[i*K + k] * __half2float(B[k*N + j]);
            }
            C[i*N + j] = __float2half(acc * scale);
        }
    }
}


void matmulTranspose(half* A, half* B, float* C, int M, int N, int K, float scale = 1.0) {
    for(int i = 0; i < M; i++) {
        for(int k = 0; k < K; k++) {
            float acc = 0;
            for(int j = 0; j < N; j++) {
                acc += __half2float(A[i*N + j]) * __half2float(B[k*N + j]);
            }
            C[i*K + k] = acc * scale;
        }
    }
}


void cumsum(float* A, int S) {
    for(int i = 1; i < S; i++) A[i] += A[i-1];
}


float log_sigmoid_cpu(half x) {
    return -log(1.0f + exp(-__half2float(x)));
}


void compute_d_tilde(float* d, float* m, float* F, half* I, unsigned int S) {
    for(int i = 0; i < S; i++) {
        m[i] = -std::numeric_limits<float>::max();
        for(int j = 0; j < S; j++) {
            d[i*S + j] = i >= j ? F[i] - F[j] + __half2float(I[j]) : -std::numeric_limits<float>::max();
            m[i] = max(m[i], d[i*S + j]);
        }
    }
}


void compute_b(float* b, float* QK, unsigned int S) {
    for(int i = 0; i < S; i++) {
        b[i] = 0.0f;
        for(int j = 0; j < S; j++) {
            b[i] += QK[i * S + j];
        }
    }
}


void mlstmCpu(half* Q, half* K, half* V, half* F, half* I, half* H,
               const unsigned int S, const unsigned int D) {
    float *QK, *F_buffer, *m, *d, *b;
    QK = (float*) malloc(S * S * sizeof(float));
    d = (float*) malloc(S * S * sizeof(float));
    F_buffer = (float *) malloc(S * sizeof(float));
    m = (float*) malloc(S * sizeof(float));
    b = (float*) malloc(S * sizeof(float));

    float scale = 1 / sqrt(D);
    for(int i = 0; i < S; i++) F_buffer[i] = log_sigmoid_cpu(F[i]);
    cumsum(F_buffer, S);

    // QK^T
    matmulTranspose(Q, K, QK, S, D, S, scale); 
    // D_tilde 
    compute_d_tilde(d, m, F_buffer, I, S);

    // C_tilde
    for(int i = 0; i < S; i++) {
        for(int j = 0; j < S; j++) {
            QK[i * S + j] *= exp(d[i * S + j] - m[i]);
        }
    }

    // B
    compute_b(b, QK, S);

    // C
    for(int i = 0; i < S; i++) {
        for(int j = 0; j < S; j++) {
            QK[i * S + j] /= (max(abs(b[i]), exp(-m[i]) + 1e-6));
        }
    }

    //could add multiplication by o here
    // H_tilde
    matmul(QK, V, H, S, D, S); 

    free(QK);
    free(d);
    free(F_buffer);
    free(m);
    free(b);
}
