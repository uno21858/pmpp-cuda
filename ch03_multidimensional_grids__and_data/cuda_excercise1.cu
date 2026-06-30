#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// (a) un thread por fila
__global__
void MatrixMulKernelExcersice1(float* M, float* N, float* P, int Width) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < Width) {
        for (int col = 0; col < Width; ++col) {
            float Pvalue = 0;
            for (int k = 0; k < Width; ++k) {
                Pvalue += M[row * Width + k] * N[k * Width + col];
            }
            P[row * Width + col] = Pvalue;
        }
    }
}

// (b) un thread por columna
__global__
void MatrixMulKernelExcersice1B(float* M, float* N, float* P, int Width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < Width) {
        for (int row = 0; row < Width; ++row) {
            float Pvalue = 0;
            for (int k = 0; k < Width; ++k) {
                Pvalue += M[row * Width + k] * N[k * Width + col];
            }
            P[row * Width + col] = Pvalue;
        }
    }
}

// Referencia en CPU para verificar que la GPU dio bien
void cpuMatrixMul(const float* M, const float* N, float* P, int Width) {
    for (int row = 0; row < Width; ++row) {
        for (int col = 0; col < Width; ++col) {
            float sum = 0;
            for (int k = 0; k < Width; ++k) {
                sum += M[row * Width + k] * N[k * Width + col];
            }
            P[row * Width + col] = sum;
        }
    }
}

void matrixMult(float* M, float* N, float* P, int Width) {
    int size = Width * Width * sizeof(float); // Width*Width pq es cuadrada
    float *M_d, *N_d, *P_d;

    CUDA_CHECK(cudaMalloc((void**)&M_d, size));
    CUDA_CHECK(cudaMalloc((void**)&N_d, size));
    CUDA_CHECK(cudaMalloc((void**)&P_d, size));

    CUDA_CHECK(cudaMemcpy(M_d, M, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(N_d, N, size, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocks = (Width + threadsPerBlock - 1) / threadsPerBlock; // ceil sin floats

    // Para probar la (b), cambia el nombre a MatrixMulKernelExcersice1B
    MatrixMulKernelExcersice1<<<blocks, threadsPerBlock>>>(M_d, N_d, P_d, Width);

    CUDA_CHECK(cudaGetLastError());      // fallo en config del launch
    CUDA_CHECK(cudaDeviceSynchronize()); // fallo durante la ejecucion

    CUDA_CHECK(cudaMemcpy(P, P_d, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(M_d));
    CUDA_CHECK(cudaFree(N_d));
    CUDA_CHECK(cudaFree(P_d));
}

int main() {
    int Width = 4;
    int size = Width * Width * sizeof(float);

    float* M     = (float*)malloc(size);
    float* N     = (float*)malloc(size);
    float* P     = (float*)malloc(size);
    float* P_ref = (float*)malloc(size);

    // Datos de prueba deterministas
    for (int i = 0; i < Width * Width; ++i) {
        M[i] = (float)(i % 7);
        N[i] = (float)((i * 3) % 5);
    }

    matrixMult(M, N, P, Width);       // GPU
    cpuMatrixMul(M, N, P_ref, Width); // CPU referencia

    int errors = 0;
    for (int i = 0; i < Width * Width; ++i) {
        if (fabs(P[i] - P_ref[i]) > 1e-3f) {
            if (errors < 5) {
                printf("Mismatch en %d: GPU=%f CPU=%f\n", i, P[i], P_ref[i]);
            }
            errors++;
        }
    }

    if (errors == 0) printf("PASS: GPU y CPU coinciden\n");
    else             printf("FAIL: %d errores\n", errors);

    free(M); free(N); free(P); free(P_ref);
    return 0;
}