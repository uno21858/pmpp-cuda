
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__
void MatrixVectorMulKernelExcersice2(float* B, float* C, float* A, int V) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

     if(i < V) {
         float Vresult = 0;
         for (int j = 0; j < V; ++j) {
             Vresult += B[i*V + j] * C[j];
         }
         A[i] = Vresult;
     }
}

void matrixMult(float* B, float* C, float* A, int V) {
    size_t sizeB = (size_t)V * V * sizeof(float);
    int sizeCA = V * sizeof(float);
    float *B_d, *C_d, *A_d;

    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaMalloc((void **) &B_d, sizeB));
    CUDA_CHECK(cudaMalloc((void **) &C_d, sizeCA));
    CUDA_CHECK(cudaMalloc((void **) &A_d, sizeCA));

    CUDA_CHECK(cudaMemcpy(B_d, B, sizeB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(C_d, C, sizeCA, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocks = (V + threadsPerBlock - 1) / threadsPerBlock;

    MatrixVectorMulKernelExcersice2<<<blocks, threadsPerBlock>>>(B_d,C_d,A_d, V);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(A, A_d, sizeCA, cudaMemcpyDeviceToHost));

    // CUDA_CHECK(cudaFree(A_d));
    // CUDA_CHECK(cudaFree(B_d));
    // CUDA_CHECK(cudaFree(C_d));
}

// Referencia en CPU para verificar la GPU
void cpuMatrixVectorMul(const float* B, const float* C, float* A, int V) {
    for (int i = 0; i < V; ++i) {
        float sum = 0;
        for (int j = 0; j < V; ++j) {
            sum += B[i * V + j] * C[j];
        }
        A[i] = sum;
    }
}

int main() {
    int V = 35000;  // no es multiplo de 256 -> ejercita el guard
    size_t sizeB = (size_t)V * V * sizeof(float);
    int sizeCA = V * sizeof(float);

    float* B     = (float*)malloc(sizeB);
    float* C     = (float*)malloc(sizeCA);
    float* A     = (float*)malloc(sizeCA);
    float* A_ref = (float*)malloc(sizeCA);

    // Datos deterministas
    for (int i = 0; i < V * V; ++i) B[i] = (float)(i % 7);
    for (int j = 0; j < V; ++j)     C[j] = (float)(j % 5);

    matrixMult(B, C, A, V);                 // GPU
    matrixMult(B, C, A, V);

    printf("Memoria asignada. Revisa nvtop. Enter para liberar...\n");
    getchar();  // el proceso se queda esperando aqui
    cpuMatrixVectorMul(B, C, A_ref, V);     // CPU referencia

    int errors = 0;
    for (int i = 0; i < V; ++i) {
        if (fabs(A[i] - A_ref[i]) > 1e-3f) {
            if (errors < 5) {
                printf("Mismatch en %d: GPU=%f CPU=%f\n", i, A[i], A_ref[i]);
            }
            errors++;
        }
    }

    if (errors == 0) printf("PASS: GPU y CPU coinciden\n");
    else             printf("FAIL: %d errores\n", errors);

    free(B); free(C); free(A); free(A_ref);
    return 0;
}