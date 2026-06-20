
#include <cstdio>
#include <cuda_runtime.h>


void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    for (int i  = 0; i < n; ++i) {
        C_h[i] = A_h[i] + B_h[i];
    }
}


int main() {
    float A[1024];
    float B[1024];
    float C[1024];
    int n = 52;

    for (int i = 0; i < n; ++i) {
        A[i] = i;
        B[i] = i * 2;
    }

    vecAdd(A, B, C, n);

    for (int i = 0; i < n; ++i) {
        printf("%.1f\n", C[i]);
    }

    return 0;
}
