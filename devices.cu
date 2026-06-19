#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("GPUs encontradas: %d\n\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("GPU %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  VRAM: %.0f MB\n", prop.totalGlobalMem / 1024.0 / 1024.0);
        printf("  SMs: %d\n", prop.multiProcessorCount);
    }
    return 0;
}
