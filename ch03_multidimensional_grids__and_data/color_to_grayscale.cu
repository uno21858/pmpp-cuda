
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
void colorToGrayscaleConvertion(unsigned char* Pout, unsigned char* Pin, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row <height) {
        // Get !D offset for the grayscale image.
        int grayOffset = row*width + col; //
        char CHANNELS = 3;

        int rgbOffset = grayOffset*CHANNELS;
        unsigned char r = Pin[rgbOffset]; // Red value
        unsigned char g = Pin[rgbOffset + 1]; // green value
        unsigned char b = Pin[rgbOffset + 2]; // blue value

        //Perform the rescaling and store ir
        // Multiplyin by floating point constants
        Pout[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}


int main() {
    return 0;
}
