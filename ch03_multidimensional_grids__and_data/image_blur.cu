
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

int BLUR_SIZE = 0;
// 1 to 3x3
// 3 to 7x7


__global__
void blurKernel(unsigned char *in, unsigned char *out, int w, int h) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < w && row < h) {
        int pixVal = 0;
        int pixels = 0;

        // Gets the average if the surrounding pixels (BlUR_SIZE X BLUR_SIZE) box
        for (int blurRow=-BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
            for (int blurCol=-BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
                int curRow = row + blurRow; // current Row
                int curCol = col + blurCol;

                // Verify we have a valid image pixel
                if (curRow>= 0 && curRow<h && curCol>=0 && curCol<w) {
                    pixVal += in[curRow * w + curCol]; // coordinates of the flatten memory (curRow*w + curCol)
                    ++pixels; // keep track of number of pixels in the avg
                }
            }
        }
        // Write our new pixel value out
        out[row * w + col] = (unsigned char) (pixVal/pixels);
    }
}


int main() {
    return 0;
}
