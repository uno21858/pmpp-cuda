cmake_minimum_required(VERSION 3.18)
project(aprenderCuda CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(vecadd_gpu ch02_data_parallel/vecadd_gpu.cu)