cmake_minimum_required(VERSION 3.18)
project(aprenderCuda CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(cuda_excercises
        ch03_multidimensional_grids__and_data/cuda_exercise2.cu)