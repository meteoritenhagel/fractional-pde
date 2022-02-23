#include "initializememory.cuh"

#include <stdio.h>
#include <iostream>

template<class T>
__global__ void initialize_memory_kernel(T* device_memory, const int size, const T value)
{
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < size)
    {
        device_memory[idx] = value;
    }
}

template<class T>
__global__ void initialize_identity_matrix_kernel(T* device_memory, const int num_rows, const int num_cols)
{
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_rows * num_cols)
    {
        if (idx % (1 + num_rows) == 0) // is diagonal element
            device_memory[idx] = static_cast<T>(1.0);
        else
            device_memory[idx] = static_cast<T>(0.0);
    }
}

template <typename T>
void device_initialize_memory(T* device_memory, const size_t size, const T value)
{
    const int block_size = 1024;
    const int number_of_blocks = size / block_size + 1;
    //std::cout << block_size << " / " << number_of_blocks << std::endl;
    dim3 gridDim(number_of_blocks, 1);
    dim3 blockDim(block_size, 1);
    initialize_memory_kernel<T> <<<gridDim, blockDim>>>(device_memory, size, value);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
}

template <typename T>
void device_initialize_identity_matrix(T* device_memory, const size_t num_rows, const size_t num_cols)
{
    const int block_size = 1024;
    const int number_of_blocks = num_rows * num_cols / block_size + 1;
    //std::cout << block_size << " / " << number_of_blocks << std::endl;
    dim3 gridDim(number_of_blocks, 1);
    dim3 blockDim(block_size, 1);
    initialize_identity_matrix_kernel<T> <<<gridDim, blockDim>>>(device_memory, num_rows, num_cols);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
                cudaGetErrorString(cudaerr));
}

// Since nvcc is going to compile this and has no information about which instantiations
// g++ needs, the necessary instantiations have to be listed here.
template void device_initialize_memory<int>(int*, const size_t, const int);
template void device_initialize_memory<float>(float*, const size_t, const float);
template void device_initialize_memory<double>(double*, const size_t, const double);
template void device_initialize_identity_matrix<int>(int*, const size_t, const size_t);
template void device_initialize_identity_matrix<float>(float*, const size_t, const size_t);
template void device_initialize_identity_matrix<double>(double*, const size_t, const size_t);
