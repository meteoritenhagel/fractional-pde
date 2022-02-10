#include "initializememory.cuh"

#include <stdio.h>
#include <iostream>

template<class T>
__global__ void initializeMemory(T* deviceMemory, const int size, const T value)
{
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < size)
    {
        deviceMemory[idx] = value;
    }
}

template<class T>
__global__ void initializeIdentityMatrix(T* deviceMemory, const int N, const int M)
{
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N*M)
    {
        if (idx % (1+N) == 0) // is diagonal element
            deviceMemory[idx] = static_cast<T>(1.0);
        else
            deviceMemory[idx] = static_cast<T>(0.0);
    }
}

template <typename T>
void deviceInitializeMemory(T* deviceMemory, const size_t size, const T value)
{
    const int block_size = 1024;
    const int number_of_blocks = size / block_size + 1;
    //std::cout << block_size << " / " << number_of_blocks << std::endl;
    dim3 gridDim(number_of_blocks, 1);
    dim3 blockDim(block_size, 1);
    initializeMemory<T> <<<gridDim, blockDim>>>(deviceMemory, size, value);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
}

template <typename T>
void deviceInitializeIdentityMatrix(T* deviceMemory, const size_t N, const size_t M)
{
    const int block_size = 1024;
    const int number_of_blocks = N*M / block_size + 1;
    //std::cout << block_size << " / " << number_of_blocks << std::endl;
    dim3 gridDim(number_of_blocks, 1);
    dim3 blockDim(block_size, 1);
    initializeIdentityMatrix<T> <<<gridDim, blockDim>>>(deviceMemory, N, M);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
                cudaGetErrorString(cudaerr));
}

// Since nvcc is going to compile this and has no information about which instantiations
// g++ needs, the necessary instantiations have to be listed here.
template void deviceInitializeMemory<int>(int*, const size_t, const int);
template void deviceInitializeMemory<float>(float*, const size_t, const float);
template void deviceInitializeMemory<double>(double*, const size_t, const double);
template void deviceInitializeIdentityMatrix<int>(int*, const size_t, const size_t);
template void deviceInitializeIdentityMatrix<float>(float*, const size_t, const size_t);
template void deviceInitializeIdentityMatrix<double>(double*, const size_t, const size_t);
