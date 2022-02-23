#include <iostream>
#include "memorymanager.h"

void *MemoryManagerDevice::copy_to(void *source, const size_t byteSize, const MemoryManager manager) const {
    void *newPtr = source;

    // Only move to other device if the device is not the same
    if (typeid(*this) != typeid(*manager)) {
        newPtr = manager->allocate(byteSize);
        manager->copy(newPtr, source, byteSize);
    }

    return newPtr;
}

void *CpuManager::allocate(const size_t byte_size) const {
    if (byte_size != 0) {
        return malloc(byte_size);
    } else
        return nullptr;
}

void CpuManager::free(void *ptr) const {
    if (ptr)
        ::free(ptr);
}

void *CpuManager::copy(void *destination, void const *source, const size_t byte_size) const {
#ifdef CPU_ONLY
    return memcpy(destination, source, byte_size);
#else
    cudaMemcpy(destination, source, byte_size, cudaMemcpyDefault);
    return destination;
#endif
}

std::string CpuManager::display() const {
    return "Cpu MANAGER";
}

#ifndef CPU_ONLY

void* GpuManager::allocate(const size_t byte_size) const {
    void *pointer = nullptr;
    cudaError_t cudaerr;

    if (byte_size != 0) {
        cudaerr = cudaMalloc(&pointer, byte_size);
        cudaerr = cudaDeviceSynchronize();

        if (cudaerr != cudaSuccess)
            printf("kernel launch failed with error \"%s\".\n",
                   cudaGetErrorString(cudaerr));
    }

    return pointer;
}

void GpuManager::free(void *ptr) const {
    if (ptr)
        cudaFree(ptr);
}

void* GpuManager::copy(void *destination, void const *source, const size_t byte_size) const {
    cudaMemcpy(destination, source, byte_size, cudaMemcpyDefault);
    return destination;
}

std::string GpuManager::display() const {
    return "Gpu MANAGER";
}

void* UnifiedManager::allocate(const size_t byte_size) const {
    void *pointer = nullptr;
    if (byte_size != 0) {
        cudaMallocManaged(&pointer, byte_size);
    }
    return pointer;
}

void UnifiedManager::free(void *ptr) const {
    if (ptr)
        cudaFree(ptr);
}

void *UnifiedManager::copy(void *destination, void const *source, const size_t byte_size) const {
    cudaMemcpy(destination, source, byte_size, cudaMemcpyDefault);
    return destination;
}

std::string UnifiedManager::display() const {
    return "UNIFIED MANAGER";
}

#endif
