#include <iostream>
#include "memorymanager.h"

void *MemoryManagerDevice::copyTo(void *pointerToMemory, const size_t byteSize, const MemoryManager manager) const {
    void *newPtr = pointerToMemory;

    // Only move to other device if the device is not the same
    if (typeid(*this) != typeid(*manager)) {
        newPtr = manager->allocate(byteSize);
        manager->copy(newPtr, pointerToMemory, byteSize);
    }

    return newPtr;
}

void *CPU_Manager::allocate(const size_t byteSize) const {
    if (byteSize != 0) {
        return malloc(byteSize);
    } else
        return nullptr;
}

void CPU_Manager::free(void *pointerToMemory) const {
    if (pointerToMemory)
        ::free(pointerToMemory);
}

void *CPU_Manager::copy(void *destination, void const *source, const size_t byteSize) const {
#ifdef CPU_ONLY
    return memcpy(destination, source, byteSize);
#else
    cudaMemcpy(destination, source, byteSize, cudaMemcpyDefault);
    return destination;
#endif
}

std::string CPU_Manager::display() const {
    return "Cpu MANAGER";
}

#ifndef CPU_ONLY

void* GPU_Manager::allocate(const size_t byteSize) const {
    void *pointer = nullptr;
    cudaError_t cudaerr;

    if (byteSize != 0) {
        cudaerr = cudaMalloc(&pointer, byteSize);
        cudaerr = cudaDeviceSynchronize();

        if (cudaerr != cudaSuccess)
            printf("kernel launch failed with error \"%s\".\n",
                   cudaGetErrorString(cudaerr));
    }

    return pointer;
}

void GPU_Manager::free(void *pointerToMemory) const {
    if (pointerToMemory)
        cudaFree(pointerToMemory);
}

void* GPU_Manager::copy(void *destination, void const *source, const size_t byteSize) const {
    cudaMemcpy(destination, source, byteSize, cudaMemcpyDefault);
    return destination;
}

std::string GPU_Manager::display() const {
    return "Gpu MANAGER";
}

void* UnifiedManager::allocate(const size_t byteSize) const {
    void *pointer = nullptr;
    if (byteSize != 0) {
        cudaMallocManaged(&pointer, byteSize);
    }
    return pointer;
}

void UnifiedManager::free(void *pointerToMemory) const {
    if (pointerToMemory)
        cudaFree(pointerToMemory);
}

void *UnifiedManager::copy(void *destination, void const *source, const size_t byteSize) const {
    cudaMemcpy(destination, source, byteSize, cudaMemcpyDefault);
    return destination;
}

std::string UnifiedManager::display() const {
    return "UNIFIED MANAGER";
}

#endif
