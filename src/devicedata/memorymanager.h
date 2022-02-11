#ifndef TR_HACK_DEVICEDATAPOINTER_H
#define TR_HACK_DEVICEDATAPOINTER_H

#ifndef CPU_ONLY
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <cstdlib>
#include <cstring>
#include <memory>
#include <typeinfo>

// TODO REMOVE
#include <iostream>

template<class T>
void hostInitializeMemory(T* hostMemory, const int size, const T value);

template<class T>
void hostInitializeIdentityMatrix(T* hostMemory, const int N, const int M);

class MemoryManagerDevice;

using MemoryManager = std::shared_ptr<MemoryManagerDevice>;

class MemoryManagerDevice {
public:
    MemoryManagerDevice() = default;
    virtual ~MemoryManagerDevice() = default;

    virtual void * allocate(const size_t byteSize) const = 0;
    virtual void free(void* pointerToMemory) const = 0;
    virtual void* copy(void* destination, void const * source, const size_t size) const = 0;
    virtual void display() const = 0;

    void* copyTo(void* pointerToMemory, const size_t byteSize, const MemoryManager manager) const;
};

class CPU_Manager : public MemoryManagerDevice {
    void* allocate(const size_t byteSize) const override;

    void free(void* pointerToMemory) const override;

    void* copy(void * destination, void const * source, const size_t byteSize) const override;

    void display() const override;
};

#ifndef CPU_ONLY

class GPU_Manager : public MemoryManagerDevice {
    void* allocate(const size_t byteSize) const override {
        void *pointer = nullptr;
        cudaError_t cudaerr;

        if (byteSize != 0)
        {
            cudaerr = cudaMalloc(&pointer, byteSize);
            cudaerr = cudaDeviceSynchronize();
            //std::cout << "ALLOCATED " << pointer << "(" << byteSize << " BYTES) ON GPU" << std::endl;
            if (cudaerr != cudaSuccess)
            printf("kernel launch failed with error \"%s\".\n",
                   cudaGetErrorString(cudaerr));
        }

        return pointer;
    }

    void free(void* pointerToMemory) const override
    {
        if (pointerToMemory)
            cudaFree(pointerToMemory);
    }

    void* copy(void* destination, void const * source, const size_t byteSize) const override
    {
//        static size_t numOfCopies = 0;
//        ++numOfCopies;
//
//        if (numOfCopies % 5000 == 0)
//            std::cout << "Copies from/to GPU: " << numOfCopies << std::endl;
        cudaMemcpy(destination, source, byteSize, cudaMemcpyDefault);
        return destination;
    }

    void display() const override
    {
        std::cout << "GPU MANAGER" << std::endl;
    }
};

class UnifiedManager : public MemoryManagerDevice {
    void* allocate(const size_t byteSize) const override {
        void *pointer = nullptr;
        if (byteSize != 0)
        {
            cudaMallocManaged(&pointer, byteSize);
        }
        return pointer;
    }

    void free(void* pointerToMemory) const override
    {
        if (pointerToMemory)
            cudaFree(pointerToMemory);
    }

    void* copy(void* destination, void const * source, const size_t byteSize) const override
    {
        cudaMemcpy(destination, source, byteSize, cudaMemcpyDefault);
        return destination;
    }

    void display() const override
    {
        std::cout << "UNIFIED MANAGER" << std::endl;
    }
};

#endif

#include "memorymanager.cpp"
#endif //TR_HACK_DEVICEDATAPOINTER_H
