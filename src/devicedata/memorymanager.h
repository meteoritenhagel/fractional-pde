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
    void* allocate(const size_t byteSize) const override;

    void free(void* pointerToMemory) const override;

    void* copy(void* destination, void const * source, const size_t byteSize) const override;

    void display() const override;
};

class UnifiedManager : public MemoryManagerDevice {
    void* allocate(const size_t byteSize) const override;
    void free(void* pointerToMemory) const override;

    void* copy(void* destination, void const * source, const size_t byteSize) const override;

    void display() const override;
};

#endif

#include "memorymanager.cpp"
#endif //TR_HACK_DEVICEDATAPOINTER_H
