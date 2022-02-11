#include "memorymanager.h"

#include <cassert>
#include <iostream>
#include <iomanip>

template<class T>
void initializeMemory(const MemoryManager& memoryManager, T* data, const int size, const T value) {
    if (typeid(*memoryManager) == typeid(*std::make_shared<CPU_Manager>())) {
        hostInitializeMemory(data, size, value);
    } else {
#ifndef CPU_ONLY
        deviceInitializeMemory(data, size, value);
#endif
    }
}

template<class T>
void initializeIdentityMatrix(const MemoryManager& memoryManager, T* data, const int N, const int M) {
    if (typeid(*memoryManager) == typeid(*std::make_shared<CPU_Manager>())) {
        hostInitializeIdentityMatrix(data, N, M);
    } else {
#ifndef CPU_ONLY
        deviceInitializeIdentityMatrix(data, N, M);
#endif
    }
}

template<class T>
DeviceDataDevice<T>::DeviceDataDevice(const MemoryManager& memoryManager)
        : _memoryManager(memoryManager) {}

template<class T>
typename DeviceDataDevice<T>::SizeType DeviceDataDevice<T>::byteSize() const
{
    return size() * sizeof(T);
}

template<class T>
MemoryManager DeviceDataDevice<T>::getMemoryManager() const
{
    return _memoryManager;
}