#include "memorymanager.h"

#include <cassert>
#include <iostream>
#include <iomanip>

template<class T>
void initialize_memory(const MemoryManager& memory_manager, T* data, const int size, const T value) {
    if (typeid(*memory_manager) == typeid(*std::make_shared<CpuManager>())) {
        host_initialize_memory(data, size, value);
    } else {
#ifndef CPU_ONLY
        device_initialize_memory(data, size, value);
#endif
    }
}

template<class T>
void initialize_identity_matrix(const MemoryManager& memory_manager, T* data, const int num_rows, const int num_cols) {
    if (typeid(*memory_manager) == typeid(*std::make_shared<CpuManager>())) {
        host_initialize_identity_matrix(data, num_rows, num_cols);
    } else {
#ifndef CPU_ONLY
        device_initialize_identity_matrix(data, num_rows, num_cols);
#endif
    }
}

template<class T>
DeviceDataDevice<T>::DeviceDataDevice(const MemoryManager& memoryManager)
        : _memory_manager(memoryManager) {}

template<class T>
typename DeviceDataDevice<T>::SizeType DeviceDataDevice<T>::byte_size() const
{
    return size() * sizeof(T);
}

template<class T>
MemoryManager DeviceDataDevice<T>::get_memory_manager() const
{
    return _memory_manager;
}