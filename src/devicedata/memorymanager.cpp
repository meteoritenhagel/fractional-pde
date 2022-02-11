template<class T>
void hostInitializeMemory(T* hostMemory, const int size, const T value)
{
    for (int i = 0; i < size; ++i)
    {
        hostMemory[i] = value;
    }
}

template<class T>
void hostInitializeIdentityMatrix(T* hostMemory, const int N, const int M)
{
    const int size = N*M;

    for (int idx = 0; idx < size; ++idx)
    {
        if (idx % (N+1) == 0) // if diagonal element
            hostMemory[idx] = static_cast<T>(1.0);
        else
            hostMemory[idx] = static_cast<T>(0.0);
    }
}

void* MemoryManagerDevice::copyTo(void* pointerToMemory, const size_t byteSize, const MemoryManager manager) const
{
    void* newPtr = pointerToMemory;

    // Only move to other device if the device is not the same
    if (typeid(*this) != typeid(*manager)) {
        newPtr = manager->allocate(byteSize);
        manager->copy(newPtr, pointerToMemory, byteSize);
    }

    return newPtr;
}

void* CPU_Manager::allocate(const size_t byteSize) const
{
    if (byteSize != 0) {
    return malloc(byteSize);
    }
    else
    return nullptr;
}

void CPU_Manager::free(void* pointerToMemory) const
{
    if (pointerToMemory)
    ::free(pointerToMemory);
}

void* CPU_Manager::copy(void * destination, void const * source, const size_t byteSize) const
{
#ifdef CPU_ONLY
    return memcpy(destination, source, byteSize);
#else
cudaMemcpy(destination, source, byteSize, cudaMemcpyDefault);
    return destination;
#endif
}

void CPU_Manager::display() const
{
    std::cout << "CPU MANAGER" << std::endl;
}
