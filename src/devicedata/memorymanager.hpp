template<class T>
void hostInitializeMemory(T *hostMemory, const int size, const T value) {
    for (int i = 0; i < size; ++i) {
        hostMemory[i] = value;
    }
}

template<class T>
void hostInitializeIdentityMatrix(T *hostMemory, const int N, const int M) {
    const int size = N * M;

    for (int idx = 0; idx < size; ++idx) {
        if (idx % (N + 1) == 0) // if diagonal element
            hostMemory[idx] = static_cast<T>(1.0);
        else
            hostMemory[idx] = static_cast<T>(0.0);
    }
}