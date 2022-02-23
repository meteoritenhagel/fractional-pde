template<class T>
void host_initialize_memory(T *host_memory, const int size, const T value) {
    for (int i = 0; i < size; ++i) {
        host_memory[i] = value;
    }
}

template<class T>
void host_initialize_identity_matrix(T *host_memory, const int num_rows, const int num_cols) {
    const int size = num_rows * num_cols;

    for (int idx = 0; idx < size; ++idx) {
        if (idx % (num_rows + 1) == 0) // if diagonal element
            host_memory[idx] = static_cast<T>(1.0);
        else
            host_memory[idx] = static_cast<T>(0.0);
    }
}