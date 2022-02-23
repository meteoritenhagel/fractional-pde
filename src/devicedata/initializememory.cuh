#ifndef TR_HACK_INITIALIZEMEMORY_CUH
#define TR_HACK_INITIALIZEMEMORY_CUH

/**
 * @warning This is a CUDA kernel, so probably you would like to call the wrapper device_initialize_memory instead.
 *
 * Given a classical C-style array in the GPU memory, this function initializes @p size elements starting
 * from @p data with value @p value.
 *
 * @tparam T data type of array elements
 * @param[in,out] device_memory first element of array (in the GPU memory)
 * @param[in] size number of elements in the array
 * @param[in] value value to initialize elements with
 */
template<class T>
__global__ void initialize_memory_kernel(T* device_memory, const int size, const T value);

/**
 * @warning This is a CUDA kernel, so probably you would like to call the wrapper device_initialize_identity_matrix instead.
 * @warning Note that starting from @p data, at least num_rows*num_cols elements must be accessible
 *
 * Given a matrix in form of a classical C-style array in the GPU memory, this function sets it to be an
 * identity matrix, where the storage is assumed to be column-wise.
 *
 * @tparam T data type of matrix elements
 * @param[in,out] device_memory pointer to start of data (in the GPU memory)
 * @param[in] num_rows number of the matrix' rows
 * @param[in] num_cols number of the matrix' columns
 *
 */
template<class T>
__global__ void initialize_identity_matrix_kernel(T* device_memory, const int num_rows, const int num_cols);

/**
 * Given a classical C-style array in the GPU memory, this function initializes @p size elements starting
 * from @p data with value @p value.
 *
 * @tparam T data type of array elements
 * @param device_memory[in/out] first element of array (in the GPU memory)
 * @param size number of elements in the array
 * @param value value to initialize elements with
 *
 * @warning For every data type TYPE you want to use, you have to manually instantiate it in initializememory.cu using
 * @warning "template void device_initialize_memory<TYPE>(TYPE*, const size_t, const TYPE);"
 */
template <typename T>
extern void device_initialize_memory(T* device_memory, const size_t size, const T value);

/**
 * Given a matrix in form of a classical C-style array in the GPU memory, this function sets it to be an
 * identity matrix, where the storage is assumed to be column-wise.
 *
 * @tparam T data type of matrix elements
 * @param device_memory[in/out] pointer to start of data (in the GPU memory)
 * @param num_rows number of the matrix' rows
 * @param num_cols number of the matrix' columns
 *
 * @warning Note that starting from @p data, at least num_rows*num_cols elements must be accessible
 * @warning For every data type TYPE you want to use, you have to manually instantiate it in initializememory.cu using
 * @warning "template void device_initialize_identity_matrix<TYPE>(TYPE*, const size_t, const size_t);"
 */
template <typename T>
extern void device_initialize_identity_matrix(T* device_memory, const size_t num_rows, const size_t num_cols);
#endif //TR_HACK_INITIALIZEMEMORY_CUH
