#ifndef TR_HACK_INITIALIZEMEMORY_CUH
#define TR_HACK_INITIALIZEMEMORY_CUH

/**
 * @warning This is a GPU kernel, so probably you would like to call the wrapper deviceInitializeMemory instead.
 *
 * Given a classical C-style array in the GPU memory, this function initializes @param size elements starting
 * from @param data with value @param value.
 *
 * @tparam T data type of array elements
 * @param deviceMemory[in/out] first element of array (in the GPU memory)
 * @param size number of elements in the array
 * @param value value to initialize elements with
 */
template<class T>
__global__ void initializeMemory(T* deviceMemory, const int size, const T value);

/**
 * @warning This is a GPU kernel, so probably you would like to call the wrapper deviceInitializeIdentityMatrix instead.
 * @warning Note that starting from @param data, at least N*M elements must be accessible
 *
 * Given a matrix in form of a classical C-style array in the GPU memory, this function sets it to be an
 * identity matrix.
 *
 * @tparam T data type of matrix elements
 * @param deviceMemory[in/out] pointer to start of data (in the GPU memory)
 * @param N number of the matrix' rows
 * @param M number of the matrix' columns
 *
 */
template<class T>
__global__ void initializeIdentityMatrix(T* deviceMemory, const int N, const int M);


/**
 * Given a classical C-style array in the GPU memory, this function initializes @param size elements starting
 * from @param data with value @param value.
 *
 * @tparam T data type of array elements
 * @param deviceMemory[in/out] first element of array (in the GPU memory)
 * @param size number of elements in the array
 * @param value value to initialize elements with
 *
 * @warning For every data type TYPE you want to use, you have to manually instantiate it in initializememory.cu using
 * @warning "template void deviceInitializeMemory<TYPE>(TYPE*, const size_t, const TYPE);"
 */
template <typename T>
extern void deviceInitializeMemory(T* deviceMemory, const size_t size, const T value);

/**
 * Given a matrix in form of a classical C-style array in the GPU memory, this function sets it to be an
 * identity matrix.
 *
 * @tparam T data type of matrix elements
 * @param deviceMemory[in/out] pointer to start of data (in the GPU memory)
 * @param N number of the matrix' rows
 * @param M number of the matrix' columns
 *
 * @warning Note that starting from @param data, at least N*M elements must be accessible
 * @warning For every data type TYPE you want to use, you have to manually instantiate it in initializememory.cu using
 * @warning "template void deviceInitializeIdentityMatrix<TYPE>(TYPE*, const size_t, const size_t);"
 */
template <typename T>
extern void deviceInitializeIdentityMatrix(T* deviceMemory, const size_t N, const size_t M);
#endif //TR_HACK_INITIALIZEMEMORY_CUH
