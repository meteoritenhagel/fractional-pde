#ifndef TR_HACK_INITIALIZEMEMORY_CUH
#define TR_HACK_INITIALIZEMEMORY_CUH

template<class T>
__global__ void initializeMemory(T* deviceMemory, const int size, const T value);

template<class T>
__global__ void initializeIdentityMatrix(T* deviceMemory, const int N, const int M);

template <typename T>
extern void deviceInitializeMemory(T* deviceMemory, const size_t size, const T value);

template <typename T>
extern void deviceInitializeIdentityMatrix(T* deviceMemory, const size_t N, const size_t M);
#endif //TR_HACK_INITIALIZEMEMORY_CUH
