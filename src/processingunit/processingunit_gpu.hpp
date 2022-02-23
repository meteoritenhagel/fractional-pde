#include "../devicedata/devicedata.h"

#include <cublas_v2.h>
#include <cusolverDn.h>

#include <cassert>

#ifdef CUBLAS_API_H_
// cuBLAS API errors
static std::string cublasErrorToString(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

static std::string cusolverErrorToString(cusolverStatus_t error)
{
    switch (error)
    {
        case CUSOLVER_STATUS_SUCCESS:
            return "CUSOLVER_STATUS_SUCCESS";

        case CUSOLVER_STATUS_NOT_INITIALIZED:
            return "CUSOLVER_STATUS_NOT_INITIALIZED";

        case CUSOLVER_STATUS_ALLOC_FAILED:
            return "CUSOLVER_STATUS_ALLOC_FAILED";

        case CUSOLVER_STATUS_INVALID_VALUE:
            return "CUSOLVER_STATUS_INVALID_VALUE";

        case CUSOLVER_STATUS_ARCH_MISMATCH:
            return "CUSOLVER_STATUS_ARCH_MISMATCH";

        case CUSOLVER_STATUS_EXECUTION_FAILED:
            return "CUSOLVER_STATUS_EXECUTION_FAILED";

        case CUSOLVER_STATUS_INTERNAL_ERROR:
            return "CUSOLVER_STATUS_INTERNAL_ERROR";

        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    }

    return "<unknown>";
}
#endif

template<class floating>
Timer Gpu<floating>::create_timer() const
{
    return std::make_unique<GpuTimer>();
}

template<class floating>
MemoryManager Gpu<floating>::get_memory_manager() const
{
    return _deviceManager;
}

template<class floating>
auto Gpu<floating>::get_cublas_handle() const
{
    return _handle.get_cublas_handle();
}

template<class floating>
auto Gpu<floating>::get_cusolver_handle() const
{
    return _handle.get_cusolver_handle();
}

template<class floating>
std::string Gpu<floating>::display() const
{
    return "Gpu";
}

template<class floating>
int Gpu<floating>::ixamax(const int n, const floating * const x, const int incx) const
{
    int output = 0;
    if constexpr(isFloat())
        cublasIsamax(get_cublas_handle(), n, x, incx, &output);
    else if constexpr(isDouble())
        cublasIdamax(get_cublas_handle(), n, x, incx, &output);
    return output - 1; //since cuBLAS uses Fortran indexing here...
}

template<class floating>
void Gpu<floating>::xaxpy(const int n, const floating alpha, const floating * const x, const int incx, floating * const y, const int incy) const
{
    if constexpr(isFloat())
        cublasSaxpy(get_cublas_handle(), n, &alpha, x, incx, y, incy);
    else if constexpr(isDouble())
        cublasDaxpy(get_cublas_handle(), n, &alpha, x, incx, y, incy);
    cudaDeviceSynchronize();
    return;
}

template<class floating>
void Gpu<floating>::xcopy(const int n, const floating * const source, const int incx, floating * const dest, const int incy) const
{
    if constexpr(isFloat())
        cublasScopy(get_cublas_handle(), n, source, incx, dest, incy);
    else if constexpr(isDouble())
        cublasDcopy(get_cublas_handle(), n, source, incx, dest, incy);
    return;
}

template<class floating>
floating Gpu<floating>::xdot(const int n, const floating * const x, const int incx, const floating * const y, const int incy) const
{
    floating result = 0;

    if constexpr(isFloat())
        cublasSdot(get_cublas_handle(), n, x, incx, y, incy, &result);
    else if constexpr(isDouble())
        cublasDdot(get_cublas_handle(), n, x, incx, y, incy, &result);
    cudaDeviceSynchronize();
    return result;
}

template<class floating>
void Gpu<floating>::xgemm(const OperationType TransA, const OperationType TransB,
                          const int M, const int N, const int K, const floating alpha, const floating * const A,
                          const int lda, const floating * const B, const int ldb, const floating beta, floating * const C, const int ldc) const
{
    if constexpr(isFloat())
        cublasSgemm(get_cublas_handle(), to_internal_operation.at(TransA), to_internal_operation.at(TransB),
                    M, N, K, &alpha, A,
                    lda, B, ldb, &beta, C, ldc);
    else if constexpr(isDouble())
        cublasDgemm(get_cublas_handle(), to_internal_operation.at(TransA), to_internal_operation.at(TransB),
                    M, N, K, &alpha, A,
                    lda, B, ldb, &beta, C, ldc);

    cudaDeviceSynchronize();
    return;
}

template<class floating>
void Gpu<floating>::xgemv(const OperationType trans, const int m, const int n,
                          const floating alpha, floating const * const a, const int lda, floating const * const x, const int incx,
                          const floating beta, floating * const y, const int incy) const
{
    if constexpr(isFloat())
        cublasSgemv(get_cublas_handle(), to_internal_operation.at(trans), m, n,
                    &alpha, a, lda, x, incx,
                    &beta, y, incy);
    else if constexpr(isDouble())
        cublasDgemv(get_cublas_handle(), to_internal_operation.at(trans), m, n,
                    &alpha, a, lda, x, incx,
                    &beta, y, incy);
    cudaDeviceSynchronize();
    return;
}


template<class floating>
void Gpu<floating>::xgetrf(int * const m, int * const n, floating * const a, int * const lda,
                           int * const ipiv, int * const info) const
{
    int workingBufferSize = 0;

    if constexpr(isFloat())
                cusolverDnSgetrf_bufferSize(get_cusolver_handle(), *m, *n, a, *lda, &workingBufferSize);
    else if constexpr(isDouble())
                cusolverDnDgetrf_bufferSize(get_cusolver_handle(), *m, *n, a, *lda, &workingBufferSize);
    cudaDeviceSynchronize();

    // allocate Gpu Memory
    const auto memoryManager = std::make_shared<GpuManager>();
    // buffer for cusolverDnSgetrf
    DeviceArray<floating> workingBuffer(workingBufferSize, 0, memoryManager);

    // int* devInfo in cusolverDn<t>getrf(..., devInfo) is expected to be on the device
    DeviceScalar<int> deviceInfo(0, memoryManager);

    if constexpr(isFloat())
        cusolverDnSgetrf(get_cusolver_handle(), *m, *n, a, *lda, workingBuffer.data(), ipiv, deviceInfo.data());
    else if constexpr(isDouble())
        cusolverDnDgetrf(get_cusolver_handle(), *m, *n, a, *lda, workingBuffer.data(), ipiv, deviceInfo.data());

    cudaDeviceSynchronize();
    *info = deviceInfo.value();
    return;
}

template<class floating>
void Gpu<floating>::xgetri(const int * const n, floating * const a, const int * const lda, const int * const ipiv,
                           floating * const work, const int * const lwork, int * const info) const
{
    // NOT PROVIDED BY cuSOLVER!
#ifdef MAGMA
    if constexpr(isFloat())
        magma_sgetri_gpu(*n, a, *lda, const_cast<int*>(ipiv), work, *lwork, info);
    else if constexpr(isDouble())
        magma_dgetri_gpu(*n, a, *lda, const_cast<int*>(ipiv), work, *lwork, info);
#else
    std::cerr << "XGETRI not supported without MAGMA" << std::endl;
    exit(-1);
#endif
}

template<class floating>
void Gpu<floating>::xgetrs(const OperationType trans, const int * const n, const int * const nrhs,
                           const floating * const a, const int * const lda, const int * const ipiv,
                           floating * const b, const int * const ldb, int * const info) const
{

    // allocate Gpu Memory
    const auto memoryManager = std::make_shared<GpuManager>();
    // int* devInfo in cusolverDn<t>getrf(..., devInfo) is expected to be on the device
    DeviceScalar<int> deviceInfo(0, memoryManager);

    if constexpr(isFloat())
        cusolverDnSgetrs(get_cusolver_handle(), to_internal_operation.at(trans), *n, *nrhs, a, *lda, ipiv, b, *ldb, deviceInfo.data());
    else if constexpr(isDouble())
        cusolverDnDgetrs(get_cusolver_handle(), to_internal_operation.at(trans), *n, *nrhs, a, *lda, ipiv, b, *ldb, deviceInfo.data());

    cudaDeviceSynchronize();
    return;
}

template<class floating>
void Gpu<floating>::xscal(const int N, const floating alpha, floating * const X, const int incX) const
{
    if constexpr(isFloat())
        cublasSscal(get_cublas_handle(), N, &alpha, X, incX);
    else if constexpr(isDouble())
        cublasDscal(get_cublas_handle(), N, &alpha, X, incX);
    cudaDeviceSynchronize();
    return;
}

template<class floating>
std::map<OperationType, cublasOperation_t> Gpu<floating>::to_internal_operation = {
        {OperationType::Identical, CUBLAS_OP_N},
        {OperationType::Transposed, CUBLAS_OP_T},
        {OperationType::Hermitian, CUBLAS_OP_HERMITAN}
};

template<class floating>
GpuHandle Gpu<floating>::_handle;

template<class floating>
MemoryManager Gpu<floating>::_deviceManager = std::make_shared<GpuManager>();
