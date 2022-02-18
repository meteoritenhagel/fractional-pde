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
Timer GPU<floating>::createTimer() const
{
    return std::make_unique<GPU_Timer>();
}

template<class floating>
MemoryManager GPU<floating>::getMemoryManager() const
{
    return _deviceManager;
}

template<class floating>
auto GPU<floating>::getCublasHandle() const
{
    return _handle.getCublasHandle();
}

template<class floating>
auto GPU<floating>::getCusolverHandle() const
{
    return _handle.getCusolverHandle();
}

template<class floating>
std::string GPU<floating>::display() const
{
    return "GPU";
}

template<class floating>
int GPU<floating>::ixamax(const int n, const floating * const x, const int incx) const
{
    int output = 0;
    if constexpr(isFloat())
        cublasIsamax(getCublasHandle(), n, x, incx, &output);
    else if constexpr(isDouble())
        cublasIdamax(getCublasHandle(), n, x, incx, &output);
    return output - 1; //since cuBLAS uses Fortran indexing here...
}

template<class floating>
void GPU<floating>::xaxpy(const int n, const floating alpha, const floating * const x, const int incx, floating * const y, const int incy) const
{
    if constexpr(isFloat())
        cublasSaxpy(getCublasHandle(), n, &alpha, x, incx, y, incy);
    else if constexpr(isDouble())
        cublasDaxpy(getCublasHandle(), n, &alpha, x, incx, y, incy);
    cudaDeviceSynchronize();
    return;
}

template<class floating>
void GPU<floating>::xcopy(const int n, const floating * const source, const int incx, floating * const dest, const int incy) const
{
    if constexpr(isFloat())
        cublasScopy(getCublasHandle(), n, source, incx, dest, incy);
    else if constexpr(isDouble())
        cublasDcopy(getCublasHandle(), n, source, incx, dest, incy);
    return;
}

template<class floating>
floating GPU<floating>::xdot(const int n, const floating * const x, const int incx, const floating * const y, const int incy) const
{
    floating result = 0;

    if constexpr(isFloat())
        cublasSdot(getCublasHandle(), n, x, incx, y, incy, &result);
    else if constexpr(isDouble())
        cublasDdot(getCublasHandle(), n, x, incx, y, incy, &result);
    cudaDeviceSynchronize();
    return result;
}

template<class floating>
void GPU<floating>::xgemm(const OperationType TransA, const OperationType TransB,
        const int M, const int N, const int K, const floating alpha, const floating * const A,
        const int lda, const floating * const B, const int ldb, const floating beta, floating * const C, const int ldc) const
{
    if constexpr(isFloat())
        cublasSgemm(getCublasHandle(), toInternalOperation.at(TransA), toInternalOperation.at(TransB),
                    M, N, K, &alpha, A,
                    lda, B, ldb, &beta, C, ldc);
    else if constexpr(isDouble())
        cublasDgemm(getCublasHandle(), toInternalOperation.at(TransA), toInternalOperation.at(TransB),
                    M, N, K, &alpha, A,
                    lda, B, ldb, &beta, C, ldc);

    cudaDeviceSynchronize();
    return;
}

template<class floating>
void GPU<floating>::xgemv(const OperationType trans, const int m, const int n,
        const floating alpha, floating const * const a, const int lda, floating const * const x, const int incx,
        const floating beta, floating * const y, const int incy) const
{
    if constexpr(isFloat())
        cublasSgemv(getCublasHandle(), toInternalOperation.at(trans), m, n,
                    &alpha, a, lda, x, incx,
                    &beta, y, incy);
    else if constexpr(isDouble())
        cublasDgemv(getCublasHandle(), toInternalOperation.at(trans), m, n,
                    &alpha, a, lda, x, incx,
                    &beta, y, incy);
    cudaDeviceSynchronize();
    return;
}


template<class floating>
void GPU<floating>::xgetrf(int * const m, int * const n, floating * const a, int * const lda,
        int * const ipiv, int * const info) const
{
    int workingBufferSize = 0;

    if constexpr(isFloat())
                cusolverDnSgetrf_bufferSize(getCusolverHandle(), *m, *n, a, *lda, &workingBufferSize);
    else if constexpr(isDouble())
                cusolverDnDgetrf_bufferSize(getCusolverHandle(), *m, *n, a, *lda, &workingBufferSize);
    cudaDeviceSynchronize();

    // allocate GPU Memory
    const auto memoryManager = std::make_shared<GPU_Manager>();
    // buffer for cusolverDnSgetrf
    DeviceArray<floating> workingBuffer(workingBufferSize, 0, memoryManager);

    // int* devInfo in cusolverDn<t>getrf(..., devInfo) is expected to be on the device
    DeviceScalar<int> deviceInfo(0, memoryManager);

    if constexpr(isFloat())
        cusolverDnSgetrf(getCusolverHandle(), *m, *n, a, *lda, workingBuffer.data(), ipiv, deviceInfo.data());
    else if constexpr(isDouble())
        cusolverDnDgetrf(getCusolverHandle(), *m, *n, a, *lda, workingBuffer.data(), ipiv, deviceInfo.data());

    cudaDeviceSynchronize();
    *info = deviceInfo.value();
    return;
}

template<class floating>
void GPU<floating>::xgetri(const int * const n, floating * const a, const int * const lda, const int * const ipiv,
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
void GPU<floating>::xgetrs(const OperationType trans, const int * const n, const int * const nrhs,
        const floating * const a, const int * const lda, const int * const ipiv,
        floating * const b, const int * const ldb, int * const info) const
{

    // allocate GPU Memory
    const auto memoryManager = std::make_shared<GPU_Manager>();
    // int* devInfo in cusolverDn<t>getrf(..., devInfo) is expected to be on the device
    DeviceScalar<int> deviceInfo(0, memoryManager);

    if constexpr(isFloat())
        cusolverDnSgetrs(getCusolverHandle(), toInternalOperation.at(trans), *n, *nrhs, a, *lda, ipiv, b, *ldb, deviceInfo.data());
    else if constexpr(isDouble())
        cusolverDnDgetrs(getCusolverHandle(), toInternalOperation.at(trans), *n, *nrhs, a, *lda, ipiv, b, *ldb, deviceInfo.data());

    cudaDeviceSynchronize();
    return;
}

template<class floating>
void GPU<floating>::xscal(const int N, const floating alpha, floating * const X, const int incX) const
{
    if constexpr(isFloat())
        cublasSscal(getCublasHandle(), N, &alpha, X, incX);
    else if constexpr(isDouble())
        cublasDscal(getCublasHandle(), N, &alpha, X, incX);
    cudaDeviceSynchronize();
    return;
}

template<class floating>
std::map<OperationType, cublasOperation_t> GPU<floating>::toInternalOperation = {
        {OperationType::Identical, CUBLAS_OP_N},
        {OperationType::Transposed, CUBLAS_OP_T},
        {OperationType::Hermitian, CUBLAS_OP_HERMITAN}
};

template<class floating>
GPU_Handle GPU<floating>::_handle;

template<class floating>
MemoryManager GPU<floating>::_deviceManager = std::make_shared<GPU_Manager>();
