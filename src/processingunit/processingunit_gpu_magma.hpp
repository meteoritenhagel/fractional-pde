#include <cublas_v2.h>
#include <cusolverDn.h>

#include <cassert>

template<class floating>
Timer GPU_MAGMA<floating>::createTimer() const
{
    return std::make_unique<GPU_Timer>();
}

template<class floating>
MemoryManager GPU_MAGMA<floating>::getMemoryManager() const
{
    return _deviceManager;
}

template<class floating>
void GPU_MAGMA<floating>::display() const
{
    std::cout << "GPU_MAGMA" << std::endl;
    return;
}

template<class floating>
auto GPU_MAGMA<floating>::getMagmaQueue() const
{
    return _queue.getMagmaQueue();
}

template<class floating>
int GPU_MAGMA<floating>::ixamax(const int n, const floating * const x, const int incx) const
{
    int output = 0;
    if constexpr(isFloat())
        output = magma_isamax(n, x, incx, getMagmaQueue());
    else if constexpr(isDouble())
        output = magma_idamax(n, x, incx, getMagmaQueue());
    return output-1; // Since MAGMA uses Fortran indexing here...
}

template<class floating>
void GPU_MAGMA<floating>::xaxpy(const int n, const floating alpha, const floating * const x, const int incx, floating * const y, const int incy, const bool synchronize) const
{
    if constexpr(isFloat())
        magma_saxpy(n, alpha, x, incx, y, incy, getMagmaQueue());
    else if constexpr(isDouble())
        magma_daxpy(n, alpha, x, incx, y, incy, getMagmaQueue());
        
    if (synchronize)
        cudaDeviceSynchronize();
        
    return;
}

template<class floating>
void GPU_MAGMA<floating>::xcopy(const int n, const floating * const source, const int incx, floating * const dest, const int incy) const
{
    if constexpr(isFloat())
        magma_scopy(n, source, incx, dest, incy, getMagmaQueue());
    else if constexpr(isDouble())
        magma_dcopy(n, source, incx, dest, incy, getMagmaQueue());
    cudaDeviceSynchronize();
    return;
}

template<class floating>
floating GPU_MAGMA<floating>::xdot(const int n, const floating * const x, const int incx, const floating * const y, const int incy) const
{
    floating result = 0;
    if constexpr(isFloat())
        result = magma_sdot(n, x, incx, y, incy, getMagmaQueue());
    else if constexpr(isDouble())
        result = magma_ddot(n, x, incx, y, incy, getMagmaQueue());

    return result;
}

template<class floating>
void GPU_MAGMA<floating>::xgemm(const OperationType TransA, const OperationType TransB,
        const int M, const int N, const int K, const floating alpha, const floating * const A,
        const int lda, const floating * const B, const int ldb, const floating beta, floating * const C, const int ldc) const
{
    if constexpr(isFloat())
        magma_sgemm(toInternalOperation.at(TransA), toInternalOperation.at(TransB),
                    M, N, K, alpha, A,
                    lda, B, ldb, beta, C, ldc, getMagmaQueue());
    else if constexpr(isDouble())
        magma_dgemm(toInternalOperation.at(TransA), toInternalOperation.at(TransB),
                    M, N, K, alpha, A,
                    lda, B, ldb, beta, C, ldc, getMagmaQueue());

    cudaDeviceSynchronize();
    return;
}

template<class floating>
void GPU_MAGMA<floating>::xgemv(const OperationType trans, const int m, const int n,
        const floating alpha, floating const * const a, const int lda, floating const * const x, const int incx,
        const floating beta, floating * const y, const int incy) const
{
    if constexpr(isFloat())
        magma_sgemv(toInternalOperation.at(trans), m, n,
                    alpha, a, lda, x, incx,
                    beta, y, incy, getMagmaQueue());
    else if constexpr(isDouble())
        magma_dgemv(toInternalOperation.at(trans), m, n,
                    alpha, a, lda, x, incx,
                    beta, y, incy, getMagmaQueue());
    cudaDeviceSynchronize();
    return;
}


template<class floating>
void GPU_MAGMA<floating>::xgetrf(int * const m, int * const n, floating * const a, int * const lda,
        int * const ipiv, int * const info) const
{
    if constexpr(isFloat())
        magma_sgetrf(*m, *n, a, *lda, ipiv, info);
    else if constexpr(isDouble())
        magma_dgetrf(*m, *n, a, *lda, ipiv, info);

    cudaDeviceSynchronize();
    return;
}

template<class floating>
void GPU_MAGMA<floating>::xgetri(const int * const n, floating * const a, const int * const lda, const int * const ipiv,
                           floating * const work, const int * const lwork, int * const info) const
{
    if constexpr(isFloat())
        magma_sgetri_gpu(*n, a, *lda, const_cast<int*>(ipiv), work, *lwork, info);
    else if constexpr(isDouble())
        magma_dgetri_gpu(*n, a, *lda, const_cast<int*>(ipiv), work, *lwork, info);
}

template<class floating>
void GPU_MAGMA<floating>::xgetrs(const OperationType trans, const int * const n, const int * const nrhs,
        const floating * const a, const int * const lda, const int * const ipiv,
        floating * const b, const int * const ldb, int * const info) const
{
    if constexpr(isFloat())
        magma_sgetrs_gpu(toInternalOperation.at(trans), *n, *nrhs, const_cast<float*>(a), *lda, const_cast<int*>(ipiv), b, *ldb, info);
    else if constexpr(isDouble())
        magma_dgetrs_gpu(toInternalOperation.at(trans), *n, *nrhs, const_cast<double*>(a), *lda, const_cast<int*>(ipiv), b, *ldb, info);

    cudaDeviceSynchronize();
    return;
}

template<class floating>
void GPU_MAGMA<floating>::xscal(const int N, const floating alpha, floating * const X, const int incX) const
{
    if constexpr(isFloat())
        magma_sscal(N, alpha, X, incX, getMagmaQueue());
    else if constexpr(isDouble())
        magma_dscal(N, alpha, X, incX, getMagmaQueue());

    cudaDeviceSynchronize();
    return;
}

template<class floating>
std::map<OperationType, magma_trans_t> GPU_MAGMA<floating>::toInternalOperation = {
        {OperationType::Identical, MagmaNoTrans},
        {OperationType::Transposed, MagmaTrans},
        {OperationType::Hermitian, MagmaConjTrans}
};

template<class floating>
GPU_MAGMA_Queue GPU_MAGMA<floating>::_queue;

template<class floating>
MemoryManager GPU_MAGMA<floating>::_deviceManager = std::make_shared<GPU_Manager>();
