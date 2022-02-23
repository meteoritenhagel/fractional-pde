#include <cublas_v2.h>
#include <cusolverDn.h>

#include <cassert>

template<class floating>
Timer GpuMagma<floating>::createTimer() const
{
    return std::make_unique<GpuTimer>();
}

template<class floating>
MemoryManager GpuMagma<floating>::get_memory_manager() const
{
    return _deviceManager;
}

template<class floating>
void GpuMagma<floating>::display() const
{
    return "GpuMagma";
}

template<class floating>
auto GpuMagma<floating>::get_magma_queue() const
{
    return _queue.get_magma_queue();
}

template<class floating>
int GpuMagma<floating>::ixamax(const int n, const floating * const x, const int incx) const
{
    int output = 0;
    if constexpr(isFloat())
        output = magma_isamax(n, x, incx, get_magma_queue());
    else if constexpr(isDouble())
        output = magma_idamax(n, x, incx, get_magma_queue());
    return output-1; // Since MAGMA uses Fortran indexing here...
}

template<class floating>
void GpuMagma<floating>::xaxpy(const int n, const floating alpha, const floating * const x, const int incx, floating * const y, const int incy) const
{
    if constexpr(isFloat())
        magma_saxpy(n, alpha, x, incx, y, incy, get_magma_queue());
    else if constexpr(isDouble())
        magma_daxpy(n, alpha, x, incx, y, incy, get_magma_queue());
    cudaDeviceSynchronize();
    return;
}

template<class floating>
void GpuMagma<floating>::xcopy(const int n, const floating * const source, const int inc_source, floating * const dest, const int inc_dest) const
{
    if constexpr(isFloat())
        magma_scopy(n, source, inc_source, dest, inc_dest, get_magma_queue());
    else if constexpr(isDouble())
        magma_dcopy(n, source, inc_source, dest, inc_dest, get_magma_queue());
    cudaDeviceSynchronize();
    return;
}

template<class floating>
floating GpuMagma<floating>::xdot(const int n, const floating * const x, const int incx, const floating * const y, const int incy) const
{
    floating result = 0;
    if constexpr(isFloat())
        result = magma_sdot(n, x, incx, y, incy, get_magma_queue());
    else if constexpr(isDouble())
        result = magma_ddot(n, x, incx, y, incy, get_magma_queue());

    return result;
}

template<class floating>
void GpuMagma<floating>::xgemm(const OperationType trans_A, const OperationType trans_B,
                               const int M, const int N, const int K, const floating alpha, const floating * const A,
                               const int lda, const floating * const B, const int ldb, const floating beta, floating * const C, const int ldc) const
{
    if constexpr(isFloat())
        magma_sgemm(to_internal_operation.at(trans_A), to_internal_operation.at(trans_B),
                    M, N, K, alpha, A,
                    lda, B, ldb, beta, C, ldc, get_magma_queue());
    else if constexpr(isDouble())
        magma_dgemm(to_internal_operation.at(trans_A), to_internal_operation.at(trans_B),
                    M, N, K, alpha, A,
                    lda, B, ldb, beta, C, ldc, get_magma_queue());

    cudaDeviceSynchronize();
    return;
}

template<class floating>
void GpuMagma<floating>::xgemv(const OperationType trans, const int m, const int n,
                               const floating alpha, floating const * const A, const int lda, floating const * const x, const int incx,
                               const floating beta, floating * const y, const int incy) const
{
    if constexpr(isFloat())
        magma_sgemv(to_internal_operation.at(trans), m, n,
                    alpha, A, lda, x, incx,
                    beta, y, incy, get_magma_queue());
    else if constexpr(isDouble())
        magma_dgemv(to_internal_operation.at(trans), m, n,
                    alpha, A, lda, x, incx,
                    beta, y, incy, get_magma_queue());
    cudaDeviceSynchronize();
    return;
}


template<class floating>
void GpuMagma<floating>::xgetrf(int * const m, int * const n, floating * const A, int * const lda,
                                int * const ipiv, int * const info) const
{
    if constexpr(isFloat())
        magma_sgetrf(*m, *n, A, *lda, ipiv, info);
    else if constexpr(isDouble())
        magma_dgetrf(*m, *n, A, *lda, ipiv, info);

    cudaDeviceSynchronize();
    return;
}

template<class floating>
void GpuMagma<floating>::xgetri(const int * const n, floating * const A, const int * const lda, const int * const ipiv,
                                floating * const work, const int * const lwork, int * const info) const
{
    if constexpr(isFloat())
        magma_sgetri_gpu(*n, A, *lda, const_cast<int*>(ipiv), work, *lwork, info);
    else if constexpr(isDouble())
        magma_dgetri_gpu(*n, A, *lda, const_cast<int*>(ipiv), work, *lwork, info);
}

template<class floating>
void GpuMagma<floating>::xgetrs(const OperationType trans, const int * const n, const int * const nrhs,
                                const floating * const A, const int * const lda, const int * const ipiv,
                                floating * const B, const int * const ldb, int * const info) const
{
    if constexpr(isFloat())
        magma_sgetrs_gpu(to_internal_operation.at(trans), *n, *nrhs, const_cast<float*>(A), *lda, const_cast<int*>(ipiv), B, *ldb, info);
    else if constexpr(isDouble())
        magma_dgetrs_gpu(to_internal_operation.at(trans), *n, *nrhs, const_cast<double*>(A), *lda, const_cast<int*>(ipiv), B, *ldb, info);

    cudaDeviceSynchronize();
    return;
}

template<class floating>
void GpuMagma<floating>::xscal(const int N, const floating alpha, floating * const x, const int incx) const
{
    if constexpr(isFloat())
        magma_sscal(N, alpha, x, incx, get_magma_queue());
    else if constexpr(isDouble())
        magma_dscal(N, alpha, x, incx, get_magma_queue());

    cudaDeviceSynchronize();
    return;
}

template<class floating>
std::map<OperationType, magma_trans_t> GpuMagma<floating>::to_internal_operation = {
        {OperationType::Identical, MagmaNoTrans},
        {OperationType::Transposed, MagmaTrans},
        {OperationType::Hermitian, MagmaConjTrans}
};

template<class floating>
GpuMagma_Queue GpuMagma<floating>::_queue;

template<class floating>
MemoryManager GpuMagma<floating>::_deviceManager = std::make_shared<GPU_Manager>();
