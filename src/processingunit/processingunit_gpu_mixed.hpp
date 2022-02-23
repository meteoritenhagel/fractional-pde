#include "gpu_data.h"

#include <cublas_v2.h>
#include <cusolverDn.h>

#include <cassert>

template<class floating>
GpuMixed<floating>::GpuMixed(const ReplacementNumber replacementNumber)
: _replacementNumber(replacementNumber) {}

template<class floating>
Timer GpuMixed<floating>::createTimer() const
{
    return _gpu.create_timer();
}

template<class floating>
MemoryManager GpuMixed<floating>::get_memory_manager() const
{
    return _gpu._memoryManager;
}

template<class floating>
void GpuMixed<floating>::display() const
{
    return "GpuMixed";
}

template<class floating>
int GpuMixed<floating>::ixamax(const int n, const floating * const x, const int incx) const
{
    if (get_replacement_number() == ReplacementNumber::IXAMAX)
        return _gpu_magma.ixamax(n, x, incx);
    else
        return _gpu.ixamax(n, x, incx);
}

template<class floating>
void GpuMixed<floating>::xaxpy(const int n, const floating alpha, const floating * const x, const int incx, floating * const y, const int incy) const
{
    if (get_replacement_number() == ReplacementNumber::XAXPY)
        _gpu_magma.xaxpy(n, alpha, x, incx, y, incy);
    else
        _gpu.xaxpy(n, alpha, x, incx, y, incy);
    return;
}

template<class floating>
void GpuMixed<floating>::xcopy(const int n, const floating * const source, const int inc_source, floating * const dest, const int inc_dest) const
{
    if constexpr(get_replacement_number() == ReplacementNumber::XCOPY)
        _gpu_magma.xcopy(n, source, inc_source, dest, inc_dest);
    else
        _gpu.xcopy(n, source, inc_source, dest, inc_dest);
    cudaDeviceSynchronize();
    return;
}

template<class floating>
floating GpuMixed<floating>::xdot(const int n, const floating * const x, const int incx, const floating * const y, const int incy) const
{
    floating result = 0;
    if (get_replacement_number() == ReplacementNumber::XDOT)
        result = _gpu_magma.xdot(n, x, incx, y, incy);
    else
        result = _gpu.xdot(n, x, incx, y, incy);

    return result;
}

template<class floating>
void GpuMixed<floating>::xgemm(const OperationType trans_A, const OperationType trans_B,
                               const int M, const int N, const int K, const floating alpha, const floating * const A,
                               const int lda, const floating * const B, const int ldb, const floating beta, floating * const C, const int ldc) const
{
    if (get_replacement_number() == ReplacementNumber::XGEMM)
        _gpu_magma.xgemm(trans_A, trans_B, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    else
        _gpu.xgemm(trans_A, trans_B, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    return;
}

template<class floating>
void GpuMixed<floating>::xgemv(const OperationType trans, const int m, const int n,
                               const floating alpha, floating const * const A, const int lda, floating const * const x, const int incx,
                               const floating beta, floating * const y, const int incy) const
{
    if (get_replacement_number() == ReplacementNumber::XGEMV)
        _gpu_magma.xgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    else
        _gpu.xgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    return;
}


template<class floating>
void GpuMixed<floating>::xgetrf(int * const m, int * const n, floating * const A, int * const lda,
                                int * const ipiv, int * const info) const
{
    if (get_replacement_number() == ReplacementNumber::XGETRF)
        _gpu_magma.xgetrf(m, n, A, lda, ipiv, info);
    else
        _gpu.xgetrf(m, n, A, lda, ipiv, info);
    return;
}

template<class floating>
void GpuMixed<floating>::xgetri(const int * const n, floating * const A, const int * const lda, const int * const ipiv,
                                floating * const work, const int * const lwork, int * const info) const
{
    // is always on MAGMA
    _gpu_magma.xgetri(n, A, lda, ipiv, work, lwork, info);
}


template<class floating>
void GpuMixed<floating>::xgetrs(const OperationType trans, const int * const n, const int * const nrhs,
                                const floating * const A, const int * const lda, const int * const ipiv,
                                floating * const B, const int * const ldb, int * const info) const
{
    if (get_replacement_number() == ReplacementNumber::XGETRS)
        _gpu_magma.xgetrs(trans, n, nrhs, A, lda, ipiv, B, ldb, info);
    else
        _gpu.xgetrs(trans, n, nrhs, A, lda, ipiv, B, ldb, info);
    return;
}

template<class floating>
void GpuMixed<floating>::xscal(const int N, const floating alpha, floating * const x, const int incx) const
{
    if (get_replacement_number() == ReplacementNumber::XSCAL)
        _gpu_magma.xscal(N, alpha, x, incx);
    else
        _gpu.xscal(N, alpha, x, incx);
    return;
}

template<class floating>
ReplacementNumber GpuMixed<floating>::get_replacement_number() const
{
    return _replacementNumber;
}

template<class floating>
Gpu<floating> GpuMixed<floating>::_gpu;

template<class floating>
GpuMagma<floating> GpuMixed<floating>::_gpu_magma;
