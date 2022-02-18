#include "gpu_data.h"

#include <cublas_v2.h>
#include <cusolverDn.h>

#include <cassert>

template<class floating>
GPU_MIXED<floating>::GPU_MIXED(const ReplacementNumber replacementNumber)
: _replacementNumber(replacementNumber) {}

template<class floating>
Timer GPU_MIXED<floating>::createTimer() const
{
    return _gpu.createTimer();
}

template<class floating>
MemoryManager GPU_MIXED<floating>::getMemoryManager() const
{
    return _gpu._memoryManager;
}

template<class floating>
void GPU_MIXED<floating>::display() const
{
    return "GPU_MIXED";
}

template<class floating>
int GPU_MIXED<floating>::ixamax(const int n, const floating * const x, const int incx) const
{
    if (getReplacementNumber() == ReplacementNumber::IXAMAX)
        return _gpu_magma.ixamax(n, x, incx);
    else
        return _gpu.ixamax(n, x, incx);
}

template<class floating>
void GPU_MIXED<floating>::xaxpy(const int n, const floating alpha, const floating * const x, const int incx, floating * const y, const int incy) const
{
    if (getReplacementNumber() == ReplacementNumber::XAXPY)
        _gpu_magma.xaxpy(n, alpha, x, incx, y, incy);
    else
        _gpu.xaxpy(n, alpha, x, incx, y, incy);
    return;
}

template<class floating>
void GPU_MIXED<floating>::xcopy(const int n, const floating * const source, const int incx, floating * const dest, const int incy) const
{
    if constexpr(getReplacementNumber() == ReplacementNumber::XCOPY)
        _gpu_magma.xcopy(n, source, incx, dest, incy);
    else
        _gpu.xcopy(n, source, incx, dest, incy);
    cudaDeviceSynchronize();
    return;
}

template<class floating>
floating GPU_MIXED<floating>::xdot(const int n, const floating * const x, const int incx, const floating * const y, const int incy) const
{
    floating result = 0;
    if (getReplacementNumber() == ReplacementNumber::XDOT)
        result = _gpu_magma.xdot(n, x, incx, y, incy);
    else
        result = _gpu.xdot(n, x, incx, y, incy);

    return result;
}

template<class floating>
void GPU_MIXED<floating>::xgemm(const OperationType TransA, const OperationType TransB,
        const int M, const int N, const int K, const floating alpha, const floating * const A,
        const int lda, const floating * const B, const int ldb, const floating beta, floating * const C, const int ldc) const
{
    if (getReplacementNumber() == ReplacementNumber::XGEMM)
        _gpu_magma.xgemm(TransA, TransB, M, N, K, alpha, A,  lda, B, ldb, beta, C, ldc);
    else
        _gpu.xgemm(TransA, TransB, M, N, K, alpha, A,  lda, B, ldb, beta, C, ldc);
    return;
}

template<class floating>
void GPU_MIXED<floating>::xgemv(const OperationType trans, const int m, const int n,
        const floating alpha, floating const * const a, const int lda, floating const * const x, const int incx,
        const floating beta, floating * const y, const int incy) const
{
    if (getReplacementNumber() == ReplacementNumber::XGEMV)
        _gpu_magma.xgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    else
        _gpu.xgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    return;
}


template<class floating>
void GPU_MIXED<floating>::xgetrf(int * const m, int * const n, floating * const a, int * const lda,
        int * const ipiv, int * const info) const
{
    if (getReplacementNumber() == ReplacementNumber::XGETRF)
        _gpu_magma.xgetrf(m, n, a, lda, ipiv, info);
    else
        _gpu.xgetrf(m, n, a, lda, ipiv, info);
    return;
}

template<class floating>
void GPU_MIXED<floating>::xgetri(const int * const n, floating * const a, const int * const lda, const int * const ipiv,
						   floating * const work, const int * const lwork, int * const info) const
{
	// is always on MAGMA
	_gpu_magma.xgetri(n, a, lda, ipiv, work, lwork, info);
}


template<class floating>
void GPU_MIXED<floating>::xgetrs(const OperationType trans, const int * const n, const int * const nrhs,
        const floating * const a, const int * const lda, const int * const ipiv,
        floating * const b, const int * const ldb, int * const info) const
{
    if (getReplacementNumber() == ReplacementNumber::XGETRS)
        _gpu_magma.xgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb, info);
    else
        _gpu.xgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb, info);
    return;
}

template<class floating>
void GPU_MIXED<floating>::xscal(const int N, const floating alpha, floating * const X, const int incX) const
{
    if (getReplacementNumber() == ReplacementNumber::XSCAL)
        _gpu_magma.xscal(N, alpha, X, incX);
    else
        _gpu.xscal(N, alpha, X, incX);
    return;
}

template<class floating>
ReplacementNumber GPU_MIXED<floating>::getReplacementNumber() const
{
    return _replacementNumber;
}

template<class floating>
GPU<floating> GPU_MIXED<floating>::_gpu;

template<class floating>
GPU_MAGMA<floating> GPU_MIXED<floating>::_gpu_magma;
