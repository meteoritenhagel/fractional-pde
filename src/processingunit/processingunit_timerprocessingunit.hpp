#include <cassert>

template<class floating>
TimerProcessingUnit<floating>::TimerProcessingUnit(const ProcessingUnit<floating> processingUnit)
: _processingUnit(processingUnit) {}

template<class floating>
Timer TimerProcessingUnit<floating>::createTimer() const
{
    return _processingUnit->createTimer();
}

template<class floating>
MemoryManager TimerProcessingUnit<floating>::getMemoryManager() const
{
    return _processingUnit->getMemoryManager();
}

template<class floating>
void TimerProcessingUnit<floating>::display() const
{
    std::cout << "TimerProcessingUnit with ";
    _processingUnit->display();

    return;
}

template<class floating>
int TimerProcessingUnit<floating>::ixamax(const int n, const floating * const x, const int incx) const
{
    return _processingUnit->ixamax(n, x, incx);

}

template<class floating>
void TimerProcessingUnit<floating>::xaxpy(const int n, const floating alpha, const floating * const x, const int incx, floating * const y, const int incy, const bool synchronize) const
{
    _processingUnit->xaxpy(n, alpha, x, incx, y, incy, synchronize);
}

template<class floating>
void TimerProcessingUnit<floating>::xcopy(const int n, const floating * const source, const int incx, floating * const dest, const int incy) const
{
    _processingUnit->xcopy(n, source, incx, dest, incy);
    return;
}

template<class floating>
floating TimerProcessingUnit<floating>::xdot(const int n, const floating * const x, const int incx, const floating * const y, const int incy) const
{
    floating result = 0;
    result = _processingUnit->xdot(n, x, incx, y, incy);

    return result;
}

template<class floating>
void TimerProcessingUnit<floating>::xgemm(const OperationType TransA, const OperationType TransB,
        const int M, const int N, const int K, const floating alpha, const floating * const A,
        const int lda, const floating * const B, const int ldb, const floating beta, floating * const C, const int ldc) const
{
    _processingUnit->xgemm(TransA, TransB, M, N, K, alpha, A,  lda, B, ldb, beta, C, ldc);
    return;
}

template<class floating>
void TimerProcessingUnit<floating>::xgemv(const OperationType trans, const int m, const int n,
        const floating alpha, floating const * const a, const int lda, floating const * const x, const int incx,
        const floating beta, floating * const y, const int incy) const
{
    _processingUnit->xgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);

    return;
}


template<class floating>
void TimerProcessingUnit<floating>::xgetrf(int * const m, int * const n, floating * const a, int * const lda,
        int * const ipiv, int * const info) const
{
    _processingUnit->xgetrf(m, n, a, lda, ipiv, info);
    return;
}

template<class floating>
void TimerProcessingUnit<floating>::xgetri(const int * const n, floating * const a, const int * const lda, const int * const ipiv,
                           floating * const work, const int * const lwork, int * const info) const
{
    _processingUnit->xgetri(n, a, lda, ipiv, work, lwork, info);
    return;
}

template<class floating>
void TimerProcessingUnit<floating>::xgetrs(const OperationType trans, const int * const n, const int * const nrhs,
        const floating * const a, const int * const lda, const int * const ipiv,
        floating * const b, const int * const ldb, int * const info) const
{
    _processingUnit->xgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb, info);
    return;
}

template<class floating>
void TimerProcessingUnit<floating>::xscal(const int N, const floating alpha, floating * const X, const int incX) const
{
    _processingUnit->xscal(N, alpha, X, incX);
    return;
}
