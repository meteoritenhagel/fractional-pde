template<class floating>
Timer Cpu<floating>::create_timer() const
{
    return std::make_unique<OmpTimer>();
}

template<class floating>
MemoryManager Cpu<floating>::get_memory_manager() const
{
    return _device_manager;
}

template<class floating>
std::string Cpu<floating>::display() const
{
    return "Cpu";
}

template<class floating>
int Cpu<floating>::ixamax(const int n, const floating * const x, const int incx) const
{
    int output = 0;
    if constexpr(isFloat())
        output = cblas_isamax(n, x, incx);
    else if constexpr(isDouble())
        output = cblas_idamax(n, x, incx);
    return output;
}

template<class floating>
void Cpu<floating>::xaxpy(const int n, const floating alpha, const floating * const x, const int incx, floating * const y, const int incy) const
{
    if constexpr(isFloat())
        cblas_saxpy(n, alpha, x, incx, y, incy);
    else if constexpr(isDouble())
        cblas_daxpy(n, alpha, x, incx, y, incy);

    return;
}

template<class floating>
void Cpu<floating>::xcopy(const int n, const floating * const source, const int inc_source, floating * const dest, const int inc_dest) const
{
    if constexpr(isFloat())
        cblas_scopy(n, source, inc_source, dest, inc_dest);
    else if constexpr(isDouble())
        cblas_dcopy(n, source, inc_source, dest, inc_dest);

    return;
}

template<class floating>
floating Cpu<floating>::xdot(const int n, const floating * const x, const int incx, const floating * const y, const int incy) const
{
    floating result = 0;
    if constexpr(isFloat())
        result = cblas_sdot(n, x, incx, y, incy);
    else if constexpr(isDouble())
        result = cblas_ddot(n, x, incx, y, incy);

    return result;
}

template<class floating>
void Cpu<floating>::xgemm(const OperationType trans_A, const OperationType trans_B,
                          const int M, const int N, const int K, const floating alpha, const floating * const A,
                          const int lda, const floating * const B, const int ldb, const floating beta, floating * const C, const int ldc) const
{
    if constexpr(isFloat())
        cblas_sgemm(CblasColMajor, to_internal_operation_blas.at(trans_A), to_internal_operation_blas.at(trans_B),
                    M, N, K, alpha, A,
                    lda, B, ldb, beta, C, ldc);
    else if constexpr(isDouble())
        cblas_dgemm(CblasColMajor, to_internal_operation_blas.at(trans_A), to_internal_operation_blas.at(trans_B),
                    M, N, K, alpha, A,
                    lda, B, ldb, beta, C, ldc);
    return;
}

template<class floating>
void Cpu<floating>::xgemv(const OperationType trans, const int m, const int n,
                          const floating alpha, floating const * const A, const int lda, floating const * const x, const int incx,
                          const floating beta, floating * const y, const int incy) const
{
    if constexpr(isFloat())
        cblas_sgemv(CblasColMajor, to_internal_operation_blas.at(trans), m, n,
                    alpha, A, lda, x, incx,
                    beta, y, incy);
    else if constexpr(isDouble())
        cblas_dgemv(CblasColMajor, to_internal_operation_blas.at(trans), m, n,
                    alpha, A, lda, x, incx,
                    beta, y, incy);

    return;
}


template<class floating>
void Cpu<floating>::xgetrf(int * const m, int * const n, floating * const A, int * const lda,
                           int * const ipiv, int * const info) const
{
    if constexpr(isFloat())
        sgetrf_(m, n, A, lda, ipiv, info);
    else if constexpr(isDouble())
        dgetrf_(m, n, A, lda, ipiv, info);

    return;
}

template<class floating>
void Cpu<floating>::xgetri(const int * const n, floating * const A, const int * const lda, const int * const ipiv,
                           floating * const work, const int * const lwork, int * const info) const
{
    if constexpr(isFloat())
        sgetri_(n, A, lda, ipiv, work, lwork, info);
    else if constexpr(isDouble())
        dgetri_(n, A, lda, ipiv, work, lwork, info);
}

template<class floating>
void Cpu<floating>::xgetrs(const OperationType trans, const int * const n, const int * const nrhs,
                           const floating * const A, const int * const lda, const int * const ipiv,
                           floating * const B, const int * const ldb, int * const info) const
{
    if constexpr(isFloat())
        sgetrs_(to_internal_operation_lapack.at(trans), n, nrhs, A, lda, ipiv, B, ldb, info);
    else if constexpr(isDouble())
        dgetrs_(to_internal_operation_lapack.at(trans), n, nrhs, A, lda, ipiv, B, ldb, info);

    return;
}

template<class floating>
void Cpu<floating>::xscal(const int N, const floating alpha, floating * const x, const int incx) const
{
    if constexpr(isFloat())
        cblas_sscal(N, alpha, x, incx);
    else if constexpr(isDouble())
        cblas_dscal(N, alpha, x, incx);

    return;
}

template<class floating>
std::map<OperationType, CBLAS_TRANSPOSE> Cpu<floating>::to_internal_operation_blas = {
        {OperationType::Identical, CblasNoTrans},
        {OperationType::Transposed, CblasTrans},
        {OperationType::Hermitian, CblasConjTrans}
};

template<class floating>
std::map<OperationType, const char * const> Cpu<floating>::to_internal_operation_lapack = {
        {OperationType::Identical, "N"},
        {OperationType::Transposed, "T"},
        {OperationType::Hermitian, "C"}
};

template<class floating>
MemoryManager Cpu<floating>::_device_manager = std::make_shared<CpuManager>();
