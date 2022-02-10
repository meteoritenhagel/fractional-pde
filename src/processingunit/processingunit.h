#ifndef FILE_PROCESSINGUNIT
#define FILE_PROCESSINGUNIT

#include "../devicedata/memorymanager.h"
#include "timer.h"

#ifndef CPU_ONLY
    #include "gpu_handle.h"

#ifdef MAGMA
    #include "gpu_magma_queue.h"
    #include "magma_v2.h"
#endif
#endif

#ifdef __INTEL_CLANG_COMPILER
  #pragma message(" ##########  Use of MKL  ###############")
  #include <mkl.h>
#else
  #pragma message(" ##########  Use of CBLAS  ###############")
  #include <cblas.h>
  #include <lapacke.h>
#endif

#include <map>
#include <type_traits>
#include <iostream>
#include <memory>

template<class floating>
class ProcessingUnitDevice;

template<class floating>
using ProcessingUnit = std::shared_ptr<ProcessingUnitDevice<floating>>;

enum class OperationType {
    Identical,
    Transposed,
    Hermitian
};

template<class floating>
class ProcessingUnitDevice {
public:
    ProcessingUnitDevice();
    virtual ~ProcessingUnitDevice() = default;

    virtual Timer createTimer() const = 0;
    virtual MemoryManager getMemoryManager() const = 0;
    virtual void display() const {};     // GH: needed to compile with ICC_CPU_ONLY_

    virtual int ixamax(const int n, const floating * const x, const int incx) const = 0;
    virtual void xaxpy(const int n, const floating alpha, const floating * const x, const int incx, floating * const y, const int incy, const bool synchronize = true) const = 0;
    virtual void xcopy(const int n, const floating * const source, const int incx, floating * const dest, const int incy) const = 0;
    virtual floating xdot(const int n, const floating * const x, const int incx, const floating * const y, const int incy) const = 0;
    virtual void xgemm(const OperationType TransA, const OperationType TransB,
                       const int M, const int N, const int K, const floating alpha, const floating * const A,
                       const int lda, const floating * const B, const int ldb, const floating beta, floating * const C, const int ldc) const = 0;
    virtual void xgemv(const OperationType trans, const int m, const int n,
                       const floating alpha, floating const * const a, const int lda, floating const * const x, const int incx,
                       const floating beta, floating * const y, const int incy) const = 0;
    virtual void xgetrf(int * const m, int * const n, floating * const a, int * const lda,
                        int * const ipiv, int * const info) const = 0;
    virtual void xgetri(const int * const n, floating * const a, const int * const lda, const int * const ipiv,
                        floating * const work, const int * const lwork, int * const info) const = 0;
    virtual void xgetrs(const OperationType trans, const int * const n, const int * const nrhs,
                        const floating * const a, const int * const lda, const int * const ipiv,
                        floating * const b, const int * const ldb, int * const info) const = 0;
    virtual void xscal(const int N, const floating alpha, floating * const X, const int incX) const = 0;

    constexpr static bool isFloat();
    constexpr static bool isDouble();

    // TODO: REMOVE
    TimerDevice::timespan_sec _multigridTime{0};
    TimerDevice::timespan_sec _factorizationTime{0};
};


template<class floating>
class CPU : public ProcessingUnitDevice<floating> {
public:
    using ProcessingUnitDevice<floating>::isFloat;
    using ProcessingUnitDevice<floating>::isDouble;

    CPU() = default;
    ~CPU() override = default;

    Timer createTimer() const override;
    MemoryManager getMemoryManager() const override;
    void display() const override;

    int ixamax(const int n, const floating * const x, const int incx) const override;
    void xaxpy(const int n, const floating alpha, const floating * const x, const int incx, floating * const y, const int incy, const bool synchronize = true) const override;
    void xcopy(const int n, const floating * const source, const int incx, floating * const dest, const int incy) const override;
    floating xdot(const int n, const floating * const x, const int incx, const floating * const y, const int incy) const override;
    void xgemm(const OperationType TransA, const OperationType TransB,
               const int M, const int N, const int K, const floating alpha, const floating * const A,
               const int lda, const floating * const B, const int ldb, const floating beta, floating * const C, const int ldc) const override;
    void xgemv(const OperationType trans, const int m, const int n,
               const floating alpha, floating const * const a, const int lda, floating const * const x, const int incx,
               const floating beta, floating * const y, const int incy) const override;
    void xgetrf(int * const m, int * const n, floating * const a, int * const lda,
                int * const ipiv, int * const info) const override;
	void xgetri(const int * const n, floating * const a, const int * const lda, const int * const ipiv,
		floating * const work, const int * const lwork, int * const info) const override;
    void xgetrs(const OperationType trans, const int * const n, const int * const nrhs,
                const floating * const a, const int * const lda, const int * const ipiv,
                floating * const b, const int * const ldb, int * const info) const override;
    void xscal(const int N, const floating alpha, floating * const X, const int incX) const override;


private:
    static std::map<OperationType, CBLAS_TRANSPOSE> toInternalOperationBLAS;
    static std::map<OperationType, const char * const> toInternalOperationLAPACK;
    static MemoryManager _deviceManager;
};

#ifndef CPU_ONLY
template<class floating>
class GPU : public ProcessingUnitDevice<floating> {
public:
    using ProcessingUnitDevice<floating>::isFloat;
    using ProcessingUnitDevice<floating>::isDouble;

    GPU() = default;
    ~GPU() override = default;

    Timer createTimer() const override;
    MemoryManager getMemoryManager() const override;
    void display() const override;
    auto getCublasHandle() const;
    auto getCusolverHandle() const;

    int ixamax(const int n, const floating * const x, const int incx) const override;
    void xaxpy(const int n, const floating alpha, const floating * const x, const int incx, floating * const y, const int incy, const bool synchronize = true) const override;
    void xcopy(const int n, const floating * const source, const int incx, floating * const dest, const int incy) const override;
    floating xdot(const int n, const floating * const x, const int incx, const floating * const y, const int incy) const override;
    void xgemm(const OperationType TransA, const OperationType TransB,
               const int M, const int N, const int K, const floating alpha, const floating * const A,
               const int lda, const floating * const B, const int ldb, const floating beta, floating * const C, const int ldc) const override;
    void xgemv(const OperationType trans, const int m, const int n,
               const floating alpha, floating const * const a, const int lda, floating const * const x, const int incx,
               const floating beta, floating * const y, const int incy) const override;
    void xgetrf(int * const m, int * const n, floating * const a, int * const lda,
                int * const ipiv, int * const info) const override;
	void xgetri(const int * const n, floating * const a, const int * const lda, const int * const ipiv,
				floating * const work, const int * const lwork, int * const info) const override;
    void xgetrs(const OperationType trans, const int * const n, const int * const nrhs,
                const floating * const a, const int * const lda, const int * const ipiv,
                floating * const b, const int * const ldb, int * const info) const override;
    void xscal(const int N, const floating alpha, floating * const X, const int incX) const override;


private:
    static std::map<OperationType, cublasOperation_t> toInternalOperation;
    static GPU_Handle _handle;
    static MemoryManager _deviceManager;

};

#ifdef MAGMA
template<class floating>
class GPU_MAGMA : public ProcessingUnitDevice<floating> {
public:
    using ProcessingUnitDevice<floating>::isFloat;
    using ProcessingUnitDevice<floating>::isDouble;

    GPU_MAGMA() = default;
    ~GPU_MAGMA() override = default;

    Timer createTimer() const override;
    MemoryManager getMemoryManager() const override;
    void display() const override;
    auto getMagmaQueue() const;

    int ixamax(const int n, const floating * const x, const int incx) const override;
    void xaxpy(const int n, const floating alpha, const floating * const x, const int incx, floating * const y, const int incy, const bool synchronize = true) const override;
    void xcopy(const int n, const floating * const source, const int incx, floating * const dest, const int incy) const override;
    floating xdot(const int n, const floating * const x, const int incx, const floating * const y, const int incy) const override;
    void xgemm(const OperationType TransA, const OperationType TransB,
               const int M, const int N, const int K, const floating alpha, const floating * const A,
               const int lda, const floating * const B, const int ldb, const floating beta, floating * const C, const int ldc) const override;
    void xgemv(const OperationType trans, const int m, const int n,
               const floating alpha, floating const * const a, const int lda, floating const * const x, const int incx,
               const floating beta, floating * const y, const int incy) const override;
    void xgetrf(int * const m, int * const n, floating * const a, int * const lda,
                int * const ipiv, int * const info) const override;
	void xgetri(const int * const n, floating * const a, const int * const lda, const int * const ipiv,
				floating * const work, const int * const lwork, int * const info) const override;
    void xgetrs(const OperationType trans, const int * const n, const int * const nrhs,
                const floating * const a, const int * const lda, const int * const ipiv,
                floating * const b, const int * const ldb, int * const info) const override;
    void xscal(const int N, const floating alpha, floating * const X, const int incX) const override;


private:
    static std::map<OperationType, magma_trans_t> toInternalOperation;
    static GPU_MAGMA_Queue _queue;
    static MemoryManager _deviceManager;
};

enum class ReplacementNumber {
    IXAMAX,
    XAXPY,
    XCOPY,
    XDOT,
    XGEMM,
    XGEMV,
    XGETRF,
    XGETRS,
    XSCAL
};

template<class floating>
class GPU_MIXED : public ProcessingUnitDevice<floating> {
public:
    GPU_MIXED(const ReplacementNumber replacementNumber = ReplacementNumber::XGETRF);
    ~GPU_MIXED() override = default;

    Timer createTimer() const override;
    MemoryManager getMemoryManager() const override;
    void display() const override;

    int ixamax(const int n, const floating * const x, const int incx) const override;
    void xaxpy(const int n, const floating alpha, const floating * const x, const int incx, floating * const y, const int incy, const bool synchronize = true) const override;
    void xcopy(const int n, const floating * const source, const int incx, floating * const dest, const int incy) const override;
    floating xdot(const int n, const floating * const x, const int incx, const floating * const y, const int incy) const override;
    void xgemm(const OperationType TransA, const OperationType TransB,
               const int M, const int N, const int K, const floating alpha, const floating * const A,
               const int lda, const floating * const B, const int ldb, const floating beta, floating * const C, const int ldc) const override;
    void xgemv(const OperationType trans, const int m, const int n,
               const floating alpha, floating const * const a, const int lda, floating const * const x, const int incx,
               const floating beta, floating * const y, const int incy) const override;
    void xgetrf(int * const m, int * const n, floating * const a, int * const lda,
                int * const ipiv, int * const info) const override;
	void xgetri(const int * const n, floating * const a, const int * const lda, const int * const ipiv,
				floating * const work, const int * const lwork, int * const info) const override;
    void xgetrs(const OperationType trans, const int * const n, const int * const nrhs,
                const floating * const a, const int * const lda, const int * const ipiv,
                floating * const b, const int * const ldb, int * const info) const override;
    void xscal(const int N, const floating alpha, floating * const X, const int incX) const override;

    ReplacementNumber getReplacementNumber() const;

private:
    static std::map<OperationType, magma_trans_t> toInternalOperation;
    static GPU<floating> _gpu;
    static GPU_MAGMA<floating> _gpu_magma;
    ReplacementNumber _replacementNumber;
};
#endif
#endif

#include "processingunit.hpp"
#include "processingunit_cpu.hpp"

template<class floating>
class TimerProcessingUnit : public ProcessingUnitDevice<floating> {
public:
    TimerProcessingUnit(const ProcessingUnit<floating> processingUnit);
    ~TimerProcessingUnit() override = default;

    Timer createTimer() const override;
    MemoryManager getMemoryManager() const override;
    void display() const override;

    int ixamax(const int n, const floating * const x, const int incx) const override;
    void xaxpy(const int n, const floating alpha, const floating * const x, const int incx, floating * const y, const int incy, const bool synchronize = true) const override;
    void xcopy(const int n, const floating * const source, const int incx, floating * const dest, const int incy) const override;
    floating xdot(const int n, const floating * const x, const int incx, const floating * const y, const int incy) const override;
    void xgemm(const OperationType TransA, const OperationType TransB,
               const int M, const int N, const int K, const floating alpha, const floating * const A,
               const int lda, const floating * const B, const int ldb, const floating beta, floating * const C, const int ldc) const override;
    void xgemv(const OperationType trans, const int m, const int n,
               const floating alpha, floating const * const a, const int lda, floating const * const x, const int incx,
               const floating beta, floating * const y, const int incy) const override;
    void xgetrf(int * const m, int * const n, floating * const a, int * const lda,
                int * const ipiv, int * const info) const override;
	void xgetri(const int * const n, floating * const a, const int * const lda, const int * const ipiv,
				floating * const work, const int * const lwork, int * const info) const override;
    void xgetrs(const OperationType trans, const int * const n, const int * const nrhs,
                const floating * const a, const int * const lda, const int * const ipiv,
                floating * const b, const int * const ldb, int * const info) const override;
    void xscal(const int N, const floating alpha, floating * const X, const int incX) const override;

private:
    ProcessingUnit<floating> _processingUnit;
};

#include "processingunit_timerprocessingunit.hpp"

#ifndef CPU_ONLY
#include "processingunit_gpu.hpp"

#ifdef MAGMA
#include "processingunit_gpu_magma.hpp"
#include "processingunit_gpu_mixed.hpp"
#endif

#endif

#endif
