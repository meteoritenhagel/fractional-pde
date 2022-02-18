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
#else
    #pragma message("CPU only mode is activated. GPU suppport is disabled. If you wish to activate GPU support, remove compiler flag CPU_ONLY")
#endif

#ifdef __INTEL_CLANG_COMPILER
  #pragma message("Intel Math Kernel Library is used for acceleration of linear algebra operations.")
  #include <mkl.h>
#else
  #pragma message("CBLAS/LAPACK/LAPACKE are used for acceleration of linear algebra operations.")
  #include <cblas.h>
  #include <lapacke.h>
#endif

#include <map>
#include <type_traits>
#include <iostream>
#include <memory>
#include <string>

template<class floating>
class ProcessingUnitDevice;

template<class floating>
using ProcessingUnit = std::shared_ptr<ProcessingUnitDevice<floating>>;

/**
 * Class OperationType is used for choosing the matrix operation
 * type in several operations of interface ProcessingUnitDevice.
 */
enum class OperationType {
    Identical, //!< taking the matrix without any changes
    Transposed, //!< taking the transposition of the matrix
    Hermitian //!< taking the complex conjugate of the matrix
};

/**
 * Class ProcessingUnitDevice is an abstract interface for easy calls to
 * BLAS/LAPACK/LAPACKE-like function calls on a certain device,
 * e.g. for CUDA/cuSOLVER on a GPU, or for MAGMA on a GPU.
 *
 * The floating point type is passed as a template parameter.
 *
 * @tparam floating floating point type
 */
template<class floating>
class ProcessingUnitDevice {
public:
    /**
     * Constructor.
     */
    ProcessingUnitDevice();

    /**
     * Destructor.
     */
    virtual ~ProcessingUnitDevice() = default;

    /**
     * Creates a new instance of a Timer stopwatch on the current device.
     * @return new Timer instance
     */
    virtual Timer createTimer() const = 0;

    /**
     * Returns the MemoryManager associated with the current device.
     * @return MemoryManager
     */
    virtual MemoryManager getMemoryManager() const = 0;

    /**
     * Returns the subclass's name in a human-readable format as a string
     * @return subclass's human-readable name
     */
    virtual std::string display() const = 0;

    /**
     * Function ixamax is an abstraction of calls isamax/idamax/etc.
     * It returns the index of the first element having maximum absolute value.
     *
     * See also
     * https://www.netlib.org/lapack/explore-html/d0/d73/group__aux__blas_ga285793254ff0adaf58c605682efb880c.html
     *
     * @param[in] n number of elements of input vector
     * @param[in] x a C-style array containing (1 + (@param n-1) * abs(@param incx) elements of type @tparam floating
     * @param[in] incx storage spacing between elements of @param x
     * @return index of first element having maximum absolute value
     */
    virtual int ixamax(const int n, const floating * const x, const int incx) const = 0;

    /**
     * Function xaxpy is an abstraction of calls saxpy/daxpy/etc.
     * It performs the calculation y := alpha * x + y
     *
     * See also
     * https://www.netlib.org/lapack/explore-html/de/da4/group__double__blas__level1_ga8f99d6a644d3396aa32db472e0cfc91c.html
     *
     * @param[in] n number of elements in input vectors
     * @param[in] alpha scalar alpha
     * @param[in] x a C-style array containing (1 + (@param n-1) * abs(@param incx) elements of type @tparam floating
     * @param[in] incx storage spacing between elements of @param x
     * @param[in,out] y a C-style array containing (1 + (@param n-1) * abs(@param incy) elements of type @tparam floating
     * @param[in] incy storage spacing between elements of @param y
     */
    virtual void xaxpy(const int n, const floating alpha, const floating * const x, const int incx, floating * const y, const int incy) const = 0;

    /**
     * // TODO: Rename incx and incy or source and dest
     *
     * Function xcopy is an abstraction of calls scopy/dcopy/etc.
     * It performs the copy operation dest := source
     *
     * See also
     * https://www.netlib.org/lapack/explore-html/de/da4/group__double__blas__level1_ga21cdaae1732dea58194c279fca30126d.html
     * @param[in] n number of elements in input vectors
     * @param[in] source a C-style array containing (1 + (@param n-1) * abs(@param incx) elements of type @tparam floating
     * @param[in] incx storage spacing between elements of @param source
     * @param[out] dest a C-style array containing (1 + (@param n-1) * abs(@param incy) elements of type @tparam floating
     * @param[in] incy storage spacing between elements of @param dest
     */
    virtual void xcopy(const int n, const floating * const source, const int incx, floating * const dest, const int incy) const = 0;

    /**
     * Function xdot is an abstraction of calls sdot/ddot/etc.
     * It performs the dot product <x, y>
     *
     * See also
     * http://www.netlib.org/lapack/explore-html/de/da4/group__double__blas__level1_ga75066c4825cb6ff1c8ec4403ef8c843a.html
     *
     * @param[in] n number of elements in input vectors
     * @param[in] x a C-style array containing (1 + (@param n-1) * abs(@param incx) elements of type @tparam floating
     * @param[in] incx spacing between elements of @param x
     * @param[in,out] y a C-style array containing (1 + (@param n-1) * abs(@param incy) elements of type @tparam floating
     * @param[in] incy spacing between elements of @param y
     * @return the value of the dot product <x, y>
     */
    virtual floating xdot(const int n, const floating * const x, const int incx, const floating * const y, const int incy) const = 0;

    /** // TODO: Change to lower-case
     * Function xgemm is an abstraction of calls sgemm/dgemm/etc.
     * It performs the matrix product operation C := alpha * TransA(A) * TransB(B) + beta * C
     *
     * See also
     * http://www.netlib.org/lapack/explore-html/d1/d54/group__double__blas__level3_gaeda3cbd99c8fb834a60a6412878226e1.html
     *
     * @param[in] TransA defines the transformation applied to matrix A (see enum class OperationType)
     * @param[in] TransB defines the transformation applied to matrix B (see enum class OperationType)
     * @param[in] M the number of rows of matrix TransA(A) and C
     * @param[in] N the number of columns of matrix TransB(B) and C
     * @param[in] K the number of columns of matrix C
     * @param[in] alpha scalar applied to A
     * @param[in] A matrix stored as a column-wise C-style array containing elements of type @tparam floating
     * @param[in] lda the first dimension of A
     * @param[in] B matrix stored as a column-wise C-style array containing elements of type @tparam floating
     * @param[in] ldb the first dimension of B
     * @param[in] beta scalar applied to C
     * @param[in,out] C matrix stored as a column-wise C-style array containing elements of type @tparam floating
     * @param[in] ldc the first dimension of C
     */
    virtual void xgemm(const OperationType TransA, const OperationType TransB,
                       const int M, const int N, const int K, const floating alpha, const floating * const A,
                       const int lda, const floating * const B, const int ldb, const floating beta, floating * const C, const int ldc) const = 0;

    /**
     * Function xgemv is an abstraction of calls sgemv/dgemv/etc.
     * It performs the matrix times vector operation y := alpha * trans(A) * x + beta * y
     *
     * See also
     * http://www.netlib.org/lapack/explore-html/d7/d15/group__double__blas__level2_gadd421a107a488d524859b4a64c1901a9.html
     *
     * @param[in] trans defines the transformation applied to matrix A (see enum class OperationType)
     * @param[in] m the number of rows of matrix A
     * @param[in] n the number of columns of matrix A
     * @param[in] alpha scalar applied to A
     * @param[in] a matrix stored as a column-wise C-style array containing elements of type @tparam floating
     * @param[in] lda the first dimension of matrix A
     * @param[in] x a C-style array containing (1 + (@param DIM-1) * abs(@param incx) elements of type @tparam floating. DIM is @param n if trans is OperationType::Identical, and @param m else.
     * @param[in] incx storage spacing between elements of @param x
     * @param[in] beta scalar applied to @param y
     * @param[in,out] y a C-style array containing (1 + (@param DIM-1) * abs(@param incy) elements of type @tparam floating. DIM is @param n if trans is OperationType::Identical, and @param m else.
     * @param[in] incy storage spacing between elements of @param y
     */
    virtual void xgemv(const OperationType trans, const int m, const int n,
                       const floating alpha, floating const * const a, const int lda, floating const * const x, const int incx,
                       const floating beta, floating * const y, const int incy) const = 0;

    /**
     * Function xgetrf is an abstraction of calls sgetrf/dgetrf/etc.
     * It computes the PLU factorization of a general M-by-N matrix A. A is overwritten by L and U,
     * and L has unit diagonal elements.
     *
     * See also
     * http://www.netlib.org/lapack/explore-html/dd/d9a/group__double_g_ecomputational_ga0019443faea08275ca60a734d0593e60.html
     *
     * @param[in] m the number of rows of matrix A
     * @param[in] n the number of columns of matrix A
     * @param[in,out] a matrix stored as a column-wise C-style array containing elements of type @tparam floating
     * @param[in] lda the first dimension of matrix A
     * @param[out] ipiv permutation array, row i of matrix A is interchanged with row ipiv[i]
     * @param[out] info If == 0, successful exit. If < 0, the i-th argument had illegal value. If > 0, U(i, i) is exactly 0 and cannot be used to solve a system of equations.
     */
    virtual void xgetrf(int * const m, int * const n, floating * const a, int * const lda,
                        int * const ipiv, int * const info) const = 0;

    /**
     * Function xgetri is an abstraction of calls sgetri/dgetri/etc.
     * It computes the inverse of a square matrix using its PLU factorization as calculated by xgetrf.
     *
     * See also
     * http://www.netlib.org/lapack/explore-html/dd/d9a/group__double_g_ecomputational_ga56d9c860ce4ce42ded7f914fdb0683ff.html
     *
     * @param[in] n order of square matrix A
     * @param[in,out] a LU-factorization matrix (as calculated by xgetrf) stored as a column-wise C-style array containing elements of type @tparam floating
     * @param[in] lda the first dimension of matrix A
     * @param[in] ipiv permutation array from xgetrf
     * @param[out] work is temporary working space in form of a C-style array containing elements of type floating. On exit, if info == 0, work[1] returns optimal @param lwork
     * @param[in] lwork dimension of array @param work. If == -1, the routine calculates the optimal size of array @param work.
     * @param[out] info If == 0, successful exit. If < 0, the i-th argument had illegal value. If > 0, U(i, i) is exactly 0 and cannot be used to solve a system of equations.
     */
    virtual void xgetri(const int * const n, floating * const a, const int * const lda, const int * const ipiv,
                        floating * const work, const int * const lwork, int * const info) const = 0;

    /**
     * Function xgetrs is an abstraction of calls sgetrs/dgetrs/etc.
     * It solves a system of linear equations trans(A) * X = B using its PLU factorization as calculated by xgetrf.
     * Matrix A has to be square, and the result is stored in B.
     *
     * See also
     * http://www.netlib.org/lapack/explore-html/dd/d9a/group__double_g_ecomputational_ga58e332cb1b8ab770270843221a48296d.html
     *
     * @param[in] trans defines the transformation applied to matrix A (see enum class OperationType)
     * @param[in] n order of square matrix A
     * @param[in] nrhs number of right hand sides, i.e. the number of columns of B
     * @param[in] a LU-factorization matrix (as calculated by xgetrf) stored as a column-wise C-style array containing elements of type @tparam floating
     * @param[in] lda the first dimension of matrix A
     * @param[in] ipiv permutation array from xgetrf
     * @param[in,out] b matrix of right hand sides stored as a column-wise C-style array containing elements of type @tparam floating
     * @param[in] ldb the first dimension of matrix B
     * @param[out] info If == 0, successful exit. If < 0, the i-th argument had illegal value.
     */
    virtual void xgetrs(const OperationType trans, const int * const n, const int * const nrhs,
                        const floating * const a, const int * const lda, const int * const ipiv,
                        floating * const b, const int * const ldb, int * const info) const = 0;


    // TODO: make lower-case
    /**
     * Function xscal is an abstraction of calls sscal/dscal/etc.
     * It performs the scaling operation y := alpha * y.
     *
     * See also
     * http://www.netlib.org/lapack/explore-html/de/da4/group__double__blas__level1_ga793bdd0739bbd0e0ec8655a0df08981a.html
     *
     * @param N number of elements in input vectors
     * @param alpha scalar alpha
     * @param X a C-style array containing (1 + (@param n-1) * abs(@param incx) elements of type @tparam floating
     * @param incX storage spacing between elements of @param x
     */
    virtual void xscal(const int N, const floating alpha, floating * const X, const int incX) const = 0;

    /**
     * Returns true if @tparam floating is float.
     * @return true if @tparam floating is float
     */
    constexpr static bool isFloat();

    /**
     * Returns true if @tparam floating is double.
     * @return true if @tparam floating is double
     */
    constexpr static bool isDouble();
};


/**
 * Class CPU provides linear algebra operations on the CPU using
 * BLAS/LAPACK/LAPACKE. All memory allocations also take place in
 * ordinary RAM.
 *
 * @tparam floating floating point type, either float or double
 */
template<class floating>
class CPU : public ProcessingUnitDevice<floating> {
public:
    using ProcessingUnitDevice<floating>::isFloat;
    using ProcessingUnitDevice<floating>::isDouble;

    /**
     * Constructor
     */
    CPU() = default;

    /**
     * Destructor
     */
    ~CPU() override = default;

    Timer createTimer() const override;
    MemoryManager getMemoryManager() const override;
    std::string display() const override;

    int ixamax(const int n, const floating * const x, const int incx) const override;
    void xaxpy(const int n, const floating alpha, const floating * const x, const int incx, floating * const y, const int incy) const override;
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
    static std::map<OperationType, CBLAS_TRANSPOSE> toInternalOperationBLAS; //!< Converts the OperationType to the corresponding CBLAS_TRANSPOSE type
    static std::map<OperationType, const char * const> toInternalOperationLAPACK; //!< Converts the OperationType to the corresponding LAPACK transposition type
    static MemoryManager _deviceManager; //!< Memory manager associated with CPU memory
};

#ifndef CPU_ONLY

/**
 * Class GPU provides linear algebra operations on the GPU using
 * cuBLAS/cuSOLVER. All memory allocations also take place in
 * GPU RAM.
 *
 * @tparam floating floating point type, either float or double
 */
template<class floating>
class GPU : public ProcessingUnitDevice<floating> {
public:
    using ProcessingUnitDevice<floating>::isFloat;
    using ProcessingUnitDevice<floating>::isDouble;

    /**
     * Constructor
     */
    GPU() = default;

    /**
     * Destructor
     */
    ~GPU() override = default;

    Timer createTimer() const override;
    MemoryManager getMemoryManager() const override;
    std::string display() const override;

    /**
     * Returns the instance's cuBLAS handle.
     * @return cuBLAS handle
     */
    auto getCublasHandle() const;

    /**
     * Returns the instance's cuSOLVER handle.
     * @return cuSOLVER handle
     */
    auto getCusolverHandle() const;

    int ixamax(const int n, const floating * const x, const int incx) const override;
    void xaxpy(const int n, const floating alpha, const floating * const x, const int incx, floating * const y, const int incy) const override;
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
    static std::map<OperationType, cublasOperation_t> toInternalOperation; //!< Converts the OperationType to the corresponding cuBLAS operation type
    static GPU_Handle _handle; //!< GPU handle for cuBLAS/cuSOLVER operations
    static MemoryManager _deviceManager; //!< Memory manager associated with CUDA memory

};

#ifdef MAGMA
/**
 * Class GPU_MAGMA provides linear algebra operations on the GPU using
 * MAGMA. All memory allocations also take place in GPU RAM.
 *
 * @tparam floating floating point type, either float or double
 */
template<class floating>
class GPU_MAGMA : public ProcessingUnitDevice<floating> {
public:
    using ProcessingUnitDevice<floating>::isFloat;
    using ProcessingUnitDevice<floating>::isDouble;

    /**
     * Constructor
     */
    GPU_MAGMA() = default;

    /**
     * Destructor
     */
    ~GPU_MAGMA() override = default;

    Timer createTimer() const override;
    MemoryManager getMemoryManager() const override;
    std::string display() const override;

    /**
     * Returns the instance's magma queue.
     */
    auto getMagmaQueue() const;

    int ixamax(const int n, const floating * const x, const int incx) const override;
    void xaxpy(const int n, const floating alpha, const floating * const x, const int incx, floating * const y, const int incy) const override;
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
    static std::map<OperationType, magma_trans_t> toInternalOperation; //!< Converts the OperationType to the corresponding MAGMA transposition type
    static GPU_MAGMA_Queue _queue; //!< MAGMA queue for MAGMA operations
    static MemoryManager _deviceManager; //!< Memory manager associated with CUDA memory
};

/**
 * Class ReplacementNumber is used for passing the operation to GPU_MIXED, which is then calculated via
 * GPU_MAGMA instead of ordinary GPU processing unit.
 */
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

/**
 * Class GPU_MIXED provides linear algebra operations on the GPU using
 * cuBLAS/cuSOLVER, except for the one operation specified by replacementNumber,
 * which is calculated using MAGMA.
 * All memory allocations also take place in GPU RAM.
 *
 * @tparam floating floating point type, either float or double
 */
template<class floating>
class GPU_MIXED : public ProcessingUnitDevice<floating> {
public:
    /**
     * Constructor
     */
    GPU_MIXED(const ReplacementNumber replacementNumber = ReplacementNumber::XGETRF);

    /**
     * Destructor
     */
    ~GPU_MIXED() override = default;

    Timer createTimer() const override;

    MemoryManager getMemoryManager() const override;

    std::string display() const override;

    int ixamax(const int n, const floating * const x, const int incx) const override;
    void xaxpy(const int n, const floating alpha, const floating * const x, const int incx, floating * const y, const int incy) const override;
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
    static GPU<floating> _gpu; //!< GPU processing unit
    static GPU_MAGMA<floating> _gpu_magma; //!< MAGMA processing unit
    ReplacementNumber _replacementNumber; //!< current instance's operation which should be exectued on MAGMA instead of cuBLAS/cuSOLVER
};
#endif
#endif

#include "processingunit.hpp"
#include "processingunit_cpu.hpp"

#ifndef CPU_ONLY
    #include "processingunit_gpu.hpp"
    #ifdef MAGMA
        #include "processingunit_gpu_magma.hpp"
        #include "processingunit_gpu_mixed.hpp"
    #endif
#endif

#endif
