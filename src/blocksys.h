#ifndef FILE_BLOCKSYS
#define FILE_BLOCKSYS

#include "algebraiccontainers/algebraiccontainers.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <vector>

#include "auxiliary.h"
#include "algebraiccontainers/containerfactory.h"

/*+
 * Class SolvingProcedure is for defining with which solver
 * the linear system of equations for the fractional PDE is
 * going to be solved.
 *
 * Note that CyclicReduction is only possible for systems
 * with both equidistant space and time grids.
 */
enum class SolvingProcedure
{
    CyclicReduction,
    PCBiCGStab,
    BiCGStab,
    Richardson,
    PCRichardson
};

/**     Class EquidistantBlock_1D is used for storing the large system matrix occurring when
 *      discretizing fractional PDEs with both equidistant time and space grids using the so-called
 *      S3 formula. Also, it provides methods for efficiently calculating its solution.
 *
 *      The preferred way of solving these systems is cyclic reduction.
 *
 *      Matrix with block tridiagonal structure.
 *      Each block has the same dimension and it is a dense matrix.
 *
 *      [    M   -B   0   ...           0   ]
 *      [   -B    A  -B                 .   ]
 *      [    0   -B   A   -B                ]
 *      [    .        .    .   .            ]
 *      [    .             .   .   .        ]
 *      [                     -B   A   -B   ]
 *      [                          -B   M   ]
 *
 *      Matrix A is computed via A := 2/h^2 (M - h^2/2 g D) except for 1st, 2nd and last row, where it
 *      is equal to M.
 */
template<class floating>
class EquidistantBlock_1D
{
public:
    using SizeType = typename AlgebraicMatrix<floating>::SizeType;

    /** Constructor. @param B, @param M and @param D have to be square AlgebraicMatrix, and they must have the same dimensions.
     *
     * @param[in] bdim number of row respectively column blocks
     * @param[in] B matrix for off diagonal blocks
     * @param[in] D used for calculation of A, the diagonal blocks
     * @param[in] M matrix regarding Dirichlet conditions for all time steps
     * @param[in] h distance between the equidistant space points
     * @param[in] alpha anomalous diffusion coefficient
     * @param[in] timeGridStepSize distance between the equidistant space points
     * @param[in] processingUnit processingUnit with which the calculations should be performed
     */
    EquidistantBlock_1D(const SizeType bdim,
                        const AlgebraicMatrix<floating> &B, const AlgebraicMatrix<floating> &D, const AlgebraicMatrix<floating> &M, const floating h,
                        const floating alpha, const floating timeGridStepSize, const ProcessingUnit<floating> processingUnit);



    /** Constructor. @param B and @param M have to be square, and they must all have the same dimension as CoefficientMatrix @param D.
      *
      * @param[in] bdim number of row respectively column blocks
      * @param[in] B matrix for off diagonal blocks
      * @param[in] D used for calculation of A, the diagonal blocks
      * @param[in] M matrix regarding Dirichlet conditions for all time steps
      * @param[in] h distance between the equidistant space points
      * @param[in] alpha anomalous diffusion coefficient
      * @param[in] timeGridStepSize distance between the equidistant space points
      * @param[in] processingUnit processingUnit with which the calculations should be performed
      */
    EquidistantBlock_1D(const SizeType bdim,
                        const AlgebraicMatrix<floating> &B, const CoefficientMatrix<floating> &D, const AlgebraicMatrix<floating> &M, const floating h,
                        const floating alpha, const floating timeGridStepSize, const ProcessingUnit<floating> processingUnit);

    /**
     * Destructor
     */
    ~EquidistantBlock_1D() = default;

    /** Returns the order of a single (square) block.
     * @return Order of a single block.
     */
    SizeType getNdim() const;

    /** Returns block row dimension. (Equal to block column dimension, since the system is square.)
     * @return Number of blocks in a row.
     */
    SizeType getBlockDim() const;

    /** Order of the square system when interpreted as a large dense matrix.
     * @return Dense system dimension.
     */
    SizeType getDenseDim() const;

    // TODO: Rename to ContainerFactory
    /**
     * Returns the container factory which can be used to create new containers
     * on the same device with the same processing unit.
     * @return container factory
     */
    ContainerFactory<floating> getMatrixFactory() const;

    // TODO: Find a better name
    /**
     * Returns the system as a dense matrix.
     * @return Dense matrix representation stored columnwise.
     */
    AlgebraicMatrix<floating> copyToDense() const;

    // TODO: do not return beta, but u instead!
    /**
     * Solves the system using CyclicReduction, which is the recommended
     * solver for systems with equidistant time and space grids.
     *
     * Note that BiCGStab and PCBiCGStab are not available.
     *
     * @param[in] rhs Right-hand side of the system K * beta = rhs
     * @return the solution beta
     */
    BlockVector<floating> solve(const BlockVector<floating> &rhs) const;

    /**
     * Solves the system K * beta = rhs for beta using a SolvingProcedure with the given parameters.
     * @param[in] rhs right-hand side
     * @param[in] maxNumberOfIterations maximal number of iterations (not used for CyclicReduction)
     * @param[in] stepsPerIteration number of steps in each iteration (not used for CyclicReduction)
     * @param[in] accuracy desired absolute accuracy
     * @param[in] solvingProcedure the solver used to solve the system
     * @return the solution beta
     */
    BlockVector<floating> solve(const BlockVector<floating> &rhs, const size_t maxNumberOfIterations, const size_t stepsPerIteration, const floating accuracy, const SolvingProcedure solvingProcedure) const;


private:
    SizeType _bdim; //!< number of block rows and block columns
    AlgebraicMatrix<floating> _B; //!< matrix contained in the system's subdiagonal blocks
    AlgebraicMatrix<floating> _D; //!< matrix used for calculation of matrix A contained in the diagonal blocks
    AlgebraicMatrix<floating> _M; //!< Dirichlet boundary condition matrix
    floating _alpha; //!< anomalous diffusion coefficient
    floating _timeGridStepSize; //!< step size between points of the equidistant time grid
    floating _h; //!< step size between points of the equidistant space grid
    AlgebraicMatrix<floating> _A; //!< matrix contained in the system's diagonal blocks
    ContainerFactory<floating> _colMatrixFactory; //!< factory for creating algebraic containers using suitable processing unit (e.g. CPU or GPU)
    std::unique_ptr<EquidistantBlock_1D<floating>> _coarseSystemPtr; //!< for optimization purposes, the coarse systems for Cyclic Reduction or Multigrid are allocated at construction

    /**
     * Constructor exclusively used for the cyclic reduction routine, where the matrix A is substituted by
     * a modified version in each step.
     *
     * @param[in] bdim number of row respectively column blocks
     * @param[in] B matrix for off diagonal blocks
     * @param[in] D used for calculation of A, the diagonal blocks
     * @param[in] M matrix regarding Dirichlet conditions for all time steps
     * @param[in] h distance between the equidistant space points
     * @param[in] alpha anomalous diffusion coefficient
     * @param[in] timeGridStepSize distance between the equidistant space points
     * @param[in] A matrix contained in the system's diagonal blocks
     * @param[in] processingUnit processingUnit with which the calculations should be performed
     */
    EquidistantBlock_1D(const SizeType bdim,
                        const AlgebraicMatrix<floating> &B, const AlgebraicMatrix<floating> &D, const AlgebraicMatrix<floating> &M,
                        const floating h, const floating alpha, const floating timeGridStepSize,
                        const AlgebraicMatrix<floating> &A, const ProcessingUnit<floating> processingUnit);

    /**
     * Calculates the coefficient needed in the calculation of matrix A.
     *
     * @param alpha anomalous diffusion coefficient
     * @param timeGridStepSize distance between the equidistant space points
     * @return coefficient needed for calculation of matrix A
     */
    static floating getSystemCoeff(const floating alpha, const floating timeGridStepSize);

    /**
     * Constructor exclusively used for the cyclic reduction routine, where the matrix A is substituted by
     * a modified version in each step.
     *
     * @param[in] bdim number of row respectively column blocks
     * @param[in] B matrix for off diagonal blocks
     * @param[in] D used for calculation of A, the diagonal blocks
     * @param[in] M matrix regarding Dirichlet conditions for all time steps
     * @param[in] h distance between the equidistant space points
     * @param[in] alpha anomalous diffusion coefficient
     * @param[in] timeGridStepSize distance between the equidistant space points
     * @param[in] A matrix contained in the system's diagonal blocks
     * @param[in] processingUnit processingUnit with which the calculations should be performed
     */

    /**
     * Calculate matrix A which occurs in the system's diagonals.
     *
     * @param B matrix for off diagonal blocks
     * @param D used for calculation of A, the diagonal blocks
     * @param M matrix regarding Dirichlet conditions for all time steps
     * @param h distance between the equidistant space points
     * @param alpha anomalous diffusion coefficient
     * @param timeGridStepSize distance between the equidistant space points
     * @return matrix A which is located in the system's diagonals
     */
    AlgebraicMatrix<floating> initializeA(const AlgebraicMatrix<floating> &B, const AlgebraicMatrix<floating> &D,
                                          const AlgebraicMatrix<floating> &M, const floating h, const floating alpha,
                                          const floating timeGridStepSize) const;

    /**
     * Returns the current instance's underlying processing unit.
     * @return current processing unit
     */
    ProcessingUnit<floating> getProcessingUnit() const;

    /**
     * Solves the current system using cyclic reduction with RHS @param f.
     * @param scaledB scaled matrix B
     * @param f right-hand side BlockVector
     * @return the system's solution calculated with cyclic reduction
     */
    BlockVector<floating> cyclicReduction(const AlgebraicMatrix<floating> &scaledB, const BlockVector<floating> &f) const;

    /** (For use with Cyclic Reduction only.) Prolongates the coarse solution BlockVector @param uc to the fine vector.
     *
     * The prolongation uses the 1D formula
     * @f$ u_f[2*i-1] =  A^{-1}\cdot \left({ f_f[2i-1] + B\cdot (u_c[2i-2]+u_c[2i])  }\right)\enspace,@f$
     * and the simple copy  @f$ u_f[2*i] = u_c[2*i]@f$ for fine nodes with coarse indices.
     *
     * @param[in] scaledB scaled matrix B
     * @param[in] ff rhs BlockVector on fine grid
     * @param[in] uc solution BlockVector on coarse grid
     *
     * @return fine vector (matrix) uf.
     */
    AlgebraicMatrix<floating> CRProlongation(const AlgebraicMatrix<floating> &scaledB, const AlgebraicMatrix<floating> &ff, const AlgebraicMatrix<floating> &uc) const;

    /** (For use with Cyclic Reduction only.) Restricts fine rhs vector (matrix) @p ff to the coarse vector.
     *
     * The restriction uses the 1D formula
     * @f$ f_c[i] = f_f[2i]+B\cdot A^{-1}\cdot (f_f[2i-1]+f_f[2i+1])@f$ and boundary condition.
     *
     * @param[in] scaledB scaled matrix B
     * @param[in] ff vector (matrix) on fine grid
     *
     * @return coarse rhs BlockVector fc.
     */
    AlgebraicMatrix<floating> CRRestriction(const AlgebraicMatrix<floating> &scaledB, const AlgebraicMatrix<floating> &ff) const;

    /**
     * Applies one multigrid iteration to the system.
     *
     * @param[in] numberOfSmoothingSteps number of pre- and post-smoothing steps
     * @param[in] f right-hand side
     * @param[in] maxNumberOfIterations maximal number of multigrid iterations
     * @param[in] accuracy absolute accuracy used for termination of the smoothing steps
     * @param[in/out] solution solution to the system which is being updated
     * @return euclidean error of the residual K * beta - f
     */
    floating multigrid(const unsigned numberOfSmoothingSteps, const BlockVector<floating> &f, const size_t maxNumberOfIterations, const floating accuracy, BlockVector<floating> &solution) const;

    /**
     * Applies the (linear interpolation) prolongation to the coarse-grid right-hand side @param ff onto the fine grid.
     * @param ff coarse-grid rhs
     * @return @param ff prolongated onto the fine grid
     */
    BlockVector<floating> prolongation(const BlockVector<floating> &ff) const;

    /**
     * Applies the full restriction to the fine-grid right-hand side @param ff onto the coarse grid.
     * @param ff fine-grid rhs
     * @return @param ff restricted onto the coarse grid
     */
    BlockVector<floating> restriction(const BlockVector<floating> &ff) const;

    /**
     * Returns the distance between space points in the coarse grid.
     * @return distance between coarse space grid points
     */
    floating getReducedGrid() const;

    /**
     * Returns the number of blocks in each row (resp. column) of the reduced, coarse system.
     * @return number of blocks in each row of the coarse system
     */
    SizeType getCoarseDim() const;

    /**
     * Initializes the coarse system and returns the pointer.
     * @return pointer to newly constructed coarse system
     */
    std::unique_ptr<EquidistantBlock_1D<floating>> initializeCoarseSystemPtr() const;

    /**
     * Returns the current instance's coarse system.
     * @return coarse system
     */
    const EquidistantBlock_1D<floating>& getCoarseSystem() const;

    /**
     * Performs steps of weighted block Jacobi iteration, until either the maximal number of iterations or the desired absolute
     * accuracy is reached.
     *
     * @param[in] omega relaxation parameter
     * @param[in] f right-hand side
     * @param[in] maxNumberOfIterations maximal number of iterations performed
     * @param[in] accuracy desired absolute accuracy
     * @param[in,out] solution initial solution which is being updated
     * @return Euclidean norm of residual K*beta - f, where beta is corresponding to @param solution
     */
    floating jacobiIteration(const floating omega, const BlockVector<floating> &f, const size_t maxNumberOfIterations, const floating accuracy, BlockVector<floating> &solution) const;

    /**
     * Performs smoothing steps (similar to jacobiIteration, but without calculation of accuracy due to performance)
     * to the system K*beta = f, where beta is corresponding to @param solution.
     *
     * @param[in] omega relaxation parameter
     * @param[in] f right-hand side
     * @param[in] maxNumberOfIterations maximal number of iterations performed
     * @param[in,out] solution initial solution which is being updated
     */
    void smooth(const floating omega, const BlockVector<floating> &f, const size_t maxNumberOfIterations, BlockVector<floating> &solution) const;

    // TODO: Rename u to beta
    /**
     * Calculate total residual K*beta - f.
     *
     * @param[in] u solution
     * @param[in] f right-hand side
     * @return residual K*beta - f
     */
    BlockVector<floating> calculateResidual(const BlockVector<floating> &u, const BlockVector<floating> &f) const;

    // TODO: Rename u to beta
    /**
     * Calculate block row @param i of residual K*beta - f
     * @param[in] i block row index
     * @param[in] u solution
     * @param[in] f right-hand side
     * @return @param i-th block row of residual K*beta - f
     */
    AlgebraicVector<floating> calculateRowResidual(const SizeType i, const BlockVector<floating> &u, const BlockVector<floating> &f) const;

    /**
     * Rescales the right-hand side to fit the symmetrization of the system.
     * @param rhs right-hand side
     * @return rescaled right-hand side
     */
    BlockVector<floating> rescale_rhs(const BlockVector<floating> &rhs) const;
};

/**     Class NonEquidistantBlock_1D is used for storing the large system matrix occurring when
 *      discretizing fractional PDEs with equidistant time grid, but arbitrary space grid using the so-called
 *      S3 formula. Also, it provides methods for efficiently calculating its solution.
 *
 *      Most memory needed is reserved when being initialized to reduce the costs of memory
 *      allocation / freeing.
 *
 *      The preferred way of solving these systems is PCBiCGStab.
 *
 *      Matrix with block tridiagonal structure.
 *      Each block has the same dimension and it is a dense matrix.
 *
 *      [    M   -B     0   ...              0   ]
 *      [   -B    A_1  -B                    .   ]
 *      [    0   -B     A_2   -B                 ]
 *      [    .            .    .   .             ]
 *      [    .              .    .   .           ]
 *      [                     -B   A_(M-1)   -B  ]
 *      [                            -B       M  ]
 *
 *      Matrices A_i are computed via A_i := 2/(h_i h_(i+1)) (M - (h_i h_(i+1))/2 g D) except for 1st, 2nd and last row,
 *      where it is equal to M.
 */
template<class floating>
class NonEquidistantBlock_1D
{
public:
    using SizeType = typename AlgebraicMatrix<floating>::SizeType;

    /** Constructor. @param B, @param M and @param D have to be square AlgebraicMatrix, and they must have the same dimensions.
     *
     * @param[in] bdim number of row respectively column blocks
     * @param[in] B matrix for off diagonal blocks
     * @param[in] D used for calculation of A, the diagonal blocks
     * @param[in] M matrix regarding Dirichlet conditions for all time steps
     * @param[in] h AlgebraicVector containing distances between subsequent space grid points
     * @param[in] alpha anomalous diffusion coefficient
     * @param[in] timeGridStepSize distance between the equidistant space points
     * @param[in] processingUnit processingUnit with which the calculations should be performed
     */
    NonEquidistantBlock_1D(const SizeType bdim,
                           const AlgebraicMatrix<floating> &B, const CoefficientMatrix<floating> &D,
                           const AlgebraicMatrix<floating> &M,
                           const AlgebraicVector<floating> &h, const floating alpha, const floating timeGridStepSize,
                           const ProcessingUnit<floating> processingUnit);

    /** Constructor. @param B and @param M have to be square, and they must all have the same dimension as CoefficientMatrix @param D.
      *
      * @param[in] bdim number of row respectively column blocks
      * @param[in] B matrix for off diagonal blocks
      * @param[in] D used for calculation of A, the diagonal blocks
      * @param[in] M matrix regarding Dirichlet conditions for all time steps
      * @param[in] h AlgebraicVector containing distances between subsequent space grid points
      * @param[in] alpha anomalous diffusion coefficient
      * @param[in] timeGridStepSize distance between the equidistant space points
      * @param[in] processingUnit processingUnit with which the calculations should be performed
      */
    NonEquidistantBlock_1D(const SizeType bdim,
                           const AlgebraicMatrix<floating> &B, const AlgebraicMatrix<floating> &D, const AlgebraicMatrix<floating> &M,
                           const AlgebraicVector<floating> &h, const floating alpha, const floating timeGridStepSize,
                           const ProcessingUnit<floating> processingUnit);

    /**
     * Destructor.
     */
    ~NonEquidistantBlock_1D() = default;

    /** Returns the order of a single (square) block.
     * @return Order of a single block.
     */
    SizeType getNdim() const;

    /** Returns block row dimension. (Equal to block column dimension, since the system is square.)
     * @return Number of blocks in a row.
     */
    SizeType getBlockDim() const;

    /** Order of the square system when interpreted as a large dense matrix.
     * @return Dense system dimension.
     */
    SizeType getDenseDim() const;

    // TODO: Rename to ContainerFactory
    /**
     * Returns the container factory which can be used to create new containers
     * on the same device with the same processing unit.
     * @return container factory
     */
    ContainerFactory<floating> getMatrixFactory() const;

    // TODO: Find a better name
    /**
     * Returns the system as a dense matrix.
     * @return Dense matrix representation stored columnwise.
     */
    AlgebraicMatrix<floating> copyToDense() const;

    // TODO: rename u to beta
    /**
     * Performs the multiplication K*beta and returns it in @param result.
     * Note that @param result has to have as many elements as @param u.
     * @param[in] u solution
     * @param[out] result result of multiplication
     */
    void mult(const BlockVector<floating> &u, BlockVector<floating> &result) const;

    // TODO: do not return beta, but u instead!
    /**
     * Solves the system K * beta = rhs for beta using a SolvingProcedure with the given parameters.
     *
     * Note that CyclicReduction is not available.
     *
     * @param[in] rhs right-hand side
     * @param[in] maxNumberOfIterations maximal number of iterations
     * @param[in] stepsPerIteration number of steps in each iteration
     * @param[in] accuracy desired absolute accuracy
     * @param[in] solvingProcedure the solver used to solve the system
     * @return the solution beta
     */
    BlockVector<floating> solve(BlockVector<floating> &rhs, const size_t maxNumberOfIterations, const size_t stepsPerIteration, const floating accuracy, const SolvingProcedure solvingProcedure = SolvingProcedure::PCBiCGStab) const;

private:
    SizeType _bdim; //!< number of block rows and block columns
    AlgebraicMatrix<floating> _B; //!< matrix contained in the system's subdiagonal blocks
    AlgebraicMatrix<floating> _D; //!< matrix used for calculation of matrix A contained in the diagonal blocks
    AlgebraicMatrix<floating> _M; //!< Dirichlet boundary condition matrix
    floating _alpha; //!< anomalous diffusion coefficient
    floating _timeGridStepSize; //!< step size between points of the equidistant time grid
    AlgebraicVector<floating> _h; //!< vector of differences between points of the equidistant space grid.
    AlgebraicMatrix<floating> _C; //!< matrix used for approximation of the matrices A_i occurring in the system's diagonals
    ContainerFactory<floating> _colMatrixFactory; //!< factory for creating algebraic containers using suitable processing unit (e.g. CPU or GPU)
    std::unique_ptr<NonEquidistantBlock_1D<floating>> _coarseSystemPtr; //!< for optimization purposes, the coarse system for Multigrid is allocated at construction
    mutable AlgebraicMatrix<floating> _vectorBuffer; //!< buffer where the individual columns can be used independently internally
    mutable AlgebraicVector<floating> _host_h; //!< the vector of space grid increments is stored on the CPU separately, because it is needed there very often
    mutable AlgebraicMatrix<floating> _buffer1; //!< buffer holding an AlgebraicMatrix/BlockVector for storage of residuals etc.
    mutable AlgebraicMatrix<floating> _buffer2; //!< buffer holding an AlgebraicMatrix/BlockVector for storage of residuals etc.
    mutable AlgebraicMatrix<floating> _buffer3; //!< buffer holding an AlgebraicMatrix/BlockVector for storage of residuals etc.
    mutable AlgebraicMatrix<floating> _buffer4; //!< buffer holding an AlgebraicMatrix/BlockVector for storage of residuals etc.
    mutable AlgebraicMatrix<floating> _buffer5; //!< buffer holding an AlgebraicMatrix/BlockVector for storage of residuals etc.
    mutable AlgebraicMatrix<floating> _buffer6; //!< buffer holding an AlgebraicMatrix/BlockVector for storage of residuals etc.
    mutable AlgebraicMatrix<floating> _coarseBuffer1; //!< buffer holding an AlgebraicMatrix/BlockVector for storage of coarse residuals etc.
    mutable AlgebraicMatrix<floating> _coarseBuffer2; //!< buffer holding an AlgebraicMatrix/BlockVector for storage of coarse residuals etc.

    /**
     * Applies steps of BiCGStab to the system K*beta = f, where beta is corresponding to @param solution,
     * until either the maximal number of iterations or the desired absolute accuracy is reached.
     *
     * @param[in] f right-hand side
     * @param[in] maxNumberOfIterations maximal number of iterations performed
     * @param[in] accuracy desired absolute accuracy
     * @param[in,out] solution solution initial solution which is being updated
     * @return Euclidean norm of residual K*beta - f, where beta is corresponding to @param solution
     */
    floating biCGStab(const BlockVector<floating> &f, const size_t maxNumberOfIterations, const floating accuracy, BlockVector<floating> &solution) const;

    /**
     * Applies steps of multigrid-preconditioned BiCGStab to the system K*beta = f, where beta is corresponding
     * to @param solution, until either the maximal number of iterations or the desired absolute accuracy is reached.
     *
     * @param[in] f right-hand side
     * @param[in] maxNumberOfIterations maximal number of iterations performed
     * @param[in] accuracy desired absolute accuracy
     * @param[in,out] solution solution initial solution which is being updated
     * @return Euclidean norm of residual K*beta - f, where beta is corresponding to @param solution
     */
    floating PCbiCGStab(const BlockVector<floating> &f, const size_t maxNumberOfIterations, const floating accuracy, BlockVector<floating> &solution) const;

    /**
     * Applies one multigrid iteration to the system.
     *
     * @param[in] numberOfSmoothingSteps number of pre- and post-smoothing steps
     * @param[in] f right-hand side
     * @param[in] maxNumberOfIterations maximal number of multigrid iterations
     * @param[in] accuracy absolute accuracy used for termination of the smoothing steps
     * @param[in/out] solution solution to the system which is being updated
     */
    void multigrid(const unsigned numberOfSmoothingSteps, const BlockVector<floating> &f, const size_t maxNumberOfIterations, const floating accuracy, BlockVector<floating> &solution) const;

    /**
     * Applies the (linear interpolation) prolongation to the coarse-grid right-hand side @param ff onto the fine grid.
     * @param ff coarse-grid rhs
     * @return @param ff prolongated onto the fine grid
     */
    void prolongation(const BlockVector<floating> &ff, BlockVector<floating> &solution) const;

    /**
     * Applies the full restriction to the fine-grid right-hand side @param ff onto the coarse grid.
     * @param ff fine-grid rhs
     * @return @param ff restricted onto the coarse grid
     */
    void restriction(const BlockVector<floating> &ff, BlockVector<floating> &ffOnCoarseGrid) const;

    /**
     * Returns the distance between space points in the coarse grid.
     * @return distance between coarse space grid points
     */
    AlgebraicVector<floating> getReducedGrid() const;

    /**
     * Returns the number of blocks in each row (resp. column) of the reduced, coarse system.
     * @return number of blocks in each row of the coarse system
     */
    SizeType getCoarseDim() const;

    /**
     * Initializes the coarse system and returns the pointer.
     * @return pointer to newly constructed coarse system
     */
    std::unique_ptr<NonEquidistantBlock_1D<floating>> initializeCoarseSystemPtr() const;

    /**
     * Returns the current instance's coarse system.
     * @return coarse system
     */
    const NonEquidistantBlock_1D<floating>& getCoarseSystem() const;

    /**
     * Performs smoothing steps (weighted block Jacobi iteration)
     * to the system K*beta = f, where beta is corresponding to @param solution,
     * until the maximal number of iterations is reached
     *
     * @param[in] omega relaxation parameter
     * @param[in] f right-hand side
     * @param[in] maxNumberOfIterations maximal number of iterations performed
     * @param[in,out] solution initial solution which is being updated
     */
    void smooth(const floating omega, const BlockVector<floating> &f, const size_t maxNumberOfIterations, BlockVector<floating> &solution) const;

    // TODO: Rename u to beta
    /**
     * Calculate total residual K*beta - f.
     *
     * @param[in] u solution
     * @param[in] f right-hand side
     * @return residual K*beta - f
     */
    void calculateResidual(const BlockVector<floating> &u, const BlockVector<floating> &f, BlockVector<floating> &residual) const;

    // TODO: Rename u to beta
    /**
     * Calculate block row @param i of residual K*beta - f
     * @param[in] i block row index
     * @param[in] u solution
     * @param[in] f right-hand side
     * @return @param i-th block row of residual K*beta - f
     */
    void calculateRowResidual(const SizeType i, const BlockVector<floating> &u, const BlockVector<floating> &f, AlgebraicVector<floating> &residual) const;

    /**
    * Calculates the coefficient needed in the calculation of matrix A.
    *
    * @param alpha anomalous diffusion coefficient
    * @param timeGridStepSize distance between the equidistant space points
    * @return coefficient needed for calculation of matrix A
    */
    floating getSystemCoeff() const;

    /**
     * Calculate matrix C, approximating the matrices A_i which occur in the system's diagonals.
     * @return matrix C
     */
    AlgebraicMatrix<floating> initializeC() const;

    /**
     * Returns the current instance's underlying processing unit.
     * @return current processing unit
     */
    ProcessingUnit<floating> getProcessingUnit() const;

    /**
     * For the approximation of matrices A_i which occur in the system's diagonals,
     * matrix C is computed, needing a preconditioner depending on all space grid increments h_i.
     * @param h vector of space grid increments
     * @return maximal entry max_i (h_i)
     */
    floating get_h_preconditioner(const AlgebraicVector<floating> &h) const;

    NonEquidistantBlock_1D() = default;

    /**
     * Rescales the right-hand side to fit the symmetrization of the system.
     * @param rhs right-hand side
     * @return rescaled right-hand side
     */
    void rescale_rhs(BlockVector<floating> &rhs) const;
};

#include "blocksys_equidistant.hpp"
#include "blocksys_nonequidistant.hpp"
#endif
