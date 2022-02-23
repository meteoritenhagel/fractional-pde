#ifndef FILE_BLOCKSYS
#define FILE_BLOCKSYS

#include "algebraiccontainers/algebraiccontainers.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <vector>

#include "fractional_pde.h"
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
    CyclicReduction, //!< Cyclic Reduction (only possible for space and time grids both equidistant)
    PCBiCGStab, //!< multigrid preconditioned BiCGStab
    BiCGStab, //!< BiCGStab
    Jacobi, //!< Jacobi iterations (or Gauss-Seidel, if preprocessor definition GAUSS_SEIDEL is active)
    PCJacobi //!< Multigrid preconditioned Jacobi iterations (or Gauss-Seidel,if preprocessor definition GAUSS_SEIDEL is active)
};

/**     Class EquidistantBlock1D is used for storing the large system matrix occurring when
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
class EquidistantBlock1D
{
public:
    using SizeType = typename AlgebraicMatrix<floating>::SizeType;

    /** Constructor. @p B, @p M and @p D have to be square AlgebraicMatrix, and they must have the same dimensions.
     *
     * @param[in] block_dim number of row respectively column blocks
     * @param[in] B matrix for off diagonal blocks
     * @param[in] D used for calculation of A, the diagonal blocks
     * @param[in] M matrix containing Dirichlet conditions
     * @param[in] space_grid_step_size distance between the equidistant space points
     * @param[in] alpha anomalous diffusion coefficient
     * @param[in] time_grid_step_size distance between the equidistant space points
     * @param[in] processing_unit processing_unit with which the calculations should be performed
     */
    EquidistantBlock1D(const SizeType block_dim,
                       const AlgebraicMatrix<floating> &B, const AlgebraicMatrix<floating> &D,
                       const AlgebraicMatrix<floating> &M, const floating space_grid_step_size,
                       const floating alpha, const floating time_grid_step_size,
                       const ProcessingUnit<floating> processing_unit);

    /** Constructor. @p B and @p M have to be square, and they must all have the same dimension as CoefficientMatrix @p D.
      *
      * @param[in] block_dim number of row respectively column blocks
      * @param[in] B matrix for off diagonal blocks
      * @param[in] D used for calculation of A, the diagonal blocks
      * @param[in] M matrix containing Dirichlet conditions
      * @param[in] space_grid_step_size distance between the equidistant space points
      * @param[in] alpha anomalous diffusion coefficient
      * @param[in] time_grid_step_size distance between the equidistant space points
      * @param[in] processing_unit processing_unit with which the calculations should be performed
      */
    EquidistantBlock1D(const SizeType block_dim,
                       const AlgebraicMatrix<floating> &B, const CoefficientMatrix<floating> &D,
                       const AlgebraicMatrix<floating> &M, const floating space_grid_step_size,
                       const floating alpha, const floating time_grid_step_size,
                       const ProcessingUnit<floating> processing_unit);

    /**
     * Destructor
     */
    ~EquidistantBlock1D() = default;

    /** Returns the order of a single (square) block.
     * @return Order of a single block.
     */
    SizeType get_num_blocks() const;

    /** Returns block row dimension. (Equal to block column dimension, since the system is square.)
     * @return Number of blocks in a row.
     */
    SizeType get_block_dim() const;

    /** Order of the square system when interpreted as a large dense matrix.
     * @return Dense system dimension.
     */
    SizeType get_dense_dim() const;

    /**
     * Returns the container factory which can be used to create new containers
     * on the same device with the same processing unit.
     * @return container factory
     */
    ContainerFactory<floating> get_container_factory() const;

    /**
     * Returns the system as a dense matrix.
     * @return Dense matrix representation stored column-wise.
     */
    AlgebraicMatrix<floating> get_dense_representation() const;

    /**
     * Solves the auxiliary linear system using CyclicReduction, which is the recommended
     * solver for systems with equidistant time and space grids, and then
     * calculates the PDE solution BlockVector out of it.
     *
     * The solution is a matrix consisting of dimension (N+1)x(M+1),
     * where N is the number of time intervals in the time grid,
     * and M is the number of space intervals in the space grid.
     * The entry solution(i, j) is the numerical PDE solution u(x_j, t_i)
     * in the space point x_j and the time point t_i,
     * where i is in [[0, N]] and j is in [[0, M]].
     *
     * The right-hand side @p rhs is obtained by executing the function initialize_rhs in fractional_pde.h.
     *
     * @param[in] rhs Right-hand side of the PDE system.
     * @return the numerical solution to the PDE as BlockVector
     */
    BlockVector<floating> solve_pde(const BlockVector<floating> &rhs) const;

    /**
     * Solves the auxiliary linear system using a SolvingProcedure with the given parameters,
     * and then calculates the PDE solution BlockVector out of it.
     *
     * Note that BiCGStab and PCBiCGStab are not available.
     * The solution is a matrix consisting of dimension (N+1)x(M+1),
     * where N is the number of time intervals in the time grid,
     * and M is the number of space intervals in the space grid.
     * The entry solution(i, j) is the numerical PDE solution u(x_j, t_i)
     * in the space point x_j and the time point t_i,
     * where i is in [[0, N]] and j is in [[0, M]].
     *
     * The right-hand side @p rhs is obtained by executing the function initialize_rhs in fractional_pde.h.
     *
     * Note that BiCGStab and PCBiCGStab are not available.
     *
     * @param[in] rhs right-hand side
     * @param[in] max_num_iterations maximal number of iterations (not used for CyclicReduction)
     * @param[in] steps_per_iteration number of steps in each iteration (not used for CyclicReduction)
     * @param[in] accuracy desired absolute accuracy
     * @param[in] solving_procedure the solver used to solve the system
     * @return the numerical solution to the PDE as BlockVector
     */
    BlockVector<floating> solve_pde(const BlockVector<floating> &rhs, const size_t max_num_iterations,
                                    const size_t steps_per_iteration, const floating accuracy,
                                    const SolvingProcedure solving_procedure) const;

    /**
     * Solves the auxiliary linear system using CyclicReduction, which is the recommended
     * solver for systems with equidistant time and space grids.
     *
     * @param[in] rhs Right-hand side of the system K * beta = rhs
     * @return the solution beta
     */
    BlockVector<floating> solve(const BlockVector<floating> &rhs) const;

    /**
     * Solves the auxiliary linear system K * beta = rhs for beta using a SolvingProcedure with the given parameters.
     * Note that BiCGStab and PCBiCGStab are not available.
     *
     * @param[in] rhs right-hand side
     * @param[in] max_num_iterations maximal number of iterations (not used for CyclicReduction)
     * @param[in] steps_per_iteration number of steps in each iteration (not used for CyclicReduction)
     * @param[in] accuracy desired absolute accuracy
     * @param[in] solving_procedure the solver used to solve the system
     * @return the solution beta
     */
    BlockVector<floating> solve(const BlockVector<floating> &rhs, const size_t max_num_iterations,
                                const size_t steps_per_iteration, const floating accuracy,
                                const SolvingProcedure solving_procedure) const;


private:
    SizeType _block_dim; //!< number of block rows and block columns
    AlgebraicMatrix<floating> _B; //!< matrix contained in the system's subdiagonal blocks
    AlgebraicMatrix<floating> _D; //!< matrix used for calculation of matrix A contained in the diagonal blocks
    AlgebraicMatrix<floating> _M; //!< Dirichlet boundary condition matrix
    floating _alpha; //!< anomalous diffusion coefficient
    floating _time_grid_step_size; //!< step size between points of the equidistant time grid
    floating _space_grid_step_size; //!< step size between points of the equidistant space grid
    AlgebraicMatrix<floating> _A; //!< matrix contained in the system's diagonal blocks
    ContainerFactory<floating> _container_factory; //!< factory for creating algebraic containers using suitable processing unit (e.g. Cpu or Gpu)
    std::unique_ptr<EquidistantBlock1D<floating>> _coarse_system; //!< for optimization purposes, the coarse systems for Cyclic Reduction or Multigrid are allocated at construction

    /**
     * Constructor exclusively used for the cyclic reduction routine, where the matrix A is substituted by
     * a modified version in each step.
     *
     * @param[in] block_dim number of row respectively column blocks
     * @param[in] B matrix for off diagonal blocks
     * @param[in] D used for calculation of A, the diagonal blocks
     * @param[in] M matrix containing Dirichlet conditions
     * @param[in] space_grid_step_size distance between the equidistant space points
     * @param[in] alpha anomalous diffusion coefficient
     * @param[in] time_grid_step_size distance between the equidistant space points
     * @param[in] A matrix contained in the system's diagonal blocks
     * @param[in] processing_unit processing_unit with which the calculations should be performed
     */
    EquidistantBlock1D(const SizeType block_dim,
                       const AlgebraicMatrix<floating> &B, const AlgebraicMatrix<floating> &D,
                       const AlgebraicMatrix<floating> &M, const floating space_grid_step_size, const floating alpha,
                       const floating time_grid_step_size, const AlgebraicMatrix<floating> &A,
                       const ProcessingUnit<floating> processing_unit);

    /**
     * Calculates the coefficient needed in the calculation of matrix A.
     *
     * @param alpha anomalous diffusion coefficient
     * @param time_grid_step_size distance between the equidistant space points
     * @return coefficient needed for calculation of matrix A
     */
    static floating get_system_coeff(const floating alpha, const floating time_grid_step_size);

    /**
     * Calculate matrix A which occurs in the system's diagonals.
     *
     * @param B matrix for off diagonal blocks
     * @param D used for calculation of A, the diagonal blocks
     * @param M matrix containing Dirichlet conditions
     * @param space_grid_step_size distance between the equidistant space points
     * @param alpha anomalous diffusion coefficient
     * @param time_grid_step_size distance between the equidistant space points
     * @return matrix A which is located in the system's diagonals
     */
    AlgebraicMatrix<floating> initialize_a(const AlgebraicMatrix<floating> &B, const AlgebraicMatrix<floating> &D,
                                           const AlgebraicMatrix<floating> &M, const floating space_grid_step_size,
                                           const floating alpha, const floating time_grid_step_size) const;

    /**
     * Returns the current instance's underlying processing unit.
     * @return current processing unit
     */
    ProcessingUnit<floating> get_processing_unit() const;

    /**
     * Solves the current system using cyclic reduction with RHS @p f.
     * @param scaled_b scaled matrix B
     * @param f right-hand side BlockVector
     * @return the system's solution calculated with cyclic reduction
     */
    BlockVector<floating> cyclic_reduction(const AlgebraicMatrix<floating> &scaled_b,
                                           const BlockVector<floating> &f) const;

    /** (For use with Cyclic Reduction only.) Prolongates the coarse solution BlockVector @p coarse_solution to the fine vector.
     *
     * The prolongation uses the 1D formula
     * @f$ u_f[2*i-1] =  A^{-1}\cdot \left({ fine_rhs[2i-1] + B\cdot (coarse_solution[2i-2]+coarse_solution[2i])  }\right)\enspace,@f$
     * and the simple copy  @f$ u_f[2*i] = coarse_solution[2*i]@f$ for fine nodes with coarse indices.
     *
     * @param[in] scaled_b scaled matrix B
     * @param[in] fine_rhs rhs BlockVector on fine grid
     * @param[in] coarse_solution solution BlockVector on coarse grid
     *
     * @return fine BlockVector u_f.
     */
    AlgebraicMatrix<floating> cr_prolongation(const AlgebraicMatrix<floating> &scaled_b,
                                              const AlgebraicMatrix<floating> &fine_rhs,
                                              const AlgebraicMatrix<floating> &coarse_solution) const;

    /** (For use with Cyclic Reduction only.) Restricts fine rhs vector (matrix) @p fine_rhs to the coarse vector.
     *
     * The restriction uses the 1D formula
     * @f$ f_c[i] = fine_rhs[2i]+B\cdot A^{-1}\cdot (fine_rhs[2i-1]+fine_rhs[2i+1])@f$ and boundary condition.
     *
     * @param[in] scaled_b scaled matrix B
     * @param[in] fine_rhs rhs BlockVector on fine grid
     *
     * @return coarse rhs BlockVector f_c.
     */
    AlgebraicMatrix<floating> cr_restriction(const AlgebraicMatrix<floating> &scaled_b,
                                             const AlgebraicMatrix<floating> &fine_rhs) const;

    /**
     * Applies one multigrid iteration to the system.
     *
     * @param[in] num_smoothing_steps number of pre- and post-smoothing steps
     * @param[in] f right-hand side
     * @param[in] max_num_iterations maximal number of multigrid iterations
     * @param[in] accuracy absolute accuracy used for termination of the smoothing steps
     * @param[in,out] solution solution to the system which is being updated
     * @return euclidean error of the residual K * beta - rhs_f
     */
    floating multigrid(const unsigned num_smoothing_steps, const BlockVector<floating> &f,
                       const size_t max_num_iterations, const floating accuracy,
                       BlockVector<floating> &solution) const;

    /**
     * Applies the (linear interpolation) prolongation to the coarse-grid right-hand side @p coarse_rhs onto the fine grid.
     * @param coarse_rhs coarse-grid rhs
     * @return @p coarse_rhs prolongated onto the fine grid
     */
    BlockVector<floating> prolongation(const BlockVector<floating> &coarse_rhs) const;

    /**
     * Applies the full restriction to the fine-grid right-hand side @p fine_rhs onto the coarse grid.
     * @param fine_rhs fine-grid rhs
     * @return @p fine_rhs restricted onto the coarse grid
     */
    BlockVector<floating> restriction(const BlockVector<floating> &fine_rhs) const;

    /**
     * Returns the distance between space points in the coarse grid.
     * @return distance between coarse space grid points
     */
    floating get_reduced_grid_step_size() const;

    /**
     * Returns the number of blocks in each row (resp. column) of the reduced, coarse system.
     * @return number of blocks in each row of the coarse system
     */
    SizeType get_coarse_block_dim() const;

    /**
     * Initializes the coarse system and returns the pointer.
     * @return pointer to newly constructed coarse system
     */
    std::unique_ptr<EquidistantBlock1D<floating>> initialize_coarse_system() const;

    /**
     * Returns the current instance's coarse system.
     * @return coarse system
     */
    const EquidistantBlock1D<floating>& get_coarse_system() const;

    /**
     * Performs steps of weighted block Jacobi iteration, until either the maximal number of iterations or the desired absolute
     * accuracy is reached.
     *
     * @param[in] omega relaxation parameter
     * @param[in] rhs right-hand side
     * @param[in] max_num_iterations maximal number of iterations performed
     * @param[in] accuracy desired absolute accuracy
     * @param[in,out] solution initial solution which is being updated
     * @return Euclidean norm of residual K*beta - rhs_f, where beta is corresponding to @p solution
     */
    floating jacobi_iteration(const floating omega, const BlockVector<floating> &rhs, const size_t max_num_iterations,
                              const floating accuracy, BlockVector<floating> &solution) const;

    /**
     * Performs smoothing steps (similar to jacobi_iteration, but without calculation of accuracy due to performance)
     * to the system K*beta = rhs_f, where beta is corresponding to @param solution.
     *
     * @param[in] omega relaxation parameter
     * @param[in] rhs right-hand side
     * @param[in] max_num_iterations maximal number of iterations performed
     * @param[in,out] solution initial solution which is being updated
     */
    void smooth(const floating omega, const BlockVector<floating> &rhs, const size_t max_num_iterations,
                BlockVector<floating> &solution) const;

    /**
     * Calculate total residual K*@p beta-@p rhs.
     *
     * @param[in] beta approximate solution of auxiliary system
     * @param[in] rhs right-hand side
     * @return residual K*@p beta-@p rhs
     */
    BlockVector<floating> calculate_residual(const BlockVector<floating> &beta, const BlockVector<floating> &rhs) const;

    /**
     * Calculate block row @param i of residual K*beta - rhs_f
     * @param[in] i block row index
     * @param[in] beta_i @p i-th block of approximate solution of auxiliary system
     * @param[in] rhs right-hand side
     * @return @p i-th block row of residual K*beta - @p rhs, must be previously allocated with same size as @p rhs[0]
     */
    AlgebraicVector<floating> calculate_row_residual(const SizeType i, const BlockVector<floating> &beta_i,
                                                     const BlockVector<floating> &rhs) const;

    /**
     * Rescales the right-hand side to fit the symmetrization of the system.
     * @param rhs right-hand side
     * @return rescaled right-hand side
     */
    BlockVector<floating> rescale_rhs(const BlockVector<floating> &rhs) const;
};

/**     Class NonEquidistantBlock1D is used for storing the large system matrix occurring when
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
class NonEquidistantBlock1D
{
public:
    using SizeType = typename AlgebraicMatrix<floating>::SizeType;

    /** Constructor. @p B, @p M and @p D have to be square AlgebraicMatrix, and they must have the same dimensions.
     *
     * @param[in] block_dim number of row respectively column blocks
     * @param[in] B matrix for off diagonal blocks
     * @param[in] D used for calculation of A, the diagonal blocks
     * @param[in] M matrix containing Dirichlet conditions
     * @param[in] grid AlgebraicVector containing distances between subsequent space grid points
     * @param[in] alpha anomalous diffusion coefficient
     * @param[in] time_grid_step_size distance between the equidistant space points
     * @param[in] processing_unit processing_unit with which the calculations should be performed
     */
    NonEquidistantBlock1D(const SizeType block_dim,
                          const AlgebraicMatrix<floating> &B, const CoefficientMatrix<floating> &D,
                          const AlgebraicMatrix<floating> &M, const AlgebraicVector<floating> &grid,
                          const floating alpha, const floating time_grid_step_size,
                          const ProcessingUnit<floating> processing_unit);

    /** Constructor. @p B and @p M have to be square, and they must all have the same dimension as
     * CoefficientMatrix @p D.
     *
     * @param[in] block_dim number of row respectively column blocks
     * @param[in] B matrix for off diagonal blocks
     * @param[in] D used for calculation of A, the diagonal blocks
     * @param[in] M matrix containing Dirichlet conditions
     * @param[in] grid AlgebraicVector containing distances between subsequent space grid points
     * @param[in] alpha anomalous diffusion coefficient
     * @param[in] time_grid_step_size distance between the equidistant space points
     * @param[in] processingUnit processingUnit with which the calculations should be performed
     */
    NonEquidistantBlock1D(const SizeType block_dim,
                          const AlgebraicMatrix<floating> &B, const AlgebraicMatrix<floating> &D,
                          const AlgebraicMatrix<floating> &M, const AlgebraicVector<floating> &grid,
                          const floating alpha, const floating time_grid_step_size,
                          const ProcessingUnit<floating> processingUnit);

    /**
     * Destructor.
     */
    ~NonEquidistantBlock1D() = default;

    /** Returns the order of a single (square) block.
     * @return Order of a single block.
     */
    SizeType get_num_blocks() const;

    /** Returns block row dimension. (Equal to block column dimension, since the system is square.)
     * @return Number of blocks in a row.
     */
    SizeType get_block_dim() const;

    /** Order of the square system when interpreted as a large dense matrix.
     * @return Dense system dimension.
     */
    SizeType get_dense_dim() const;

    /**
     * Returns the container factory which can be used to create new containers
     * on the same device with the same processing unit.
     * @return container factory
     */
    ContainerFactory<floating> get_container_factory() const;

    /**
     * Returns the system as a dense matrix.
     * @return Dense matrix representation stored columnwise.
     */
    AlgebraicMatrix<floating> get_dense_representation() const;

    /**
     * Performs the multiplication K*@p beta, where K is the auxiliary
     * system stored in this class, and returns it in @p result.
     * Note that @p result has to have as many elements as @p beta.
     * @param[in] beta solution
     * @param[out] result result of multiplication
     */
    void multiply(const BlockVector<floating> &beta, BlockVector<floating> &result) const;

    /**
     * Solves the auxiliary linear system using a SolvingProcedure with the given parameters,
     * and then calculates the PDE solution BlockVector out of it.
     *
     * Note that BiCGStab and PCBiCGStab are not available.
     * The solution is a matrix consisting of dimension (N+3)x(M+1),
     * where N is the number of time intervals in the time grid,
     * and M is the number of space intervals in the space grid.
     * The entry solution(i, j) is
     *     *  for i == 0:
     *        the boundary condition d/dt u(x_j, 0)
     *     *  for i == N+2:
     *        the boundary condition d/dt u(x_j, t_N)
     *     *  otherwise, it is the numerical PDE solution u(x_j, t_(i-1))
     *        in the space point x_j and the time point t_(i-1),
     *        where i is in [[1, N+1]] and j is in [[0, M]].
     *
     * The right-hand side @p rhs is obtained by executing the function initialize_rhs in fractional_pde.h.
     *
     * Note that CyclicReduction is not available.
     *
     * @param[in] rhs right-hand side
     * @param[in] max_num_iterations maximal number of iterations (not used for CyclicReduction)
     * @param[in] steps_per_iteration number of steps in each iteration (not used for CyclicReduction)
     * @param[in] accuracy desired absolute accuracy
     * @param[in] solving_procedure the solver used to solve the system
     * @return the numerical solution to the PDE as BlockVector
     */
    BlockVector<floating> solve_pde(const BlockVector<floating> &rhs, const size_t max_num_iterations,
                                    const size_t steps_per_iteration, const floating accuracy,
                                    const SolvingProcedure solving_procedure) const;

    /**
     * Solves the auxiliary linear system K * beta = rhs for beta using a SolvingProcedure with the given parameters.
     * Note that CyclicReduction is not available.
     *
     * @param[in] rhs right-hand side
     * @param[in] max_num_iterations maximal number of iterations (not used for CyclicReduction)
     * @param[in] steps_per_iteration number of steps in each iteration (not used for CyclicReduction)
     * @param[in] accuracy desired absolute accuracy
     * @param[in] solving_procedure the solver used to solve the system
     * @return the solution beta
     */
    BlockVector<floating> solve(const BlockVector<floating> &rhs, const size_t max_num_iterations,
                                const size_t steps_per_iteration, const floating accuracy,
                                const SolvingProcedure solving_procedure) const;

private:
    SizeType _block_dim; //!< number of block rows and block columns
    AlgebraicMatrix<floating> _B; //!< matrix contained in the system's subdiagonal blocks
    AlgebraicMatrix<floating> _D; //!< matrix used for calculation of matrix A contained in the diagonal blocks
    AlgebraicMatrix<floating> _M; //!< Dirichlet boundary condition matrix
    floating _alpha; //!< anomalous diffusion coefficient
    floating _time_grid_step_size; //!< step size between points of the equidistant time grid
    AlgebraicVector<floating> _grid; //!< vector of differences between points of the space grid.
    AlgebraicMatrix<floating> _C; //!< matrix used for approximation of the matrices A_i occurring in the system's diagonals
    ContainerFactory<floating> _container_factory; //!< factory for creating algebraic containers using suitable processing unit (e.g. Cpu or Gpu)
    std::unique_ptr<NonEquidistantBlock1D<floating>> _coarse_system; //!< for optimization purposes, the coarse system for Multigrid is allocated at construction
    mutable AlgebraicMatrix<floating> _vector_buffer; //!< buffer where the individual columns can be used independently internally
    mutable AlgebraicVector<floating> _host_h; //!< the vector of space grid increments is stored on the Cpu separately, because it is needed there very often
    mutable AlgebraicMatrix<floating> _buffer_1; //!< buffer holding an AlgebraicMatrix/BlockVector for storage of residuals etc.
    mutable AlgebraicMatrix<floating> _buffer_2; //!< buffer holding an AlgebraicMatrix/BlockVector for storage of residuals etc.
    mutable AlgebraicMatrix<floating> _buffer_3; //!< buffer holding an AlgebraicMatrix/BlockVector for storage of residuals etc.
    mutable AlgebraicMatrix<floating> _buffer_4; //!< buffer holding an AlgebraicMatrix/BlockVector for storage of residuals etc.
    mutable AlgebraicMatrix<floating> _buffer_5; //!< buffer holding an AlgebraicMatrix/BlockVector for storage of residuals etc.
    mutable AlgebraicMatrix<floating> _buffer_6; //!< buffer holding an AlgebraicMatrix/BlockVector for storage of residuals etc.
    mutable AlgebraicMatrix<floating> _coarse_buffer_1; //!< buffer holding an AlgebraicMatrix/BlockVector for storage of coarse residuals etc.
    mutable AlgebraicMatrix<floating> _coarse_buffer_2; //!< buffer holding an AlgebraicMatrix/BlockVector for storage of coarse residuals etc.

    /**
     * Applies steps of BiCGStab to the system K*beta = rhs, where beta is corresponding to @p solution,
     * until either the maximal number of iterations or the desired absolute accuracy is reached.
     *
     * @param[in] rhs right-hand side
     * @param[in] max_num_iterations maximal number of iterations performed
     * @param[in] accuracy desired absolute accuracy
     * @param[in,out] solution solution initial solution which is being updated
     * @return Euclidean norm of residual K*beta-@p rhs, where beta is corresponding to @p solution
     */
    floating biCGStab(const BlockVector<floating> &rhs, const size_t max_num_iterations, const floating accuracy,
                      BlockVector<floating> &solution) const;

    /**
     * Applies steps of multigrid-preconditioned BiCGStab to the system K*beta = rhs, where beta is corresponding
     * to @param solution, until either the maximal number of iterations or the desired absolute accuracy is reached.
     *
     * @param[in] rhs right-hand side
     * @param[in] max_num_iterations maximal number of iterations performed
     * @param[in] accuracy desired absolute accuracy
     * @param[in,out] solution solution initial solution which is being updated
     * @return Euclidean norm of residual K*beta - rhs, where beta is corresponding to @p solution
     */
    floating PCbiCGStab(const BlockVector<floating> &rhs, const size_t max_num_iterations, const floating accuracy,
                        BlockVector<floating> &solution) const;

    /**
     * Applies one multigrid iteration to the system.
     *
     * @param[in] num_smoothing_steps number of pre- and post-smoothing steps
     * @param[in] rhs right-hand side
     * @param[in] max_num_iterations maximal number of multigrid iterations
     * @param[in] accuracy absolute accuracy used for termination of the smoothing steps
     * @param[in/out] solution solution to the system which is being updated
     */
    void multigrid(const unsigned num_smoothing_steps, const BlockVector<floating> &rhs,
                   const size_t max_num_iterations, const floating accuracy, BlockVector<floating> &solution) const;

    /**
     * Applies the (linear interpolation) prolongation to the coarse-grid right-hand side @p coarse_rhs onto the fine grid.
     * @param[in] coarse_rhs coarse-grid rhs
     * @param[out] fine_rhs @p coarse_rhs prolongated onto the fine grid, must be previously allocated
     */
    void prolongation(const BlockVector<floating> &coarse_rhs, BlockVector<floating> &fine_rhs) const;

    /**
     * Applies the full restriction to the fine-grid right-hand side @p fine_rhs onto the coarse grid.
     * @param fine_rhs fine-grid rhs
     * @param coarse_rhs @p fine_rhs restricted onto the coarse grid, must be previously allocated
     */
    void restriction(const BlockVector<floating> &fine_rhs, BlockVector<floating> &coarse_rhs) const;

    /**
     * Returns the distance between space points in the coarse grid.
     * @return distance between coarse space grid points
     */
    AlgebraicVector<floating> get_reduced_grid() const;

    /**
     * Returns the number of blocks in each row (resp. column) of the reduced, coarse system.
     * @return number of blocks in each row of the coarse system
     */
    SizeType get_coarse_dim() const;

    /**
     * Initializes the coarse system and returns the pointer.
     * @return pointer to newly constructed coarse system
     */
    std::unique_ptr<NonEquidistantBlock1D<floating>> initialize_coarse_system() const;

    /**
     * Returns the current instance's coarse system.
     * @return coarse system
     */
    const NonEquidistantBlock1D<floating>& get_coarse_system() const;

    /**
     * Performs smoothing steps (weighted block Jacobi iteration)
     * to the system K*beta = @p rhs, where beta is corresponding to @p solution,
     * until the maximal number of iterations is reached
     *
     * @param[in] omega relaxation parameter
     * @param[in] rhs right-hand side
     * @param[in] max_num_iterations maximal number of iterations performed
     * @param[in,out] solution initial solution which is being updated
     */
    void smooth(const floating omega, const BlockVector<floating> &rhs, const size_t max_num_iterations,
                BlockVector<floating> &solution) const;

    /**
     * Calculate total residual K*@p beta-@p rhs.
     *
     * @param[in] beta approximate solution of auxiliary system
     * @param[in] rhs right-hand side
     * @param[out] residual K*@p beta-@p rhs, must be previously allocated with same size as @p rhs
     */
    void calculate_residual(const BlockVector<floating> &beta, const BlockVector<floating> &rhs,
                            BlockVector<floating> &residual) const;

    /**
     * Calculate block row @param i of residual K*beta - rhs_f
     * @param[in] i block row index
     * @param[in] beta_i @p i-th block of approximate solution of auxiliary system
     * @param[in] rhs right-hand side
     * @param[out] @p i-th block row of residual K*beta - @p rhs, must be previously allocated with same size as @p rhs[0]
     */
    void calculate_row_residual(const SizeType i, const BlockVector<floating> &beta_i, const BlockVector<floating> &rhs,
                                AlgebraicVector<floating> &residual) const;

    /**
    * Calculates the coefficient needed in the calculation of matrix A.
    *
    * @param alpha anomalous diffusion coefficient
    * @param timeGridStepSize distance between the equidistant space points
    * @return coefficient needed for calculation of matrix A
    */
    floating get_system_coeff() const;

    /**
     * Calculate matrix C, approximating the matrices A_i which occur in the system's diagonals.
     * @return matrix C
     */
    AlgebraicMatrix<floating> initialize_c() const;

    /**
     * Returns the current instance's underlying processing unit.
     * @return current processing unit
     */
    ProcessingUnit<floating> get_processing_unit() const;

    /**
     * For the approximation of matrices A_i which occur in the system's diagonals,
     * matrix C is computed, needing a preconditioner depending on all space grid increments h_i.
     * @param grid vector of space grid increments
     * @return maximal entry max_i (h_i)
     */
    floating get_grid_preconditioner(const AlgebraicVector<floating> &grid) const;

    /**
     * Rescales the right-hand side to fit the symmetrization of the system.
     * @param[in/out] rhs right-hand side being rescaled
     */
    void rescale_rhs(BlockVector<floating> &rhs) const;
};

#include "blocksys_equidistant.hpp"
#include "blocksys_nonequidistant.hpp"
#endif
