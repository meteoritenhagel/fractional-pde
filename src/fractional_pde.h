#ifndef FILE_MYLIB
#define FILE_MYLIB

#include "algebraiccontainers/algebraiccontainers.h"
#include "fractional_pde.h"
#include "blocksys.h"
#include "devicedata/devicedata.h"
#include "processingunit/processingunit.h"

#include <functional>
#include <vector>

// SFINAE helper to ensure that T is integral type
template<typename T>
using enable_if_is_integral = std::enable_if_t<std::is_integral_v<T>, bool>;

/**
 * The SpacePoint alias is for making the kind of function expected clear to the user,
 * marking a function argument as belonging to the time domain.
 */
template<class floating>
using SpacePoint = floating;

/**
 * The TimePoint alias is for making the kind of function expected clear to the user,
 * marking a function argument as belonging to the space domain.
 */
template<class floating>
using TimePoint = floating;

/**
 * SpaceTimeFunction is a shortcut for functions f(x, t),
 * with the first variable x being the space value
 * and the second variable t being the time value.
 *
 * The return value is the value of this function in (x, t)
 * must be a floating point number.
 */
template<class floating>
using SpaceTimeFunction = std::function<floating(SpacePoint<floating>, TimePoint<floating>)>;

/**
 * SpaceFunction is a shortcut for functions f(x),
 * with the variable x being the space value.
 *
 * The return value is the value of this function in x
 * must be a floating point number.
 */
template<class floating>
using SpaceFunction = std::function<floating(SpacePoint<floating>)>;

/**
 * TimeFunction is a shortcut for functions f(t),
 * with the variable t being the space value.
 *
 * The return value is the value of this function in t
 * must be a floating point number.
 */
template<class floating>
using TimeFunction = std::function<floating(TimePoint<floating>)>;

/**
 * PDEFunctionTuple is a shortcut for all the
 * functional input arguments the solver is needing.
 * The variables in the tuple correspond to the following
 * functions as described in the introduction of
 * https://unipub.uni-graz.at/obvugrhs/download/pdf/6408510
 *
 * phi(t), a Dirichlet boundary condition
 * varphi(t), a Dirichlet boundary condition
 * d/dt u(x, t), which must be given for t=0 and t=T, and all space points
 * u_0(x), an initial value condition
 * f(x, t), the right-hand side function of the fractional PDE
 */
template<class floating>
using PDEFunctionTuple = std::tuple<TimeFunction<floating>,
                                    TimeFunction<floating>,
                                    SpaceTimeFunction<floating>,
                                    SpaceFunction<floating>,
                                    SpaceTimeFunction<floating>>;

/**
 * Function solve_equidistant returns the solution matrix of the fractional PDE
 * using a discretization according to the so-called S3 formula.
 *
 * Both time and space grid are automatically created as equidistant.
 *
 * Note that BiCGStab and PCBiCGStab are not available.
 *
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
 * @tparam floating floating point type
 * @param[in] processingUnit processingUnit with which the calculations should be performed
 * @param[in] N number of time intervals in the equidistant time grid
 * @param[in] M number of space intervals in the equidistant space grid
 * @param[in] T end of time horizon
 * @param[in] alpha anomalous diffusion coefficient
 * @param[in] pde_function_tuple the input functions (containing boundary conditions, right-hand side function, etc.) bundled as a tuple
 * @param[in] maxNumberOfIterations maximal number of iterations (not used for CyclicReduction)
 * @param[in] stepsPerIteration number of steps in each iteration (not used for CyclicReduction)
 * @param[in] accuracy desired absolute accuracy
 * @param[in] solvingProcedure the solver used to solve the system
 * @return the numerical solution to the PDE as BlockVector
 */
template<class floating>
BlockVector<floating> solve_equidistant(const ProcessingUnit<floating> processingUnit,
                                            const int N, const int M, const floating T, const floating alpha,
                                            const PDEFunctionTuple<floating>& pde_function_tuple,
                                            const size_t maxNumberOfIterations, const size_t stepsPerIteration,
                                            const floating accuracy, const SolvingProcedure solvingProcedure);

/**
 * Function solve_equidistant returns the solution matrix of the fractional PDE
 * using a discretization according to the so-called S3 formula.
 *
 * The time grid is constructed as equidistant,
 * and the space grid is an input parameter @p grid.
 *
 * Note that CyclicReduction is not available.
 *
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
 * @tparam floating floating point type
 * @param[in] processingUnit processingUnit with which the calculations should be performed
 * @param[in] N number of time intervals in the time grid
 * @param[in] T end of time horizon
 * @param[in] alpha anomalous diffusion coefficient
 * @param[in] grid the vector of space grid interval lenghts
 * @param[in] pde_function_tuple the input functions (containing boundary conditions, right-hand side function, etc.) bundled as a tuple
 * @param[in] maxNumberOfIterations maximal number of iterations (not used for CyclicReduction)
 * @param[in] stepsPerIteration number of steps in each iteration (not used for CyclicReduction)
 * @param[in] accuracy desired absolute accuracy
 * @param[in] solvingProcedure the solver used to solve the system
 * @return the numerical solution to the PDE as BlockVector
 */
template<class floating>
BlockVector<floating> solve_nonequidistant(const ProcessingUnit<floating> processingUnit,
                                               const int N, const floating T, const floating alpha,
                                               const AlgebraicVector<floating>& grid,
                                               const PDEFunctionTuple<floating>& pde_function_tuple,
                                               const size_t maxNumberOfIterations, const size_t stepsPerIteration,
                                               const floating accuracy, const SolvingProcedure solvingProcedure);

/**
 * Calculates the system matrices @p B and @p MM
 * for the equidistant space grid.
 *
 * Note that @p B and @p MM must be square matrices of order N+3 each
 * and have been allocated previously
 *
 * @tparam floating floating point type
 * @param N[in] number of time intervals in time grid
 * @param T[in] end of time horizon
 * @param B[out] matrix for off diagonal blocks
 * @param MM[out] matrix containing Dirichlet conditions
 */
template<class floating>
void initialize_matrices(const int N, const floating T,
                         AlgebraicMatrix<floating> &B, AlgebraicMatrix<floating> &MM);

/**
 * Calculates the linear system's right-hand side, which must be a BlockVector
 * in the shape of an (N+3)x(M+1) matrix and previously allocated.
 *
 * @tparam floating floating point type
 * @param N[in] number of time intervals in time grid
 * @param T[in] end of time horizon
 * @param pde_function_tuple[in] the input functions (containing boundary conditions, right-hand side function, etc.) bundled as a tuple
 * @param grid[in] the vector of space grid interval lenghts
 * @param rhs[out] the right-hand side of the linear system
 */
template<class floating>
void initialize_rhs(const int N, const floating T, const PDEFunctionTuple<floating> &pde_function_tuple,
                    const AlgebraicVector<floating> &grid, AlgebraicMatrix<floating> &rhs);

/**
 * Calculates 2 to the power of @p x.
 * @tparam T integral type
 * @param x exponent
 * @return 2^x
 */
template<class T, enable_if_is_integral<T> = true>
static constexpr T twoToThe(T const x);

/** Generates an equidistant vector having @p N elements from interval [@p a, @p b].
 *
 *  @param N number of elements
 * 	@param[in] a    interval start
 *  @param[in] b    interval end
 *
 *  @return    vector with the equidistant elements
*/
template<class floating>
std::vector<floating> linspace(const int N, const floating a, const floating b);

/** Given a square matrix @p B, B is modified along the tridiagonal to contain the values @p a, @p b, @p a like this:
 *
 *        [a     b    a   B_14 ... B_1n]
 *   B =  [B_21  a    b   a    ... B_2n]
 *        [...         ...          ...]
 *        [B_n1 ...       a     b     a]
 *
 * @tparam floating floating point type
 * @param a outer diagonals value
 * @param b diagonal value
 * @param[in, out] B the matrix is modified to contain [a, b, a] on the tridiagonal.
 */
template<class floating>
void apply_tridiagonals(const floating a, const floating b, AlgebraicMatrix<floating> &B);

/**
 * Calculates the vectors f_0 and f_M as in in equation (3) of
 * https://unipub.uni-graz.at/obvugrhs/download/pdf/6408510
 * The output parameter rhs_entry has to have the size N+3, where
 * @p N is the number of equidistant time steps.
 *
 * @tparam floating floating point type
 * @param[in] x space value
 * @param[in] T end of time horizon
 * @param[in] up_exact the time derivative d/dt u(x, t) used for dirichlet boundary condition. Must at least be defined for every space point and time points 0, T.
 * @param[in] u_zero the solution's initial value condition: u(x, 0) = u_zero(x) for every x in [0, 1]
 * @param[in] inner_function the function used for the calculation of the inner vector values
 * @param[out] rhs_entry corresponding sub-vector of right-hand side block vector.
 */
template<class floating>
void rhs_helper(const floating x, const floating T,
                const SpaceTimeFunction<floating> &up_exact, const SpaceFunction<floating> &u_zero,
                const SpaceTimeFunction<floating> &inner_function,
                AlgebraicVector<floating>& rhs_entry);

#include "fractional_pde.hpp"

#endif
