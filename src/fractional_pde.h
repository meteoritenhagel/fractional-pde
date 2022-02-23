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
 * SpaceTimeFunction is a shortcut for functions rhs_f(x, t),
 * with the first variable x being the space value
 * and the second variable t being the time value.
 *
 * The return value is the value of this function in (x, t)
 * must be a floating point number.
 */
template<class floating>
using SpaceTimeFunction = std::function<floating(floating, floating)>;

/**
 * SpaceFunction is a shortcut for functions rhs_f(x),
 * with the variable x being the space value.
 *
 * The return value is the value of this function in x
 * must be a floating point number.
 */
template<class floating>
using SpaceFunction = std::function<floating(floating)>;

template<class floating>
using PDEFunctionTuple = std::tuple<SpaceTimeFunction<floating>,
                                    SpaceTimeFunction<floating>,
                                    SpaceTimeFunction<floating>,
                                    SpaceFunction<floating>,
                                    SpaceTimeFunction<floating>>;

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

/** Given a square matrix @p B, B is modified along the tridiagonal to contain the values a, b, a like this:
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
void applyTriDiagonals(const floating a, const floating b, AlgebraicMatrix<floating> &B);

/**
 * Calculates the vectors f_0 and f_M as in in equation (3) of
 * https://unipub.uni-graz.at/obvugrhs/download/pdf/6408510
 * The output parameter rhs_entry has to have the size N+3, where
 * N is the number of equidistant time steps.
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

template<class floating>
AlgebraicVector<floating> solve_equidistant(const ProcessingUnit<floating> processingUnit,
                                            const int N, const int M, const floating T, const floating alpha,
                                            const PDEFunctionTuple<floating>& pde_function_tuple,
                                            const size_t maxNumberOfIterations, const size_t stepsPerIteration,
                                            const floating accuracy, const SolvingProcedure solvingProcedure);

template<class floating>
AlgebraicVector<floating> solve_nonequidistant(const ProcessingUnit<floating> processingUnit,
                                               const int N, const int M, const floating T, const floating alpha,
                                               const AlgebraicVector<floating>& grid,
                                               const PDEFunctionTuple<floating>& pde_function_tuple,
                                               const size_t maxNumberOfIterations, const size_t stepsPerIteration,
                                               const floating accuracy, const SolvingProcedure solvingProcedure);

template<class floating>
void initializeMatricesEquidistant(const int N, const floating T,
                                   AlgebraicMatrix<floating> &B, AlgebraicMatrix<floating> &MM);

template<class floating>
void initializeMatricesNonEquidistant(const int N, const floating T,
                                      AlgebraicMatrix<floating> &B, AlgebraicMatrix<floating> &MM);

template<class floating>
void initializeRhs(const int N, const int M, const floating T, const PDEFunctionTuple<floating> &pde_function_tuple,
                   const AlgebraicVector<floating> &grid, AlgebraicMatrix<floating> &rhs);


#include "fractional_pde.hpp"

#endif
