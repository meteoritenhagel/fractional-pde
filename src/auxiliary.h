#ifndef FILE_MYLIB
#define FILE_MYLIB

#include "algebraiccontainers/algebraiccontainers.h"
#include "devicedata/devicedata.h"

#include <vector>

// SFINAE helper to ensure that T is integral type
template<typename T>
using enable_if_is_integral = std::enable_if_t<std::is_integral_v<T>, bool>;

/**
 * Calculates 2 to the power of @param x.
 * @tparam T integral type
 * @param x exponent
 * @return 2^x
 */
template<class T, enable_if_is_integral<T> = true>
static constexpr T twoToThe(T const x);

/** Generates an equidistant vector having @p N elements from interval [@p a, @p b].
 *
 *  @param N number of elements
 * 	@param[in] a	interval start
 *  @param[in] b    interval end
 *
 *	@return    vector with the equidistant elements
*/
template<class floating>
std::vector<floating> linspace(const int N, const floating a, const floating b);

/** Given a square matrix @param B, B is modified along the tridiagonal to contain the values a, b, a like this:
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

/** The right hand side function f(x, t) used in the differential equation.
 *  See also https://unipub.uni-graz.at/obvugrhs/download/pdf/6408510 in the
 *  formulation of the fractional PDE.
 *
 *  Note that f does not need to be dependent on @param alpha.
 *
 * @tparam floating floating point type
 * @param x space value
 * @param t time value
 * @param alpha anomalous diffusion coefficient
 * @return f(x, t)
 */
template<class floating>
floating f(const floating x, const floating t, const floating alpha);

/**
 * The solution of the PDE, only used for comparing the solver's result
 * with the known exact result.
 *
 * @tparam floating floating point type
 * @param x space value
 * @param t time value
 * @param alpha anomalous diffusion coefficient
 * @return The exact solution u(x, t) to the PDE as input.
 */
template<class floating>
floating u_exact(const floating x, const floating t, const floating alpha);

/**
 * The time derivative of the solution u(x, t), i.e. d/dt u(x, t).
 * It is used as some kind of boundary condition in the PDE's discretization.
 * Refer to the calculation of f in equation (3) of
 * https://unipub.uni-graz.at/obvugrhs/download/pdf/6408510
 *
 * Note that up_exact does not need to be dependent on @param alpha.
 *
 * @tparam floating floating point type
 * @param x space value
 * @param t time value
 * @param alpha anomalous diffusion coefficient
 * @return d/dt u(x, t)
 */
template<class floating>
floating up_exact(floating const x, floating const t, floating const alpha);

// TODO: Change constF1 and constF2 to contain phi and varphi boundary conditions
// TODO: hand over u_exact and up_exact as parameters

/**
 * Calculates the vectors f_0 and f_M as in in equation (3) of
 * https://unipub.uni-graz.at/obvugrhs/download/pdf/6408510
 * The output parameter ff has to have the size N+3, where
 * N is the number of equidistant time steps.
 *
 * @tparam floating floating point type
 * @param x space value
 * @param T end of time horizon
 * @param alpha anomalous diffusion coefficient
 * @param[out] ff The right-hand side result
 */
template<class floating>
void constF1(const floating x, const floating T, const floating alpha, AlgebraicVector<floating>& ff);


// TODO: hand over f, u_exact and up_exact as parameters

/**
 * Calculates the vectors f_1 to f_M-1 as in in equation (3) of
 * https://unipub.uni-graz.at/obvugrhs/download/pdf/6408510
 * The output parameter ff has to have the size N+3, where
 * N is the number of equidistant time steps.
 *
 * @tparam floating floating point type
 * @param x space value
 * @param T end of time horizon
 * @param alpha anomalous diffusion coefficient
 * @param[out] ff The right-hand side result
 */
template<class floating>
void constF2(const floating x, const floating T, const floating alpha, AlgebraicVector<floating>& ff);

#include "auxiliary.hpp"

#endif
