#ifndef FILE_MYLIB
#define FILE_MYLIB

#include "algebraiccontainers/algebraiccontainers.h"
#include "devicedata/devicedata.h"

#include <algorithm>  // std::max()
#include <cassert>
#include <cmath>      // exp()
#include <vector>

template<typename T>
using enable_if_is_integral = std::enable_if_t<std::is_integral_v<T>, bool>;

template<class T, enable_if_is_integral<T> = true>
static constexpr T twoToThe(T const x)
{
    return 1<<x;
}

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

template<class floating, class S, class T>
floating max_norm(S const &u, T const &w);

template<class floating>
void applyTriDiagonals(const floating a, const floating b, AlgebraicMatrix<floating> &B);

template<class floating>
floating f(const floating x, const floating t, const floating alpha);

template<class floating>
floating u_exact(const floating x, const floating t, const floating alpha);

template<class floating>
floating up_exact(floating const x, floating const t, floating const alpha);

template<class floating>
void constF1(const floating x, const floating T, const floating alpha, AlgebraicVector<floating>& ff);

template<class floating>
void constF2(const floating x, const floating T, const floating alpha, AlgebraicVector<floating>& ff);

#include "auxiliary.hpp"

#endif
