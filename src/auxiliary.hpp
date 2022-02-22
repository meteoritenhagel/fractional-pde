#include <cassert>
#include <cmath>      // exp()
#include <algorithm>  // std::max()


template<class T, enable_if_is_integral<T>>
static constexpr T twoToThe(T const x)
{
    return 1<<x;
}

template<class floating>
std::vector<floating> linspace(const int N, const floating a, const floating b)
{
    std::vector<floating> x(N);
    floating const h = (b - a) / (N - 1.0);
    for (int k = 0; k < N; ++k) {
        x[k] = k * h + a;
    }
    return x;
}

template<class floating>
void applyTriDiagonals(const floating a, const floating b, AlgebraicMatrix<floating> &B)
{
    assert(B.isSquare());
    for (unsigned int i = 0; i < B.getNcols(); i++)
    {
        B(i,i) = b;
        if (i != B.getNcols() - 1) B(i,i+1) = a;
        if (i != 0) B(i,i-1) = a;
    }

    return;
}

template<class floating>
floating f(const floating x, const floating t, const floating alpha)
{
    return sin(M_PI*x) * (pow(t, 4) * tgamma(5+alpha) / 24.0 + M_PI*M_PI*pow(t, 4+alpha));
}

template<class floating>
floating u_exact_f(const floating x, const floating t, const floating alpha)
{
    return pow(t, 4+alpha) * sin(M_PI*x);
}

template<class floating>
floating up_exact_f(floating const x, floating const t, floating const alpha)
{
    return (4+alpha) * pow(t, 3 + alpha) * sin(M_PI*x);
}

template<class floating>
void rhs_helper(const floating x, const floating T,
                const SpaceTimeFunction<floating> &up_exact, const SpaceFunction<floating> &u_zero,
                const SpaceTimeFunction<floating> &inner_function,
                AlgebraicVector<floating>& rhs_entry)
{
    const int N = rhs_entry.size();
    std::vector<floating> t = linspace<floating>(N - 2, 0, T);

    rhs_entry[0] = up_exact(x, static_cast<floating>(0.0));
    rhs_entry[1] = u_zero(x);
    for (int i = 2; i < N - 1; i++) rhs_entry[i] = inner_function(x, t.at(i - 1));
    rhs_entry[N - 1] = up_exact(x, T);
    return;
}