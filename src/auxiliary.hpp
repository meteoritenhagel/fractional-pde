#include "auxiliary.h"

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

template<class floating, class S, class T>
floating max_norm(S const &u, T const &w)
{
    assert(u.size() == w.size());
    floating s = 0.0;
    for (unsigned int k = 0; k < u.size(); ++k) {
        floating u1 = u[k];
        floating w1 = w[k];
        const floating diff1 = u1 - w1;
        floating difference = std::fabs(diff1);
        s = std::max(s, difference);
    }
    return s;
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
//    return exp(x) * pow(t, 4.0) * (tgamma(5 + alpha) / 24.0 - pow(t, alpha));
    return sin(M_PI*x) * (pow(t, 4) * tgamma(5+alpha) / 24.0 + M_PI*M_PI*pow(t, 4+alpha));
}

template<class floating>
floating u_exact(const floating x, const floating t, const floating alpha)
{
//    return exp(x) * pow(t, 4 + alpha);
    return pow(t, 4+alpha) * sin(M_PI*x);
}

template<class floating>
floating up_exact(floating const x, floating const t, floating const alpha)
{
//    return (4 + alpha) * exp(x) * pow(t, 3 + alpha);
    return (4+alpha) * pow(t, 3 + alpha) * sin(M_PI*x);
}

template<class floating>
void constF1(const floating x, const floating T, const floating alpha, AlgebraicVector<floating>& ff)
{
    const int N = ff.size();
    std::vector<floating> t = linspace<floating>(N - 2, 0, T);

    ff[0] = up_exact(x, static_cast<floating>(0.0), alpha);
    ff[1] = u_exact(x, static_cast<floating>(0.0), alpha);
    for (int i = 2; i < N - 1; i++) ff[i] = u_exact(x, t.at(i - 1), alpha);
    ff[N - 1] = up_exact(x, T, alpha);
    return;
}

template<class floating>
void constF2(const floating x, const floating T, const floating alpha, AlgebraicVector<floating>& ff)
{
    const int N = ff.size();
    std::vector<floating> t = linspace<floating>(N - 2, 0, T);
    ff[0] = up_exact(x, static_cast<floating>(0.0), alpha);
    ff[1] = u_exact(x, static_cast<floating>(0.0), alpha);
    ff[N - 1] = up_exact(x, T, alpha);
    for (int i = 2; i < N - 1; i++) ff[i] = f(x, t.at(i - 1), alpha);
    return;
}
