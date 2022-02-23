//#define NDEBUG

//#define GAUSS_SEIDEL 1
//#define PLU
//#define PRINT

#include "demo.h"

#ifdef CPU_ONLY
#pragma message("CPU_ONLY mode activated. GPU features are not available.")
#endif

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
floating u_exact_f(const floating x, const floating t, const floating alpha)
{
    return pow(t, 4+alpha) * sin(M_PI*x);
}

/**
 * The time derivative of the solution u(x, t), i.e. d/dt u(x, t).
 * It is used as some kind of boundary condition in the PDE's discretization.
 * Refer to the calculation of rhs_f in equation (3) of
 * https://unipub.uni-graz.at/obvugrhs/download/pdf/6408510
 *
 * Note that up_exact_f does not need to be dependent on @p alpha.
 *
 * @tparam floating floating point type
 * @param x space value
 * @param t time value
 * @param alpha anomalous diffusion coefficient
 * @return d/dt u(x, t)
 */
template<class floating>
floating up_exact_f(floating const x, floating const t, floating const alpha)
{
    return (4+alpha) * pow(t, 3 + alpha) * sin(M_PI*x);
}

/** The right hand side function rhs_f(x, t) used in the differential equation.
 *  See also https://unipub.uni-graz.at/obvugrhs/download/pdf/6408510 in the
 *  formulation of the fractional PDE.
 *
 *  Note that rhs_f does not need to be dependent on @p alpha.
 *
 * @tparam floating floating point type
 * @param x space value
 * @param t time value
 * @param alpha anomalous diffusion coefficient
 * @return rhs_f(x, t)
 */
template<class floating>
floating rhs_f(const floating x, const floating t, const floating alpha)
{
    return sin(M_PI*x) * (pow(t, 4) * tgamma(5+alpha) / 24.0 + M_PI*M_PI*pow(t, 4+alpha));
}


int main()
{
    using floating = float;
    //ProcessingUnit<floating> pu = std::make_shared<CPU<floating>>();
    ProcessingUnit<floating> pu = std::make_shared<GPU<floating>>();

    const size_t N = two_to_the(5);
    const size_t M = two_to_the(7);

    const floating T = 1;
    const floating alpha = 0.9;

    const size_t max_num_iterations = 20;
    const size_t steps_per_iteration = 20;
    const floating accuracy = 1e-9;

    std::cout << "Calculation via " << pu->display() << std::endl
              << "N (time steps)  = " << N << std::endl
              << "M (space steps) = " << M << std::endl << std::endl
              << "####################################################################" << std::endl
              << "Equidistant grid:" << std::endl;

    const auto error_equi = equidistant_test_solver_against_exact_solution(pu, N, M, T, alpha,
                                                                           u_exact_f<floating>,
                                                                           up_exact_f<floating>,
                                                                           rhs_f<floating>,
                                                                           max_num_iterations, steps_per_iteration,
                                                                           accuracy,
                                                                           SolvingProcedure::CyclicReduction);

    std::cout << std::endl << "max norm of absolute error (equidistant grid) = " << error_equi << std::endl;

    std::cout << "####################################################################" << std::endl << std::endl
              << "####################################################################" << std::endl
              << "Non-equidistant grid:" << std::endl
              << "    target system absolute accuracy: " << accuracy << std::endl << std::endl;

    const auto error_nonequi = non_equidistant_test_solver_against_exact_solution(pu, N, M, T, alpha,
                                                                                  u_exact_f<floating>,
                                                                                  up_exact_f<floating>,
                                                                                  rhs_f<floating>,
                                                                                  max_num_iterations,
                                                                                  steps_per_iteration, accuracy,
                                                                                  SolvingProcedure::PCBiCGStab);
    std::cout << std::endl << "max norm of absolute error (nonequidistant grid) = " << error_nonequi << std::endl
              << "####################################################################" << std::endl;

    return 0;
}