//#define NDEBUG

//#define GAUSS_SEIDEL 1
//#define PLU
//#define PRINT

#include "demo.h"

#ifdef CPU_ONLY
#pragma message("CPU_ONLY mode activated. GPU features are not available.")
#endif

int main()
{
    using floating = float;
    //ProcessingUnit<floating> pu = std::make_shared<CPU<floating>>();
    ProcessingUnit<floating> pu = std::make_shared<GPU<floating>>();

    const size_t N = twoToThe(5);
    const size_t M = twoToThe(7);

    const floating T = 1;
    const floating alpha = 0.9;

    const size_t maxNumberOfIterations = 20;
    const size_t stepsPerIteration = 20;
    const floating accuracy = 1e-9;

    std::cout << "Calculation via " << pu->display() << std::endl
              << "N (time steps)  = " << N << std::endl
              << "M (space steps) = " << M << std::endl << std::endl
              << "####################################################################" << std::endl
              << "Equidistant grid:" << std::endl;

    const auto error_equi = equidistant_test_solver_against_exact_solution(pu, N, M, T, alpha,
                                                                           maxNumberOfIterations, stepsPerIteration,
                                                                           accuracy,
                                                                           SolvingProcedure::CyclicReduction);

    std::cout << std::endl << "max norm of absolute error (equidistant grid) = " << error_equi << std::endl;

    std::cout << "####################################################################" << std::endl << std::endl
              << "####################################################################" << std::endl
              << "Non-equidistant grid:" << std::endl
              << "    target system absolute accuracy: " << accuracy << std::endl << std::endl;

    const auto error_nonequi = non_equidistant_test_solver_against_exact_solution(pu, N, M, T, alpha,
                                                                                  maxNumberOfIterations,
                                                                                  stepsPerIteration, accuracy,
                                                       SolvingProcedure::PCBiCGStab);
    std::cout << std::endl << "max norm of absolute error (nonequidistant grid) = " << error_nonequi << std::endl
              << "####################################################################" << std::endl;

    return 0;
}