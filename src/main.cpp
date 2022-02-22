//#define NDEBUG

//#define MAGMA 1
//#define UNSYMMETRIZED 1
//#define CPU_ONLY 1
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
    ProcessingUnit<floating> pu = std::make_shared<CPU<floating>>();
    //ProcessingUnit<floating> pu = std::make_shared<GPU<floating>>();

    const size_t N = twoToThe(5);
    const size_t M = twoToThe(7);

    const floating T = 1;
    const floating alpha = 0.9;

    const size_t maxNumberOfIterations = 20;
    const size_t stepsPerIteration = 20;
    const floating accuracy = 1e-9;

    testEquidistantGeneralSolvingProcedure(pu, N, M, T, alpha,
                                           maxNumberOfIterations, stepsPerIteration, accuracy,
                                           SolvingProcedure::CyclicReduction);

    testNonEquidistantWithGeneralGrid(pu, N, M, T, alpha,
                                      maxNumberOfIterations, stepsPerIteration, accuracy,
                                      SolvingProcedure::PCBiCGStab);

    return 0;
}