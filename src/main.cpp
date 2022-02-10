#define NDEBUG

//#define MAGMA 1
//#define UNSYMMETRIZED 1
//#define CPU_ONLY 1
//#define GAUSS_SEIDEL 1
//#define PLU
//#define PRINT

#include "demo.h"

#ifdef CPU_ONLY
#pragma message(" ##########  CPU_ONLY MODE ACTIVATED. GPU features are not available.  ###############")
#endif

#include <iostream>
#include <iomanip>
#include <vector>

int main()
{
    using floating = float;
    ProcessingUnit<floating> pu = std::make_shared<CPU<floating>>();
    //ProcessingUnit<floating> pu = std::make_shared<GPU<floating>>();

    const std::vector<int> Nvec{5};
    const std::vector<int> Mvec{6};

    const size_t maxNumberOfIterations = 20;//400;
    const size_t stepsPerIteration = 20;
    const floating accuracy = 1e-9;
    const int numberOfTimingLoops = 1;

    const std::vector<size_t> Ms{8};
    const size_t Mvalue = 5;
    testSolvingProceduresNonEquidistant(pu, Mvalue, Ms, accuracy, maxNumberOfIterations, stepsPerIteration, 1);
    //testProceduresN(Ns, Mvalue, accuracy, maxNumberOfIterations, stepsPerIteration, numberOfTimingLoops);

    return 0;
}