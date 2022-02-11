#ifndef DEMO_H
#define DEMO_H

#include "blocksys.h"
#include "algebraiccontainers/algebraiccontainers.h"
#include "processingunit/processingunit.h"

#include <cmath>
#include <iostream>
#include <vector>

#include "auxiliary.h"
#include "algebraiccontainers/containerfactory.h"

template<class floating>
floating testEquidistantGeneralSolvingProcedure(const ProcessingUnit<floating> processingUnit,
                                                const int N, const int M, const floating T, const floating alpha,
                                                const size_t maxNumberOfIterations, const size_t stepsPerIteration,
                                                const floating accuracy, const SolvingProcedure solvingProcedure);

template<class floating>
floating testNonEquidistantWithGeneralGrid(const ProcessingUnit<floating> processingUnit, const int N, const int M,
                                           const floating T, const floating alpha,
                                           const size_t maxNumberOfIterations, const size_t stepsPerIteration,
                                           const floating accuracy, const SolvingProcedure solvingProcedure);

template<class floating>
void initializeMatricesEquidistant(const int N, const floating T,
                                   AlgebraicMatrix<floating> &B, AlgebraicMatrix<floating> &MM);

template<class floating>
void initializeMatricesNonEquidistant(const int N, const floating T,
                                      AlgebraicMatrix<floating> &B, AlgebraicMatrix<floating> &MM);
template<class floating>
void initializeRhs(const int N, const int M, const floating T, const floating alpha,
                   const AlgebraicVector<floating> &grid, AlgebraicMatrix<floating> &rhs);
template<class floating>
std::vector<floating> exactSolution(const int M, const floating T, const floating alpha,
                                    const AlgebraicVector<floating> &grid);

template<class floating>
void getGeneralGrid(AlgebraicVector<floating>& grid);

#include "demo.hpp"

#endif //DEMO_H
