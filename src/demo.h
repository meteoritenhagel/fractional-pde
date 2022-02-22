#ifndef DEMO_H
#define DEMO_H

#include "auxiliary.h"
#include "algebraiccontainers/containerfactory.h"
#include "blocksys.h"
#include "algebraiccontainers/algebraiccontainers.h"
#include "processingunit/processingunit.h"

#include <cmath>
#include <iostream>
#include <tuple>
#include <vector>

template<class floating>
using FunctionTuple = std::tuple<SpaceTimeFunction<floating>, SpaceTimeFunction<floating>, SpaceTimeFunction<floating>, SpaceFunction<floating>, SpaceTimeFunction<floating>>;

template<class floating>
using SpaceTimeCoeffFunction = std::function<floating(floating, floating, floating)>;

template<class floating>
FunctionTuple<floating> exact_solution_to_pde_condition_functions(const SpaceTimeCoeffFunction<floating>& exact_solution,
                                                                  const SpaceTimeCoeffFunction<floating>& exact_solution_dt,
                                                                  const floating alpha);

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
void initializeRhs(const int N, const int M, const floating T,
                   const SpaceTimeFunction<floating> &phi, const SpaceTimeFunction<floating> &varphi,
                   const SpaceTimeFunction<floating> &up_exact, const SpaceFunction<floating> &u_zero,
                   const SpaceTimeFunction<floating> &rhs_function,
                   const AlgebraicVector<floating> &grid, AlgebraicMatrix<floating> &rhs);

template<class floating>
std::vector<floating> exactSolution(const int M, const floating T, const floating alpha,
                                    const AlgebraicVector<floating> &grid);

template<class floating>
void getGeneralGrid(AlgebraicVector<floating>& grid);

#include "demo.hpp"

#endif //DEMO_H
