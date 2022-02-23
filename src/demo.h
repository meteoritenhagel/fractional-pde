#ifndef DEMO_H
#define DEMO_H

#include "fractional_pde.h"

template<class floating>
using SpaceTimeCoeffFunction = std::function<floating(floating, floating, floating)>;

template<class floating>
PDEFunctionTuple<floating> exact_solution_to_pde_condition_functions(const SpaceTimeCoeffFunction<floating>& exact_solution,
                                                                     const SpaceTimeCoeffFunction<floating>& exact_solution_dt,
                                                                     const floating alpha);

template<class floating>
floating equidistant_test_solver_against_exact_solution(const ProcessingUnit<floating> processingUnit, const int N, const int M,
                                                        const floating T, const floating alpha,
                                                        const SpaceTimeCoeffFunction<floating>& exact_solution,
                                                        const SpaceTimeCoeffFunction<floating>& exact_solution_dt,
                                                        const SpaceTimeCoeffFunction<floating>& rhs_function,
                                                        const size_t maxNumberOfIterations, const size_t stepsPerIteration,
                                                        const floating accuracy, const SolvingProcedure solvingProcedure);

template<class floating>
floating non_equidistant_test_solver_against_exact_solution(const ProcessingUnit<floating> processingUnit, const int N, const int M,
                                                            const floating T, const floating alpha,
                                                            const SpaceTimeCoeffFunction<floating>& exact_solution,
                                                            const SpaceTimeCoeffFunction<floating>& exact_solution_dt,
                                                            const SpaceTimeCoeffFunction<floating>& rhs_function,
                                                            const size_t maxNumberOfIterations, const size_t stepsPerIteration,
                                                            const floating accuracy, const SolvingProcedure solvingProcedure);

template<class floating>
std::vector<floating> get_exact_solution_vector(const int M, const floating T, const floating alpha,
                                                const SpaceTimeCoeffFunction<floating>& exact_solution,
                                                const AlgebraicVector<floating> &grid);

template<class floating>
void getGeneralGrid(AlgebraicVector<floating>& grid);

#include "demo.hpp"

#endif //DEMO_H
