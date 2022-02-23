#ifndef DEMO_H
#define DEMO_H

#include "fractional_pde.h"

/**
 * The AnomalousCoefficient alias is for making the kind of function expected clear to the user,
 * marking a function argument as being interpreted as anomalous diffusion coefficient.
 */
template<class floating>
using AnomalousCoefficient = floating;

/**
 * SpaceTimeCoeffFunction is a shortcut for functions f(x, t, alpha),
 * with the first variable x being the space value,
 * the second variable t being the time value,
 * and the third variable alpha being the anomalous diffusion coefficient.
 *
 * The return value is the value of this function in (x, t, alpha)
 * must be a floating point number.
 */
template<class floating>
using SpaceTimeCoeffFunction = std::function<floating(SpacePoint<floating>, TimePoint<floating>, AnomalousCoefficient<floating>)>;

/**
 * This helper function converts the solution for a given fractional PDE to
 * the functional input parameters of the solver.
 *
 * @tparam floating floating point type
 * @param[in] exact_solution exact solution of the given fractional PDE u(x, t, alpha)
 * @param[in] exact_solution_dt time derivative of @p exact_solution
 * @param[in] alpha anomalous diffusion coefficient
 * @return PDEFunctionTuple containing the functional input parameters for the solver
 */
template<class floating>
PDEFunctionTuple<floating> exact_solution_to_pde_condition_functions(const SpaceTimeCoeffFunction<floating>& exact_solution,
                                                                     const SpaceTimeCoeffFunction<floating>& exact_solution_dt,
                                                                     const floating alpha);

/**
 * Allocates an equidistant grid and given the exact solution of the fractional PDE,
 * feeds its boundary and initial value conditions to the solver, and then
 * compares the given exact solution to the numerical solution calculated by the solver.
 *
 * BiCGStab and PCBiCGStab are not supported solvers.
 *
 * @tparam floating floating point type
 * @param[in] processingUnit processingUnit with which the calculations should be performed
 * @param[in] N number of time intervals in the equidistant time grid
 * @param[in] M number of space intervals in the equidistant space grid
 * @param[in] T end of time horizon
 * @param[in] alpha anomalous diffusion coefficient
 * @param[in] exact_solution the function returning the PDE's exact solution u(x, t, alpha)
 * @param[in] exact_solution_dt time derivative of @p exact_solution
 * @param[in] rhs_function the fractional PDE's right-hand side function f(x, t)
 * @param[in] maxNumberOfIterations maximal number of iterations (not used for CyclicReduction)
 * @param[in] stepsPerIteration number of steps in each iteration (not used for CyclicReduction)
 * @param[in] accuracy desired absolute accuracy
 * @param[in] solvingProcedure the solver used to solve the system
 * @return the maximum distance between the exact solution vector and the numerical solution vector
 */
template<class floating>
floating equidistant_test_solver_against_exact_solution(const ProcessingUnit<floating> processingUnit, const int N, const int M,
                                                        const floating T, const floating alpha,
                                                        const SpaceTimeCoeffFunction<floating>& exact_solution,
                                                        const SpaceTimeCoeffFunction<floating>& exact_solution_dt,
                                                        const SpaceTimeCoeffFunction<floating>& rhs_function,
                                                        const size_t maxNumberOfIterations, const size_t stepsPerIteration,
                                                        const floating accuracy, const SolvingProcedure solvingProcedure);

/**
 * Allocates a non-equidistant grid and given the exact solution of the fractional PDE,
 * feeds its boundary and initial value conditions to the solver, and then
 * compares the given exact solution to the numerical solution calculated by the solver.
 *
 * CyclicReduction is not a supported solver.
 *
 * @tparam floating floating point type
 * @param[in] processing_unit processing_unit with which the calculations should be performed
 * @param[in] N number of time intervals in the equidistant time grid
 * @param[in] M number of space intervals in the non-equidistant space grid
 * @param[in] T end of time horizon
 * @param[in] alpha anomalous diffusion coefficient
 * @param[in] exact_solution the function returning the PDE's exact solution u(x, t, alpha)
 * @param[in] exact_solution_dt time derivative of @p exact_solution
 * @param[in] rhs_function the fractional PDE's right-hand side function f(x, t)
 * @param[in] max_num_iterations maximal number of iterations
 * @param[in] steps_per_iteration number of steps in each iteration
 * @param[in] accuracy desired absolute accuracy
 * @param[in] solving_procedure the solver used to solve the system
 * @return the maximum distance between the exact solution vector and the numerical solution vector
 */
template<class floating>
floating non_equidistant_test_solver_against_exact_solution(const ProcessingUnit<floating> processing_unit, const int N, const int M,
                                                            const floating T, const floating alpha,
                                                            const SpaceTimeCoeffFunction<floating>& exact_solution,
                                                            const SpaceTimeCoeffFunction<floating>& exact_solution_dt,
                                                            const SpaceTimeCoeffFunction<floating>& rhs_function,
                                                            const size_t max_num_iterations, const size_t steps_per_iteration,
                                                            const floating accuracy, const SolvingProcedure solving_procedure);

/**
 * Gets the exact solution for all space points in the @p grid,
 * but only for time point @p T.
 *
 * The i-th entry of the returned vector equals u(x_i, T).
 *
 * @tparam floating
 * @param[in] T end of time horizon
 * @param[in] alpha anomalous diffusion coefficient
 * @param[in] exact_solution SpaceTimeCoeffFunction giving the exact solution
 * @param[in] grid the vector of space grid interval lenghts
 * @return the AlgebraicVector containing the solution
 */
template<class floating>
AlgebraicVector<floating> get_exact_solution_vector(const floating T, const floating alpha,
                                                    const SpaceTimeCoeffFunction<floating>& exact_solution,
                                                    const AlgebraicVector<floating> &grid);

/**
 * Creates a space grid in [0, 1],
 * where the grid's contents actually are the increments between the
 * space points.
 *
 * @tparam floating floating point parameter
 * @param[in,out] grid grid, must be allocated and nonempty
 */
template<class floating>
void get_general_grid(AlgebraicVector<floating>& grid);

#include "demo.hpp"

#endif //DEMO_H
