#include "blocksys.h"
#include "algebraiccontainers/algebraiccontainers.h"
#include "processingunit/processingunit.h"

#include "fractional_pde.h"
#include "algebraiccontainers/containerfactory.h"


template<class floating>
PDEFunctionTuple<floating> exact_solution_to_pde_condition_functions(const SpaceTimeCoeffFunction<floating>& exact_solution,
                                                                     const SpaceTimeCoeffFunction<floating>& exact_solution_dt,
                                                                     const SpaceTimeCoeffFunction<floating>& rhs_function,
                                                                     const floating alpha)
{
    // Define the boundary and initial value conditions based on the given exact solution and its symbolic derivative
    const TimeFunction<floating> phi = [exact_solution, alpha](TimePoint<floating> t){ return exact_solution(0., t, alpha); };
    const TimeFunction<floating> varphi = [exact_solution, alpha](TimePoint<floating> t){ return exact_solution(1., t, alpha); };
    const SpaceTimeFunction<floating> up_exact = [exact_solution_dt, alpha](SpacePoint<floating> x, TimePoint<floating> t){ return exact_solution_dt(x, t, alpha); };
    const SpaceFunction<floating> u_zero = [exact_solution, alpha](SpacePoint<floating> x){ return exact_solution(x, 0., alpha); };
    const SpaceTimeFunction<floating> rhs = [rhs_function, alpha](SpacePoint<floating> x, TimePoint<floating> t){ return rhs_function(x, t, alpha); };

    return std::make_tuple(phi, varphi, up_exact, u_zero, rhs);
}

template<class floating>
floating equidistant_test_solver_against_exact_solution(const ProcessingUnit<floating> processingUnit, const int N, const int M,
                                                        const floating T, const floating alpha,
                                                        const SpaceTimeCoeffFunction<floating>& exact_solution,
                                                        const SpaceTimeCoeffFunction<floating>& exact_solution_dt,
                                                        const SpaceTimeCoeffFunction<floating>& rhs_function,
                                                        const size_t maxNumberOfIterations, const size_t stepsPerIteration,
                                                        const floating accuracy, const SolvingProcedure solvingProcedure)
{
    ProcessingUnit<floating> cpu = std::make_shared<Cpu<floating>>();
    ContainerFactory<floating> colMatrixFactory(cpu);

    const auto pde_function_tuple = exact_solution_to_pde_condition_functions(exact_solution, exact_solution_dt, rhs_function, alpha);
    auto full_solution = solve_equidistant(processingUnit, N, M, T, alpha,
                                           pde_function_tuple,
                                           maxNumberOfIterations, stepsPerIteration, accuracy, solvingProcedure);

    // only take latest time point solution, but for every space point
    full_solution.moveTo(cpu);
    auto xx = full_solution.getRow(N+1);

    const auto grid = *colMatrixFactory.createColumn(M,  T / static_cast<floating>(M));

    const auto ue = get_exact_solution_vector(T, alpha, exact_solution, grid);

    floating error_max = std::abs((ue-xx).getMaximum());
    return error_max;
}

template<class floating>
floating non_equidistant_test_solver_against_exact_solution(const ProcessingUnit<floating> processing_unit, const int N, const int M,
                                                            const floating T, const floating alpha,
                                                            const SpaceTimeCoeffFunction<floating>& exact_solution,
                                                            const SpaceTimeCoeffFunction<floating>& exact_solution_dt,
                                                            const SpaceTimeCoeffFunction<floating>& rhs_function,
                                                            const size_t max_num_iterations, const size_t steps_per_iteration,
                                                            const floating accuracy, const SolvingProcedure solving_procedure)
{
    const auto pde_function_tuple = exact_solution_to_pde_condition_functions(exact_solution, exact_solution_dt, rhs_function, alpha);

    ProcessingUnit<floating> cpu = std::make_shared<Cpu<floating>>();
    ContainerFactory<floating> colMatrixFactory(cpu);

    floating dt = T / static_cast<floating>(N);

    auto grid = *colMatrixFactory.createColumn(M);
    get_general_grid(grid);

    auto full_solution = solve_nonequidistant(processing_unit, N, T, alpha, grid,
                                              pde_function_tuple,
                                              max_num_iterations, steps_per_iteration, accuracy, solving_procedure);

    // only take latest time point solution, but for every space point
    full_solution.moveTo(cpu);
    auto xx = full_solution.getRow(N+1);

    const auto ue = get_exact_solution_vector(T, alpha, exact_solution, grid);

    floating error_max = std::abs((ue-xx).getMaximum());
    return error_max;
}

template<class floating>
AlgebraicVector<floating> get_exact_solution_vector(const floating T, const floating alpha,
                                                    const SpaceTimeCoeffFunction<floating>& exact_solution,
                                                    const AlgebraicVector<floating> &grid)
{
    assert(typeid(*grid.get_processing_unit()) == typeid(*std::make_shared<Cpu<floating>>()) && "Must be on Cpu");
    const auto M = grid.size();
    auto solution = *grid.get_container_factory().createColumn(M + 1);

    floating spacePoint = 0;
    for (int i = 0; i < M; i++)
    {
        solution[i] = exact_solution(spacePoint, T, alpha);
        spacePoint += grid[i];
    }
    solution[M] = exact_solution(spacePoint, T, alpha);

    return solution;
}

template<class floating>
void get_general_grid(AlgebraicVector<floating>& grid)
{
    auto M = grid.size();

    const floating rho = 0.5;
    const floating xs = 1/(rho+1);

    auto x1 = linspace(M/2+1, static_cast<floating>(0), xs);
    auto x2 = linspace(M/2+1, xs, static_cast<floating>(1));

    auto x = std::vector<floating>(M+1);


    for (size_t i = 0; i < x1.size(); ++i)
        x[i] = x1[i];

    for (size_t i = x1.size(); i < x.size(); ++i)
        x[i] = x2[i-x1.size()+1];

    for (size_t i = 0; i < grid.size(); ++i)
    {
        grid[i] = x[i+1]-x[i];
    }

    return;
}