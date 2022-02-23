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

    // phi and varphi are actually TimeFunction, but for compatibility with rhs_helper they are
    // interpreted as SpaceTimeFunction.
    const SpaceTimeFunction<floating> phi = [exact_solution, alpha](floating x, floating t){ return exact_solution(0., t, alpha); };
    const SpaceTimeFunction<floating> varphi = [exact_solution, alpha](floating x, floating t){ return exact_solution(1., t, alpha); };
    const SpaceTimeFunction<floating> up_exact = [exact_solution_dt, alpha](floating x, floating t){ return exact_solution_dt(x, t, alpha); };
    const SpaceFunction<floating> u_zero = [exact_solution, alpha](floating x){ return exact_solution(x, 0., alpha); };
    const SpaceTimeFunction<floating> rhs = [rhs_function, alpha](floating x, floating t){ return rhs_function(x, t, alpha); };

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
    ProcessingUnit<floating> cpu = std::make_shared<CPU<floating>>();
    ContainerFactory<floating> colMatrixFactory(cpu);

    const auto pde_function_tuple = exact_solution_to_pde_condition_functions(exact_solution, exact_solution_dt, rhs_function, alpha);
    const auto xx = solve_equidistant(processingUnit, N, M, T, alpha,
                                      pde_function_tuple,
                                      maxNumberOfIterations, stepsPerIteration, accuracy, solvingProcedure);

    const auto grid = *colMatrixFactory.createColumn(M,  T / static_cast<floating>(M));

    std::vector<floating> ue(M+1);
    ue = get_exact_solution_vector(M, T, alpha, exact_solution, grid);
    auto ue_device = *colMatrixFactory.createColumn(ue.size());
    memcpy(ue_device.data(), ue.data(), ue.size()*sizeof(floating));
    ue_device.moveTo(processingUnit);

    floating error_max = std::abs((ue_device-xx).getMaximum());
    return error_max;
}

template<class floating>
floating non_equidistant_test_solver_against_exact_solution(const ProcessingUnit<floating> processingUnit, const int N, const int M,
                                                            const floating T, const floating alpha,
                                                            const SpaceTimeCoeffFunction<floating>& exact_solution,
                                                            const SpaceTimeCoeffFunction<floating>& exact_solution_dt,
                                                            const SpaceTimeCoeffFunction<floating>& rhs_function,
                                                            const size_t maxNumberOfIterations, const size_t stepsPerIteration,
                                                            const floating accuracy, const SolvingProcedure solvingProcedure)
{
    const auto pde_function_tuple = exact_solution_to_pde_condition_functions(exact_solution, exact_solution_dt, rhs_function, alpha);

    ProcessingUnit<floating> cpu = std::make_shared<CPU<floating>>();
    ContainerFactory<floating> colMatrixFactory(cpu);

    floating dt = T / static_cast<floating>(N);

    auto grid = *colMatrixFactory.createColumn(M);
    getGeneralGrid(grid);

    const auto xx = solve_nonequidistant(processingUnit, N, M, T, alpha, grid,
                                         pde_function_tuple,
                                         maxNumberOfIterations, stepsPerIteration, accuracy, solvingProcedure);
    std::vector<floating> ue(M+1);
    ue = get_exact_solution_vector(M, T, alpha, exact_solution, grid);
    auto ue_device = *colMatrixFactory.createColumn(ue.size());
    memcpy(ue_device.data(), ue.data(), ue.size()*sizeof(floating));
    ue_device.moveTo(processingUnit);

    floating error_max = std::abs((ue_device-xx).getMaximum());
    return error_max;
}

template<class floating>
std::vector<floating> get_exact_solution_vector(const int M, const floating T, const floating alpha,
                                                const SpaceTimeCoeffFunction<floating>& exact_solution,
                                                const AlgebraicVector<floating> &grid)
{
    std::vector<floating> solution(M+1);

    floating spacePoint = 0;
    for (int i = 0; i < M; i++)
    {
        solution.at(i) = exact_solution(spacePoint, T, alpha);
        spacePoint += grid[i];
    }
    solution.at(M) = exact_solution(spacePoint, T, alpha);

    return solution;
}

template<class floating>
void getGeneralGrid(AlgebraicVector<floating>& grid)
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