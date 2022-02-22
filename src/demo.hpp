#include "demo.h"

#include "blocksys.h"
#include "algebraiccontainers/algebraiccontainers.h"
#include "processingunit/processingunit.h"

#include "auxiliary.h"
#include "algebraiccontainers/containerfactory.h"

template<class floating>
FunctionTuple<floating> exact_solution_to_pde_condition_functions(const SpaceTimeCoeffFunction<floating>& exact_solution,
                                                                  const SpaceTimeCoeffFunction<floating>& exact_solution_dt,
                                                                  const floating alpha)
{
    // Define the boundary and initial value conditions based on the given exact solution and its symbolic derivative

    // phi and varphi are actually TimeFunction, but for compatibility with rhs_helper they are
    // interpreted as SpaceTimeFunction.
    const SpaceTimeFunction<floating> phi = [exact_solution, alpha](floating x, floating t){ return exact_solution(0., t, alpha); };
    const SpaceTimeFunction<floating> varphi = [exact_solution, alpha](floating x, floating t){ return exact_solution(1., t, alpha); };
    const SpaceTimeFunction<floating> up_exact = [exact_solution_dt, alpha](floating x, floating t){ return exact_solution_dt(x, t, alpha); };
    const SpaceFunction<floating> u_zero = [exact_solution, alpha](floating x){ return exact_solution(x, 0., alpha); };
    const SpaceTimeFunction<floating> rhs_function = [alpha](floating x, floating t){ return f(x, t, alpha); };

    return std::make_tuple(phi, varphi, up_exact, u_zero, rhs_function);
}


template<class floating>
floating testEquidistantGeneralSolvingProcedure(const ProcessingUnit<floating> processingUnit,
                                                const int N, const int M, const floating T, const floating alpha,
                                                const size_t maxNumberOfIterations, const size_t stepsPerIteration,
                                                const floating accuracy, const SolvingProcedure solvingProcedure)
{
    const auto [phi, varphi, up_exact, u_zero, rhs_function] = exact_solution_to_pde_condition_functions(u_exact_f<floating>,
                                                                                                         up_exact_f<floating>,
                                                                                                         alpha);
    ProcessingUnit<floating> cpu = std::make_shared<CPU<floating>>();
    ContainerFactory<floating> colMatrixFactory(cpu);

    floating dt = T / static_cast<floating>(N);
    floating dx = T / static_cast<floating>(M);

    const auto grid = *colMatrixFactory.createColumn(M, dx);

    std::cout << "Equidistant space grid." << std::endl;
    std::cout << "Calculation via " << processingUnit->display() << std::endl << std::endl;

    std::cout << "N (time steps)  = " << N << std::endl
              << "M (space steps) = " << M << std::endl;

    auto B = *colMatrixFactory.createMatrix(N+3, N+3);
    auto D = *colMatrixFactory.createCoefficientMatrix(N, alpha);
    auto MM = *colMatrixFactory.createMatrix(N+3, N+3);

    initializeMatricesEquidistant(N, T, B, MM);

    auto rhs = *colMatrixFactory.createMatrix(N+3, M+1);

    initializeRhs<floating>(N, M, T, phi, varphi, up_exact, u_zero, rhs_function, grid, rhs);

    const int block_dim = M + 1;
    EquidistantBlock_1D<floating> C(block_dim, B, D, MM, dx, alpha, dt, processingUnit);

    BlockVector<floating> CC = C.solve(rhs, maxNumberOfIterations, stepsPerIteration, accuracy, solvingProcedure);

    auto B_row = *colMatrixFactory.createColumn(N+3);
    for (int i = 0; i < N+3; i++)  B_row[i] = MM(N+1, i);
    B_row.moveTo(processingUnit);

    auto xx = B_row * CC;

    std::vector<floating> ue(M+1);
    ue = exactSolution(M, T, alpha, grid);
    auto ue_device = *colMatrixFactory.createColumn(ue.size());
    memcpy(ue_device.data(), ue.data(), ue.size()*sizeof(floating));
    ue_device.moveTo(processingUnit);

    floating error_max = std::abs((ue_device-xx).getMaximum());
    std::cout << std::endl << "max norm of absolute error = " << error_max << std::endl << std::endl;
    return error_max;
}

template<class floating>
floating testNonEquidistantWithGeneralGrid(const ProcessingUnit<floating> processingUnit, const int N, const int M,
                                           const floating T, const floating alpha,
                                           const size_t maxNumberOfIterations, const size_t stepsPerIteration,
                                           const floating accuracy, const SolvingProcedure solvingProcedure)
{
    const auto [phi, varphi, up_exact, u_zero, rhs_function] = exact_solution_to_pde_condition_functions(u_exact_f<floating>,
                                                                                                         up_exact_f<floating>,
                                                                                                         alpha);

    ProcessingUnit<floating> cpu = std::make_shared<CPU<floating>>();
    ContainerFactory<floating> colMatrixFactory(cpu);

    floating dt = T / static_cast<floating>(N);

    std::cout << "Non-equidistant space grid." << std::endl;
    std::cout << "Calculation via " << processingUnit->display() << std::endl << std::endl;

    std::cout << "N (time steps)  = " << N << std::endl
              << "M (space steps) = " << M << std::endl;

    auto grid = *colMatrixFactory.createColumn(M);
    getGeneralGrid(grid);

    auto B = *colMatrixFactory.createMatrix(N+3, N+3);
    auto D = *colMatrixFactory.createCoefficientMatrix(N, alpha);
    auto MM = *colMatrixFactory.createMatrix(N+3, N+3);

    initializeMatricesNonEquidistant(N, T, B, MM);

    auto rhs = *colMatrixFactory.createMatrix(N+3, M+1);

    initializeRhs<floating>(N, M, T, phi, varphi, up_exact, u_zero, rhs_function, grid, rhs);

    const int block_dim = M + 1;
    NonEquidistantBlock_1D<floating> C(block_dim, B, D, MM, grid, alpha, dt, processingUnit);

    BlockVector<floating> CC = C.solve(rhs, maxNumberOfIterations, stepsPerIteration, accuracy, solvingProcedure);

    auto B_row = *colMatrixFactory.createColumn(N+3);
    for (int i = 0; i < N+3; i++)  B_row[i] = MM(N+1, i);
    B_row.moveTo(processingUnit);

    auto xx = B_row * CC;

    std::vector<floating> ue(M+1);
    ue = exactSolution(M, T, alpha, grid);
    auto ue_device = *colMatrixFactory.createColumn(ue.size());
    memcpy(ue_device.data(), ue.data(), ue.size()*sizeof(floating));
    ue_device.moveTo(processingUnit);

    floating error_max = std::abs((ue_device-xx).getMaximum());
    std::cout << std::endl << "max norm of absolute error = " << error_max << std::endl << std::endl;
    return error_max;
}

template<class floating>
void initializeMatricesEquidistant(const int N, const floating T,
                                   AlgebraicMatrix<floating> &B, AlgebraicMatrix<floating> &MM)
{
    initializeMatricesNonEquidistant(N, T, B, MM);
    return;
}

template<class floating>
void initializeMatricesNonEquidistant(const int N, const floating T,
                                      AlgebraicMatrix<floating> &B, AlgebraicMatrix<floating> &MM)
{
    floating dt = T / static_cast<floating>(N);

    // constructing Matrix MM
    applyTriDiagonals<floating>(1 / static_cast<floating>(6), 4 / static_cast<floating>(6), MM);

    MM(0,0) = -1 / (2 * dt);
    MM(0,1) = 0.0;
    MM(0,2) = 1 / (2 * dt);
    MM(N+2, N) = -1 / (2 * dt);
    MM(N+2, N+1) = 0.0;
    MM(N+2, N+2) = 1 / (2 * dt);

    // constructing Matrix B
    B = MM;

    for (int i = 0; i < 2 ; i++ ) for (int j = 0; j < 3; j++ ) B(i,j) = 0.0;
    for (int j = N; j < N + 3; j++ )B(N+2, j) = 0.0;
}

template<class floating>
void initializeRhs(const int N, const int M, const floating T,
                   const SpaceTimeFunction<floating> &phi, const SpaceTimeFunction<floating> &varphi,
                   const SpaceTimeFunction<floating> &up_exact, const SpaceFunction<floating> &u_zero,
                   const SpaceTimeFunction<floating> &rhs_function,
                   const AlgebraicVector<floating> &grid, AlgebraicMatrix<floating> &rhs)
{
    floating spacePoint = 0;
    rhs_helper<floating>(spacePoint, T, up_exact, u_zero, phi, rhs[0]);

    spacePoint += grid[0];

    std::vector<floating> gg(N+3);
    for (int i = 1; i < M; i++)
    {
        rhs_helper<floating>(spacePoint, T, up_exact, u_zero, rhs_function, rhs[i]);
        spacePoint += grid[i];
    }

    rhs_helper<floating>(spacePoint, T, up_exact, u_zero, varphi, rhs[M]);

    return;
}

template<class floating>
std::vector<floating> exactSolution(const int M, const floating T, const floating alpha,
                                    const AlgebraicVector<floating> &grid)
{
    std::vector<floating> solution(M+1);

    floating spacePoint = 0;
    for (int i = 0; i < M; i++)
    {
        solution.at(i) = u_exact_f(spacePoint, T, alpha);
        spacePoint += grid[i];
    }
    solution.at(M) = u_exact_f(spacePoint, T, alpha);

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