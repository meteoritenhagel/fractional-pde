#include <cassert>
#include <cmath>      // exp()
#include <algorithm>  // std::max()


template<class floating>
BlockVector<floating> solve_equidistant(const ProcessingUnit<floating> processingUnit,
                                            const int N, const int M, const floating T, const floating alpha,
                                            const PDEFunctionTuple<floating>& pde_function_tuple,
                                            const size_t maxNumberOfIterations, const size_t stepsPerIteration,
                                            const floating accuracy, const SolvingProcedure solvingProcedure)
{
    ProcessingUnit<floating> cpu = std::make_shared<CPU<floating>>();
    ContainerFactory<floating> colMatrixFactory(cpu);

    floating dt = T / static_cast<floating>(N);
    floating dx = T / static_cast<floating>(M);

    const auto grid = *colMatrixFactory.createColumn(M, dx);

    auto B = *colMatrixFactory.createMatrix(N+3, N+3);
    auto D = *colMatrixFactory.createCoefficientMatrix(N, alpha);
    auto MM = *colMatrixFactory.createMatrix(N+3, N+3);

    initialize_matrices(N, T, B, MM);

    auto rhs = *colMatrixFactory.createMatrix(N+3, M+1);

    initialize_rhs<floating>(N, T, pde_function_tuple, grid, rhs);

    const int block_dim = M + 1;
    EquidistantBlock_1D<floating> C(block_dim, B, D, MM, dx, alpha, dt, processingUnit);

    BlockVector<floating> solution = C.solve_pde(rhs, maxNumberOfIterations, stepsPerIteration, accuracy, solvingProcedure);

    return solution;
}

template<class floating>
BlockVector<floating> solve_nonequidistant(const ProcessingUnit<floating> processingUnit,
                                               const int N, const floating T, const floating alpha,
                                               const AlgebraicVector<floating>& grid,
                                               const PDEFunctionTuple<floating>& pde_function_tuple,
                                               const size_t maxNumberOfIterations, const size_t stepsPerIteration,
                                               const floating accuracy, const SolvingProcedure solvingProcedure)
{
    ProcessingUnit<floating> cpu = std::make_shared<CPU<floating>>();
    ContainerFactory<floating> colMatrixFactory(cpu);

    const int M = grid.size();

    floating dt = T / static_cast<floating>(N);

    auto B = *colMatrixFactory.createMatrix(N+3, N+3);
    auto D = *colMatrixFactory.createCoefficientMatrix(N, alpha);
    auto MM = *colMatrixFactory.createMatrix(N+3, N+3);

    initialize_matrices(N, T, B, MM);

    auto rhs = *colMatrixFactory.createMatrix(N+3, M+1);

    initialize_rhs<floating>(N, T, pde_function_tuple, grid, rhs);

    const int block_dim = M + 1;
    NonEquidistantBlock_1D<floating> C(block_dim, B, D, MM, grid, alpha, dt, processingUnit);

    BlockVector<floating> solution = C.solve_pde(rhs, maxNumberOfIterations, stepsPerIteration, accuracy, solvingProcedure);

    return solution;
}


template<class floating>
void initialize_matrices(const int N, const floating T,
                         AlgebraicMatrix<floating> &B, AlgebraicMatrix<floating> &MM)
{
    floating dt = T / static_cast<floating>(N);

    // constructing Matrix MM
    apply_tridiagonals<floating>(1 / static_cast<floating>(6), 4 / static_cast<floating>(6), MM);

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
void initialize_rhs(const int N, const floating T, const PDEFunctionTuple<floating> &pde_function_tuple,
                    const AlgebraicVector<floating> &grid, AlgebraicMatrix<floating> &rhs)
{
    const auto [phi, varphi, up_exact, u_zero, rhs_function] = pde_function_tuple;

    // For use with rhs_helper, phi and varphi must be SpaceTimeFunction
    const auto conv_phi = [phi](SpacePoint<floating> x, TimePoint<floating> t){ return phi(t); };
    const auto conv_varphi = [varphi](SpacePoint<floating> x, TimePoint<floating> t){ return varphi(t); };

    const auto M = grid.size();

    floating spacePoint = 0;
    rhs_helper<floating>(spacePoint, T, up_exact, u_zero, conv_phi, rhs[0]);

    spacePoint += grid[0];

    std::vector<floating> gg(N+3);
    for (int i = 1; i < M; i++)
    {
        rhs_helper<floating>(spacePoint, T, up_exact, u_zero, rhs_function, rhs[i]);
        spacePoint += grid[i];
    }

    rhs_helper<floating>(spacePoint, T, up_exact, u_zero, conv_varphi, rhs[M]);

    return;
}

template<class T, enable_if_is_integral<T>>
static constexpr T twoToThe(T const x)
{
    return 1<<x;
}

template<class floating>
std::vector<floating> linspace(const int N, const floating a, const floating b)
{
    std::vector<floating> x(N);
    floating const h = (b - a) / (N - 1.0);
    for (int k = 0; k < N; ++k) {
        x[k] = k * h + a;
    }
    return x;
}

template<class floating>
void apply_tridiagonals(const floating a, const floating b, AlgebraicMatrix<floating> &B)
{
    assert(B.isSquare());
    for (unsigned int i = 0; i < B.getNcols(); i++)
    {
        B(i,i) = b;
        if (i != B.getNcols() - 1) B(i,i+1) = a;
        if (i != 0) B(i,i-1) = a;
    }

    return;
}


template<class floating>
void rhs_helper(const floating x, const floating T,
                const SpaceTimeFunction<floating> &up_exact, const SpaceFunction<floating> &u_zero,
                const SpaceTimeFunction<floating> &inner_function,
                AlgebraicVector<floating>& rhs_entry)
{
    const int N = rhs_entry.size();
    std::vector<floating> t = linspace<floating>(N - 2, 0, T);

    rhs_entry[0] = up_exact(x, static_cast<floating>(0.0));
    rhs_entry[1] = u_zero(x);
    for (int i = 2; i < N - 1; i++) rhs_entry[i] = inner_function(x, t.at(i - 1));
    rhs_entry[N - 1] = up_exact(x, T);
    return;
}