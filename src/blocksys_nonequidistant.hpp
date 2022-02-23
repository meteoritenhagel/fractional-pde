#ifndef CPU_ONLY
#include "gpukernels.cuh"
#endif

template<class floating>
NonEquidistantBlock1D<floating>::NonEquidistantBlock1D(const SizeType block_dim,
                                                       const AlgebraicMatrix<floating> &B,
                                                       const CoefficientMatrix<floating> &D,
                                                       const AlgebraicMatrix<floating> &M,
                                                       const AlgebraicVector<floating> &grid, const floating alpha,
                                                       const floating time_grid_step_size,
                                                       const ProcessingUnit<floating> processing_unit)
        : NonEquidistantBlock1D(block_dim, B, D.copyToDense(), M, grid, alpha, time_grid_step_size, processing_unit) {}

template<class floating>
NonEquidistantBlock1D<floating>::NonEquidistantBlock1D(const SizeType block_dim,
                                                       const AlgebraicMatrix<floating> &B, const AlgebraicMatrix<floating> &D,
                                                       const AlgebraicMatrix<floating> &M,
                                                       const AlgebraicVector<floating> &grid, const floating alpha,
                                                       const floating time_grid_step_size,
                                                       const ProcessingUnit<floating> processingUnit)
        : _block_dim(block_dim), _B(B), _D(D), _M(M), _alpha(alpha), _time_grid_step_size(time_grid_step_size), _grid(grid),
          _C(initialize_c()), _container_factory(processingUnit), _coarse_system(std::move(initialize_coarse_system())),
          _vector_buffer(*get_container_factory().createMatrix(get_num_blocks(), 10)),
          _host_h(*ContainerFactory(static_cast<ProcessingUnit<floating>>(std::make_shared<CPU<floating>>())).createColumn(
                  get_num_blocks())),
          _buffer_1(*get_container_factory().createMatrix(get_num_blocks(), get_block_dim())),
          _buffer_2(*get_container_factory().createMatrix(get_num_blocks(), get_block_dim())),
          _buffer_3(*get_container_factory().createMatrix(get_num_blocks(), get_block_dim())),
          _buffer_4(*get_container_factory().createMatrix(get_num_blocks(), get_block_dim())),
          _buffer_5(*get_container_factory().createMatrix(get_num_blocks(), get_block_dim())),
          _buffer_6(*get_container_factory().createMatrix(get_num_blocks(), get_block_dim())),
          _coarse_buffer_1(*get_container_factory().createMatrix(get_num_blocks(), get_coarse_dim())),
          _coarse_buffer_2(*get_container_factory().createMatrix(get_num_blocks(), get_coarse_dim()))
{
    assert(_B.isSquare() && _M.isSquare());
    assert(_B.getNrows() == _D.getNrows());
    assert(_B.getNrows() == _M.getNrows());
    assert(_B.getNcols() == _M.getNcols());

    assert(_grid.size() == _block_dim - 1);

    // Move everything to device:
    _B.moveTo(processingUnit);
    _C.moveTo(processingUnit);
    _D.moveTo(processingUnit);
    _M.moveTo(processingUnit);

    // _space_grid_step_size is used in coefficients only, which must be on the CPU
    _grid.moveTo(std::make_shared<CPU<floating>>());
    _host_h = _grid;
    _grid.moveTo(processingUnit);


    // calculate inverses for multigrid
    _C.getInverse();
    _M.getInverse();
}

template<class floating>
typename NonEquidistantBlock1D<floating>::SizeType NonEquidistantBlock1D<floating>::get_num_blocks() const
{
    return _B.getNrows();
}

template<class floating>
typename NonEquidistantBlock1D<floating>::SizeType NonEquidistantBlock1D<floating>::get_block_dim() const
{
    return _block_dim;
}

template<class floating>
typename NonEquidistantBlock1D<floating>::SizeType NonEquidistantBlock1D<floating>::get_dense_dim() const
{
    return get_block_dim() * get_num_blocks();
}

template<class floating>
ContainerFactory<floating> NonEquidistantBlock1D<floating>::get_container_factory() const
{
    return _container_factory;
}

template<class floating>
BlockVector<floating> NonEquidistantBlock1D<floating>::solve_pde(const BlockVector<floating> &rhs,
                                                                 const size_t max_num_iterations, const size_t steps_per_iteration, const floating accuracy,
                                                                 const SolvingProcedure solving_procedure) const
{
    auto pde_solution = solve(rhs, max_num_iterations, steps_per_iteration, accuracy, solving_procedure);
    const auto M = pde_solution.getNrows();
    const auto N = pde_solution.getNcols();

    for(size_t j = 0; j < N; ++j)
        pde_solution[j] = _M * pde_solution[j];

    return pde_solution;
}

template<class floating>
BlockVector<floating> NonEquidistantBlock1D<floating>::solve(const BlockVector<floating> &rhs, const size_t max_num_iterations, const size_t steps_per_iteration, const floating accuracy, const SolvingProcedure solving_procedure) const
{
    const SizeType N = get_num_blocks();
    const SizeType M = get_block_dim();

    auto rhs_copy(rhs);
    rhs_copy.moveTo(get_processing_unit());

    // Since we work with the symmetrized system, also rhs must be rescaled appropiately
    rescale_rhs(rhs_copy);

    BlockVector<floating> solution = *get_container_factory().createMatrix(N, M);

    auto residual{solution};
    calculate_residual(solution, rhs_copy, residual);

    std::cout << "initial system error: " << residual.getEuclidean() << std::endl;

    solution[0] = _M / rhs_copy[0];
    solution[M-1] = _M / rhs_copy[M-1];

    floating euclideanError = 0;

    size_t totalCounter = 0;

    size_t n = 0;
    do
    {
        switch(solving_procedure)
        {
            case SolvingProcedure::CyclicReduction:
                throw std::runtime_error("EquidistantBlock1D does not support Cyclic Reduction. Abort.");
            case SolvingProcedure::PCBiCGStab:
                euclideanError = PCbiCGStab(rhs_copy, steps_per_iteration, accuracy, solution);
                break;

            case SolvingProcedure::BiCGStab:
                euclideanError = biCGStab(rhs_copy, steps_per_iteration, accuracy, solution);
                break;

            case SolvingProcedure::Jacobi:
                smooth(static_cast<floating>(1.0), rhs_copy, steps_per_iteration, solution);
                calculate_residual(solution, rhs_copy, residual);
                euclideanError = residual.getEuclidean();

                ++totalCounter;
                std::cout << "   " << totalCounter << ": " << euclideanError << std::endl;
                break;

            case SolvingProcedure::PCJacobi:
                multigrid(2, rhs_copy, steps_per_iteration, accuracy, solution);

                calculate_residual(solution, rhs_copy, residual);
                euclideanError = residual.getEuclidean();
                ++totalCounter;
                std::cout << "   " << totalCounter << ": " << euclideanError << std::endl;

                break;
            }

        ++n;
    }
    // check for the two breaking conditions:
    // 1. if max_num_iterations has a feasible value, repeat until max_num_iterations is reached
    // 2. if maxNormResidual has a feasible value, repeat until relative error is smaller than desired accuracy
    while ((max_num_iterations <= 0 || n < max_num_iterations) && (accuracy <= 0 || euclideanError >= accuracy));

    return solution;
}

// private:

template<class floating>
AlgebraicMatrix<floating> NonEquidistantBlock1D<floating>::get_dense_representation() const
{
    const SizeType loc_rows = _B.getNrows();      // #rows per block
    // dimensions of dense matrix
    const SizeType glob_rows = get_dense_dim();

    const auto cpu = std::make_shared<CPU<floating>>();
    auto DD = *ContainerFactory(cpu).createMatrix(glob_rows, glob_rows);

    const auto host_M(_M);
    const auto host_D(_D);
    const auto host_B(_B);

    host_M.moveTo(cpu);
    host_D.moveTo(cpu);
    host_B.moveTo(cpu);

    // Now, copy the data
    for (SizeType j = 0; j < loc_rows; j++)
        for (SizeType i = 0; i < loc_rows; i++)
            DD(i,j) = host_M(i,j);

    for (SizeType k = 1; k < get_block_dim() - 1; ++k)
    {
        const SizeType startIndex = k * loc_rows;
        const SizeType endIndex = (k+1) * loc_rows;

        for (SizeType i = startIndex; i < endIndex; i++)
        {
            const SizeType blockNumber = i / loc_rows - 1;

            // Symmetrised system
            const floating coeff_left =  - 2/_host_h[blockNumber];
            const floating coeff_middle = 2*(_host_h[blockNumber] + _host_h[blockNumber+1])/_host_h[blockNumber]/_host_h[blockNumber+1];
            const floating coeff_right =  - 2/_host_h[blockNumber+1];
            const floating scale_factor = (_host_h[blockNumber] + _host_h[blockNumber+1]);

            // TODO: exchange loops for i and j (column major storage)
            for (SizeType j = startIndex - loc_rows; j < startIndex; j++) DD(i,j) = coeff_left * _B(i - startIndex, j - startIndex + loc_rows);

            // A_i
            for (SizeType j = startIndex; j < endIndex; j++)
            {
                if (i-startIndex == 0 || i-startIndex == 1 || i-startIndex == loc_rows-1)
                {
                    DD(i,j) = scale_factor * host_M(i - startIndex, j - startIndex);
                }
                else
                {
                    DD(i,j) = coeff_middle * host_M(i - startIndex, j - startIndex) - scale_factor * get_system_coeff() * host_D(i - startIndex, j - startIndex);
                }
            }

            for (SizeType j = endIndex; j < endIndex + loc_rows; j++) DD(i,j) = coeff_right * host_B(i - startIndex, j - endIndex);
        }
    }


    for (SizeType j = glob_rows - loc_rows; j < glob_rows; j++)
        for (SizeType i = glob_rows - loc_rows; i < glob_rows; i++)
            DD(i,j) = host_M(i - glob_rows + loc_rows, j - glob_rows + loc_rows);

    DD.moveTo(get_processing_unit());
    return DD;
}

template<class floating>
void NonEquidistantBlock1D<floating>::mult(const BlockVector<floating> &beta, BlockVector<floating> &result) const
{
    const SizeType N = get_num_blocks();
    const SizeType M = beta.getNcols();
    assert(beta.getNrows() == N && M == get_block_dim() && "ERROR: Dimension mismatch.");

    //result[0] <- _M*beta[0];
    get_processing_unit()->xgemv(OperationType::Identical, _M.getNrows(), _M.getNcols(),
                                 static_cast<floating>(1.), _M.data(), _M.getNrows(), beta[0].data(), 1,
                                 static_cast<floating>(0.), result[0].data(), 1);

    for (SizeType i = 1; i < get_block_dim() - 1; ++i)
    {
        // Symmetrised system
        const floating coeff_left =  2/_host_h[i-1];
        const floating coeff_middle = 2*(_host_h[i-1] + _host_h[i])/_host_h[i-1]/_host_h[i];
        const floating coeff_right =  2/_host_h[i];
        const floating scale_factor = (_host_h[i-1] + _host_h[i]);

        // WE WANT: scaled -_B * beta[i-1] + _A * beta[i] + -_B * beta[i+1]
        // Note that _A = (c1 * _M - c2 * _D) except for rows 1, 2, N, where it is _M

        // result[i] <- scaled _M * beta[i]
        get_processing_unit()->xgemv(OperationType::Identical, _M.getNrows(), _M.getNcols(),
                                     scale_factor, _M.data(), _M.getNrows(), beta[i].data(), 1,
                                     static_cast<floating>(0), result[i].data(), 1);

        _vector_buffer[0] = result[i];

        // _vector_buffer[0] holds scaled _A * beta[i] except in the rows 1, 2, N (!!! because of this we cannot work without buffer)
        get_processing_unit()->xgemv(OperationType::Identical, _D.getNrows(), _D.getNcols(),
                                   -scale_factor * get_system_coeff(), _D.data(), _D.getNrows(), beta[i].data(), 1,
                                   coeff_middle/scale_factor, _vector_buffer[0].data(), 1);


        // result[i] holds scaled _A * beta[i]
        get_processing_unit()->getMemoryManager()->copy(result[i].data() + 2, _vector_buffer[0].data() + 2, (N - 3) * sizeof(floating));


        // _vector_buffer[1] holds scaled _B * (beta[i-1] + beta[i+1])
        _vector_buffer[1] = beta[i - 1];
        get_processing_unit()->xaxpy(_vector_buffer[1].size(), coeff_right / coeff_left, beta[i + 1].data(), 1, _vector_buffer[1].data(), 1);

        // result[i] holds scaled -_B * beta[i-1] + _A * beta[i] + -_B * beta[i+1], which is what we want
        get_processing_unit()->xgemv(OperationType::Identical, _B.getNrows(), _B.getNcols(),
                                     -coeff_left, _B.data(), _B.getNrows(), _vector_buffer[1].data(), 1,
                                     static_cast<floating>(1.), result[i].data(), 1);
    }

    //result[M-1] <- _M*beta[M-1];
    get_processing_unit()->xgemv(OperationType::Identical, _M.getNrows(), _M.getNcols(),
                                 static_cast<floating>(1.), _M.data(), _M.getNrows(), beta[M - 1].data(), 1,
                                 static_cast<floating>(0.), result[M-1].data(), 1);
}

template<class floating>
floating NonEquidistantBlock1D<floating>::biCGStab(const BlockVector<floating> &rhs, const size_t max_num_iterations, const floating accuracy, BlockVector<floating> &solution) const
{
    const auto numberOfIterations = std::min(get_dense_dim(), static_cast<SizeType>(max_num_iterations));


    auto residual{solution};
    calculate_residual(solution, rhs, residual);

    const auto residual_0 = residual;

    floating rho = 1;
    floating alpha = 1;
    floating omega = 1;

    auto v = *rhs.get_container_factory().createMatrix(rhs.getNrows(), rhs.getNcols());
    auto p = v;

    // reserve space for calculation below
    auto s = v;
    auto t = s;
    auto solutionBuffer = solution;


    floating euclideanError = residual.getEuclidean();

    static int totalCounter = 1;

    bool iterate = true;
    SizeType counter = 0;
    do
    {
        const auto rho_new = scalarProduct(residual_0, residual);
        const auto beta = (rho_new/rho)*(alpha/omega);
        rho = rho_new;
        p = residual + beta*(p - omega * v);
        this->mult(p, v);
        alpha = rho/scalarProduct(residual_0, v);
        s = residual - alpha*v;
        this->mult(s, t);
        omega = scalarProduct(t,s)/scalarProduct(t,t);

        solutionBuffer = solution + omega*s + alpha*p;
        residual = s - omega*t;

        auto currentError = residual.getEuclidean();

        if (std::isnan(currentError))
        {
            iterate = false;
        }
        else
        {
            solution = solutionBuffer;
            euclideanError = residual.getEuclidean();
        }

        if (currentError < accuracy)
        {
            iterate = false;
        }

        //statusBar.draw(counter, numberOfIterations, currentError , 0, 10);
        ++counter;

        std::cout << "   " << totalCounter << " " << euclideanError << std::endl;
        ++totalCounter;


    }
    while (iterate && counter < numberOfIterations);

    return euclideanError;
}

template<class floating>
floating NonEquidistantBlock1D<floating>::PCbiCGStab(const BlockVector<floating> &rhs, const size_t max_num_iterations, const floating accuracy, BlockVector<floating> &solution) const
{

    const unsigned numberOfSmoothingSteps = 2;
    const unsigned numberOfMultiGridSteps = 20; //3000; //200;

    const auto numberOfIterations = std::min(get_dense_dim(), static_cast<SizeType>(max_num_iterations));

    auto &residual = _buffer_6;
    calculate_residual(solution, rhs, residual);

    const auto residual_0 = residual;

    floating rho = 1;
    floating alpha = 1;
    floating omega = 1;

    auto v = *rhs.get_container_factory().createMatrix(rhs.getNrows(), rhs.getNcols());
    auto p = v;

    auto s = v;
    auto t = v;
    auto z = v;
    auto y = v;

    auto solutionBuffer = y;
    floating euclideanError = residual.getEuclidean();

    static int totalCounter = 1;

    bool iterate = true;
    SizeType counter = 0;
    do
    {
        const auto rho_new = scalarProduct(residual_0, residual);
        const auto beta = (rho_new/rho)*(alpha/omega);
        rho = rho_new;

        //p = residual + beta*(p - omega * v);
        get_processing_unit()->xaxpy(p.getNelems(), -omega, v.data(), 1, p.data(), 1);
        p.scale(beta);
        get_processing_unit()->xaxpy(p.getNelems(), static_cast<floating>(1.0), residual.data(), 1, p.data(), 1);
        y.add(y, static_cast<floating>(-1.0)); // y <- 0
        multigrid(numberOfSmoothingSteps, p, numberOfMultiGridSteps, accuracy, y);
        this->mult(y, v);
        alpha = rho/scalarProduct(residual_0, v);

        //s = residual - alpha*v;
        s = residual;
        get_processing_unit()->xaxpy(s.getNelems(), -alpha, v.data(), 1, s.data(), 1);
        z.add(z, static_cast<floating>(-1.0)); // z <- 0
        multigrid(numberOfSmoothingSteps, s, numberOfMultiGridSteps, accuracy, z);
        this->mult(z, t);

        omega = scalarProduct(t, s)/scalarProduct(t, t);

        //solution = solution + omega*z + alpha*y;
        get_processing_unit()->xaxpy(solution.getNelems(), omega, z.data(), 1, solution.data(), 1);
        get_processing_unit()->xaxpy(solution.getNelems(), alpha, y.data(), 1, solution.data(), 1);

        //residual = s - omega*t;
        residual = s;
        get_processing_unit()->xaxpy(residual.getNelems(), -omega, t.data(), 1, residual.data(), 1);

        auto currentError = residual.getEuclidean();

        if (std::isnan(currentError))
        {
            iterate = false;
        }
        else
        {
            if (currentError < accuracy) {
                iterate = false;
            }
            euclideanError = currentError;

        }

        ++counter;

        std::cout << "   " << totalCounter << " " << euclideanError << std::endl;

        ++totalCounter;
    }
    while (iterate && counter < numberOfIterations);

    return euclideanError;
}


template<class floating>
void NonEquidistantBlock1D<floating>::multigrid(const unsigned num_smoothing_steps, const BlockVector<floating> &rhs, const size_t max_num_iterations, const floating accuracy, BlockVector<floating> &solution) const
{
    const SizeType N = get_num_blocks();
    const SizeType M = get_block_dim();

    BlockVector<floating> &coarseSolution = _coarse_buffer_1;
    initializeMemory(get_processing_unit()->getMemoryManager(), coarseSolution.data(), coarseSolution.getNelems(), static_cast<floating>(0.0));

    BlockVector<floating> &ffOnCoarseGrid = _coarse_buffer_2;
    initializeMemory(get_processing_unit()->getMemoryManager(), ffOnCoarseGrid.data(), ffOnCoarseGrid.getNelems(), static_cast<floating>(0.0));

    BlockVector<floating> &residual = _buffer_1;
    initializeMemory(get_processing_unit()->getMemoryManager(), residual.data(), residual.getNelems(), static_cast<floating>(0.0));

    const auto &coarseSystem = get_coarse_system();

    assert(solution.getNrows() == N && solution.getNcols() == M && "ERROR: Dimension mismatch.");

    floating euclideanError = 1;

    const floating omega = 2/3.0;

    if (M == 3)
    {
        // exact solution on the coarsest grid with 3 points
        smooth(1.0, rhs, max_num_iterations, solution);
    }
    else
    {

        smooth(omega, rhs, num_smoothing_steps, solution);
        calculate_residual(solution, rhs, residual);
        restriction(residual, ffOnCoarseGrid);
        coarseSystem.multigrid(num_smoothing_steps, ffOnCoarseGrid, max_num_iterations, accuracy, coarseSolution);
        prolongation(coarseSolution, solution);
        smooth(omega, rhs, num_smoothing_steps, solution);
    }
}

template<class floating>
void NonEquidistantBlock1D<floating>::prolongation(const BlockVector<floating> &coarse_rhs, BlockVector<floating> &fine_rhs) const
{
    const SizeType N = get_num_blocks();
    const SizeType M = get_block_dim();

    assert(coarse_rhs.getNrows() == N && coarse_rhs.getNcols() == (M + 1) / 2 && "ERROR: Wrong dimensions.");

    if (typeid(*this->get_processing_unit()) == typeid(*std::make_shared<CPU<floating>>()))
    {
        fine_rhs[0] += coarse_rhs[0];
#pragma omp parallel for  if (M>127)               // GH
        for (SizeType i = 1; i < get_block_dim() - 1; i += 2)
        {
            const auto coeff1 = _grid[i] / (_grid[i - 1] + _grid[i]);
            const auto coeff2 = _grid[i] / (_grid[i - 1] + _grid[i]);
            fine_rhs[i + 1] += coarse_rhs[(i + 1) / 2];
            //fine_rhs[i] += coeff1 * coarse_rhs[(i - 1) / 2];
            get_processing_unit()->xaxpy(N, coeff1, coarse_rhs[(i - 1) / 2].data(), 1, fine_rhs[i].data(), 1);
            //fine_rhs[i] += coeff2 * coarse_rhs[(i + 1) / 2];
            get_processing_unit()->xaxpy(N, coeff2, coarse_rhs[(i + 1) / 2].data(), 1, fine_rhs[i].data(), 1);
        }
    }
    else
    {
#ifndef CPU_ONLY
        device_prolongation(N, M, _grid.data(), coarse_rhs.data(), fine_rhs.data());
#endif
    }
}

template<class floating>
void NonEquidistantBlock1D<floating>::restriction(const BlockVector<floating> &fine_rhs, BlockVector<floating> &coarse_rhs) const
{
    const SizeType N = get_num_blocks();
    const SizeType M = get_block_dim();
    assert(get_block_dim() % 2 == 1 && "ERROR: Need odd number of rows in each step.");

    if (typeid(*this->get_processing_unit()) == typeid(*std::make_shared<CPU<floating>>()))
    {
#pragma omp parallel for  if (M>127)
        for (SizeType j = 2; j < M - 1; j += 2)
        {
            const auto coeff1 = _grid[j - 2] / (_grid[j - 2] + _grid[j - 1]);
            const auto coeff2 = _grid[j + 1] / (_grid[j] + _grid[j + 1]);
            coarse_rhs[j / 2] = fine_rhs[j];
            get_processing_unit()->xaxpy(get_num_blocks(), coeff1, fine_rhs[j - 1].data(), 1, coarse_rhs[j / 2].data(), 1);
            get_processing_unit()->xaxpy(get_num_blocks(), coeff2, fine_rhs[j + 1].data(), 1, coarse_rhs[j / 2].data(), 1);
        }
    }
    else
    {
#ifndef CPU_ONLY
        device_restriction(N, M, _grid.data(), fine_rhs.data(), coarse_rhs.data());
#endif
    }
}

template<class floating>
AlgebraicVector<floating> NonEquidistantBlock1D<floating>::get_reduced_grid() const
{
    assert(_grid.size() % 2 == 0 && "ERROR: Dimension mismatch. Grid size has to be even number.");

    auto coarseGrid = *ContainerFactory<floating>(std::make_shared<CPU<floating>>()).createColumn(_grid.size() / 2);

    SizeType const Mf = coarseGrid.size();                // GH

    if (typeid(*this->get_processing_unit()) == typeid(*std::make_shared<CPU<floating>>()))
    {
#pragma omp parallel for  //if (Mf >127)
        for (SizeType i = 0; i < Mf; ++i)
            coarseGrid[i] = _grid[2 * i] + _grid[2 * i + 1];
    }
    else
    {
#ifndef CPU_ONLY
        // Since this funcion is called during initialization, it is not yet guaranteed that
        // _space_grid_step_size is already on the right processing unit.
        auto h_copy(_grid);
        h_copy.moveTo(get_processing_unit());
        coarseGrid.moveTo(get_processing_unit());
        device_get_reduced_grid(Mf, h_copy.data(), coarseGrid.data());
#endif
    }

    return coarseGrid;
}

template<class floating>
typename NonEquidistantBlock1D<floating>::SizeType NonEquidistantBlock1D<floating>::get_coarse_dim() const
{
    return (get_block_dim() + 1) / 2;
}

template<class floating>
std::unique_ptr<NonEquidistantBlock1D<floating>> NonEquidistantBlock1D<floating>::initialize_coarse_system() const
{
    if (get_block_dim() <= 3)
        return nullptr;
    else
        return std::make_unique<NonEquidistantBlock1D<floating>>(get_coarse_dim(), _B, _D, _M, get_reduced_grid(),
                                                                 _alpha, _time_grid_step_size, get_processing_unit());
}

template<class floating>
const NonEquidistantBlock1D<floating>& NonEquidistantBlock1D<floating>::get_coarse_system() const
{
    return *_coarse_system;
}

template<class floating>
void NonEquidistantBlock1D<floating>::smooth(const floating omega, const BlockVector<floating> &rhs, const size_t max_num_iterations, BlockVector<floating> &solution) const {
    const SizeType N = get_num_blocks();
    const SizeType M = rhs.getNcols();

    assert(solution.getNrows() == N && solution.getNcols() == M && "ERROR: Dimension mismatch.");

    size_t n = 0;
    // Since these two lines are invariants of the algortithm, we assume that they already
    // have been computed!
    //solution[0] = _M / rhs_f[0];
    //solution[M - 1] = _M / rhs_f[M-1];

    // #################### GAUSS SEIDEL ##################
#ifdef GAUSS_SEIDEL
    auto &residual = _vector_buffer[4];
    auto &outputBuffer = _vector_buffer[5];

    do {

// Gauss-Seidel cannot be parallelized
        for (unsigned int i = 1; i < M - 1; ++i) {
            calculateRowResidual(i, solution, rhs_f, residual);

            // scale residual: omega * (D * C)^-1 * r_i = omega * C^-1 * (D^-1 * r_i)
            if (typeid(*this->getProcessingUnit()) == typeid(*std::make_shared<CPU<floating>>()))
            {
                // Symmetrised system
                const floating inv_scale_factor = 1/(_space_grid_step_size[i-1] + _space_grid_step_size[i]);
                const floating inv_coeff_middle = 1/(2*(_space_grid_step_size[i-1] + _space_grid_step_size[i])/_space_grid_step_size[i-1]/_space_grid_step_size[i]);
                getProcessingUnit()->xscal(N, inv_scale_factor, residual.data(), 1);
                getProcessingUnit()->xscal(N-3, inv_coeff_middle/inv_scale_factor, residual.data()+2, 1);
            }
            else
            {
#ifndef CPU_ONLY
                device_smooth_scale(N, i, _space_grid_step_size.data(), residual.data());
#endif
            }

            //solution[i] += omega*(_C/residual);
            _C.invTimes(residual, outputBuffer);
            get_processing_unit()->xaxpy(N, omega, outputBuffer.data(), 1, solution[i].data(), 1);
        }

        ++n;
    }
        // check for the breaking conditions:
        // repeat until max_num_iterations is reached
    while (n < max_num_iterations);

#else // ######################### JACOBI ###########################
    auto &fullResidual = _buffer_2;

    do {
        calculate_residual(solution, rhs, fullResidual);
        // scale residual: omega * (D * C)^-1 * r_i = omega * C^-1 * (D^-1 * r_i)


        if (typeid(*this->get_processing_unit()) == typeid(*std::make_shared<CPU<floating>>()))
        {
#pragma omp parallel for if (M>63) //schedule(dynamic)
            for (unsigned int j = 1; j < M - 1; ++j) {
                // Symmetrised system
                const floating inv_scale_factor = 1 / (_grid[j - 1] + _grid[j]);
                const floating inv_coeff_middle = 1 / (2 * (_grid[j - 1] + _grid[j]) / _grid[j - 1] / _grid[j]);
                get_processing_unit()->xscal(N, inv_scale_factor, fullResidual[j].data(), 1);
                get_processing_unit()->xscal(N - 3, inv_coeff_middle / inv_scale_factor, fullResidual[j].data() + 2, 1);
            }
        }
        else
        {
#ifndef CPU_ONLY
            device_smooth_full_scale(N, M, _grid.data(), fullResidual.data());
#endif
        }

#ifndef PLU
        const auto Cinv = _C.accessInverse().data();
        // solution *= omega * C^-1 * residual
        get_processing_unit()->xgemm(OperationType::Identical, OperationType::Identical,
                                     N, M-2, N,
                                     omega, Cinv, N,
                                     fullResidual[1].data(), N,
                                     static_cast<floating>(1.0), solution[1].data(), N);
#else
        int info(0);
        const int NN = N;
        const int NRHS = M-2;
        getProcessingUnit()->xgetrs(OperationType::Identical, &NN, &NRHS, _C.accessInverse().data(), &NN, _C._ipiv.data(), fullResidual[1].data(), &NN, &info);
        get_processing_unit()->xaxpy(NN * NRHS, omega, fullResidual[1].data(), 1, solution[1].data(), 1);
#endif
        ++n;
    }
        // check for the breaking conditions:
        // repeat until max_num_iterations is reached
    while (n < max_num_iterations);
#endif
}

template<class floating>
void NonEquidistantBlock1D<floating>::calculate_residual(const BlockVector<floating> &beta, const BlockVector<floating> &rhs, BlockVector<floating> &residual) const
{
    const SizeType N = get_num_blocks();
    const SizeType M = rhs.getNcols();

    assert(beta.getNrows() == N && beta.getNcols() == M && "ERROR: Dimension mismatch.");

#pragma omp parallel for  //if (M>127)
    for (unsigned int i = 0; i < M; ++i)
    {
        calculate_row_residual(i, beta, rhs, residual[i]);
    }
}

template<class floating>
void NonEquidistantBlock1D<floating>::calculate_row_residual(const SizeType i, const BlockVector<floating> &beta_i, const BlockVector<floating> &rhs, AlgebraicVector<floating> &residual) const
{
    const SizeType N = get_num_blocks();
    const SizeType M = rhs.getNcols();

    assert(beta_i.getNrows() == N && beta_i.getNcols() == M && "ERROR: Dimension mismatch.");

    get_processing_unit()->xcopy(residual.size(), rhs[i].data(), 1, residual.data(), 1);

    if (i == 0 || i == M-1)
    {
        get_processing_unit()->xgemv(OperationType::Identical, _M.getNrows(), _M.getNcols(), -1.0, _M.data(), _M.getNrows(), beta_i[i].data(), 1, 1.0, residual.data(), 1);
    }
    else
    {
        // Symmetrised system
        const floating coeff_left =  2/_host_h[i-1];
        const floating coeff_middle = 2*(_host_h[i-1] + _host_h[i])/_host_h[i-1]/_host_h[i];
        const floating coeff_right =  2/_host_h[i];
        const floating scale_factor = (_host_h[i-1] + _host_h[i]);
        // WE WANT: rhs_f[i] - (scaled -_B * beta_i[i-1] + _A * beta_i[i] + -_B * beta_i[i+1])
        // Note that _A = (c1 * _M - c2 * _D) except for rows 1, 2, N, where it is _M

        // result[i] <- scaled _M * beta_i[i]

        auto &buf1 = _buffer_4;
        auto &buf2 = _buffer_5;
        auto &buf3 = _buffer_3;

        auto &b1 = buf1[i];
        auto &b2 = buf2[i];
        auto &b3 = buf3[i];

        get_processing_unit()->xgemv(OperationType::Identical, _M.getNrows(), _M.getNcols(),
                                     scale_factor, _M.data(), _M.getNrows(), beta_i[i].data(), 1,
                                     static_cast<floating>(0), b3.data(), 1);

        //b1 = b3;
        get_processing_unit()->xcopy(b1.size(), b3.data(), 1, b1.data(), 1);

        // _b1 holds scaled _A * beta_i[i] except in the rows 1, 2, N (!!! because of this we cannot work without buffer)
        get_processing_unit()->xgemv(OperationType::Identical, _D.getNrows(), _D.getNcols(),
                                   -scale_factor * get_system_coeff(), _D.data(), _D.getNrows(), beta_i[i].data(), 1,
                                   coeff_middle/scale_factor, b1.data(), 1);


        // b3 holds scaled _A * beta_i[i]
        //get_processing_unit()->getMemoryManager()->copy(b3.data()+2, b1.data()+2, (N-3)*sizeof(floating));
        get_processing_unit()->xcopy(N - 3, b1.data() + 2, 1, b3.data() + 2, 1);


        // b2 holds scaled _B * (beta_i[i-1] + beta_i[i+1])
        // b2 = beta_i[i-1];
        get_processing_unit()->xcopy(b2.size(), beta_i[i - 1].data(), 1, b2.data(), 1);
        get_processing_unit()->xaxpy(b2.size(), coeff_right / coeff_left, beta_i[i + 1].data(), 1, b2.data(), 1);

        // b3 holds scaled -_B * beta_i[i-1] + _A * beta_i[i] + -_B * beta_i[i+1], which is what we want
        get_processing_unit()->xgemv(OperationType::Identical, _B.getNrows(), _B.getNcols(),
                                     -coeff_left, _B.data(), _B.getNrows(), b2.data(), 1,
                                     static_cast<floating>(1.), b3.data(), 1);

        // residual holds what we want
        residual.add(b3, -1.0);
    }
}

template<class floating>
floating NonEquidistantBlock1D<floating>::get_system_coeff() const
{
    floating coeff = pow(_time_grid_step_size, -_alpha) / tgamma(4 - _alpha);
    return coeff;
}

template<class floating>
AlgebraicMatrix<floating> NonEquidistantBlock1D<floating>::initialize_c() const
{
    const SizeType N = get_num_blocks();

    floating max_h = get_grid_preconditioner(_grid);

    auto C = _M - max_h * max_h / 2 * get_system_coeff() * _D;

    const SizeType rowIndices[3] = {0, 1, N-1};

#pragma omp parallel for  if (N>127)
    for (SizeType col = 0; col < N; ++col)
    {
        for (SizeType row  : rowIndices)
        {
            C(row, col) = _M(row, col);
        }
    }

    return C;
}

template<class floating>
ProcessingUnit<floating> NonEquidistantBlock1D<floating>::get_processing_unit() const
{
    return _container_factory.getProcessingUnit();
}

template<class floating>
floating NonEquidistantBlock1D<floating>::get_grid_preconditioner(const AlgebraicVector<floating> &grid) const
{
    return grid.getMaximum();
}

template<class floating>
void NonEquidistantBlock1D<floating>::rescale_rhs(BlockVector<floating> &rhs) const
{
    const SizeType N = get_num_blocks();
    const SizeType M = get_block_dim();
    //assert(rhs.getNrows() == N && rhs.getNcols() == M && "ERROR: Dimension mismatch.");

    if (typeid(*this->get_processing_unit()) == typeid(*std::make_shared<CPU<floating>>()))
    {
#pragma omp parallel for  if (M>127)
        for (SizeType j = 1; j < M-1; ++j)
        {
            rhs[j].scale(_grid[j] + _grid[j - 1]);
        }
    }
    else
    {
#ifndef CPU_ONLY
        device_rescale_rhs(N, M, _grid.data(), rhs.data());
#endif
    }
}
