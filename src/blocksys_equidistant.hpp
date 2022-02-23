template<class floating>
EquidistantBlock1D<floating>::EquidistantBlock1D(const SizeType block_dim, const AlgebraicMatrix <floating> &B,
                                                 const AlgebraicMatrix <floating> &D,
                                                 const AlgebraicMatrix <floating> &M, const floating space_grid_step_size,
                                                 const floating alpha,
                                                 const floating time_grid_step_size,
                                                 const ProcessingUnit <floating> processing_unit)
        : EquidistantBlock1D(block_dim, B, D, M, space_grid_step_size, alpha, time_grid_step_size,
                             initialize_a(B, D, M, space_grid_step_size, alpha, time_grid_step_size), processing_unit) {}

template<class floating>
EquidistantBlock1D<floating>::EquidistantBlock1D(const SizeType block_dim, const AlgebraicMatrix <floating> &B,
                                                 const CoefficientMatrix <floating> &D,
                                                 const AlgebraicMatrix <floating> &M, const floating space_grid_step_size,
                                                 const floating alpha,
                                                 const floating time_grid_step_size,
                                                 const ProcessingUnit <floating> processing_unit)
        : EquidistantBlock1D(block_dim, B, D.copyToDense(), M, space_grid_step_size, alpha, time_grid_step_size,
                             initialize_a(B, D.copyToDense(), M, space_grid_step_size, alpha,
                                          time_grid_step_size), processing_unit) {}

template<class floating>
typename EquidistantBlock1D<floating>::SizeType EquidistantBlock1D<floating>::get_num_blocks() const
{
    return _B.getNrows();
}

template<class floating>
typename EquidistantBlock1D<floating>::SizeType EquidistantBlock1D<floating>::get_block_dim() const
{
    return _block_dim;
}

template<class floating>
typename EquidistantBlock1D<floating>::SizeType EquidistantBlock1D<floating>::get_dense_dim() const
{
    return get_block_dim() * get_num_blocks();
}

template<class floating>
ContainerFactory<floating> EquidistantBlock1D<floating>::get_container_factory() const
{
    return _container_factory;
}

template<class floating>
AlgebraicMatrix<floating> EquidistantBlock1D<floating>::get_dense_representation() const
{
    const SizeType loc_rows = _B.getNrows();
    const SizeType glob_rows = get_dense_dim();

    auto cpu = std::make_shared<CPU<floating>>();

    auto host_A(_A);
    auto host_B(_B);
    auto host_M(_M);

    host_A.moveTo(cpu);
    host_B.moveTo(cpu);
    host_M.moveTo(cpu);

    auto DD = *host_M.get_container_factory().createMatrix(glob_rows, glob_rows);

    const floating scale_factor = 1.0;
    const floating scale_b = 1.0;

    // Now, copy the data
    for (SizeType j = 0; j < loc_rows; j++)
        for (SizeType i = 0; i < loc_rows; i++)
            DD(i,j) = host_M(i,j);

    // TODO: exchange loops for i and j (column major storage) ?
    for (SizeType k = 1; k < get_block_dim() - 1; ++k)
    {
        const SizeType startIndex = k * loc_rows;
        const SizeType endIndex = (k+1) * loc_rows;
        for (SizeType i = startIndex; i < endIndex; i++)
        {
            for (SizeType j = startIndex - loc_rows; j < startIndex; j++)
            {
                DD(i,j) = - (scale_factor*scale_b)*host_B(i - startIndex, j - startIndex + loc_rows);
            }
            for (SizeType j = startIndex; j < endIndex; j++)
            {
                DD(i,j) = scale_factor*host_A(i - startIndex, j - startIndex);
            }
            for (SizeType j = endIndex; j < endIndex + loc_rows; j++)
            {
                DD(i,j) = - (scale_factor*scale_b)*host_B(i - startIndex, j - endIndex);
            }
        }
    }

    for (SizeType j = glob_rows - loc_rows; j < glob_rows; j++)
        for (SizeType i = glob_rows - loc_rows; i < glob_rows; i++)
            DD(i,j) = host_M(i - glob_rows + loc_rows, j - glob_rows + loc_rows);

    DD.moveTo(get_processing_unit());
    return DD;
}

template<class floating>
BlockVector<floating> EquidistantBlock1D<floating>::solve_pde(const BlockVector<floating> &rhs) const
{
    return solve_pde(rhs, 0.0, 0.0, SolvingProcedure::CyclicReduction);
}

template<class floating>
BlockVector<floating> EquidistantBlock1D<floating>::solve_pde(const BlockVector<floating> &rhs,
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
BlockVector<floating> EquidistantBlock1D<floating>::solve(const BlockVector<floating> &rhs) const
{
    return solve(rhs, 0, 0, 0.0, SolvingProcedure::CyclicReduction);
}

template<class floating>
BlockVector<floating> EquidistantBlock1D<floating>::solve(const BlockVector<floating> &rhs,
                                                          const size_t max_num_iterations, const size_t steps_per_iteration, const floating accuracy,
                                                          const SolvingProcedure solving_procedure) const
{
    const SizeType N = get_num_blocks();
    const SizeType M = get_block_dim();

    auto rhs_copy(rhs);
    rhs_copy.moveTo(get_processing_unit());
    auto rs_rhs = rescale_rhs(rhs_copy);

    BlockVector<floating> solution = *get_container_factory().createMatrix(N, M);
    auto x = calculate_residual(solution, rs_rhs);
    std::cout << "initial system error: " << x.getEuclidean() << std::endl;

    if (solving_procedure == SolvingProcedure::PCJacobi)
    {
        // calculate inverses for multigrid
        _A.getInverse();
        _M.getInverse();
    }

    floating euclideanError = 0;

    size_t n = 0;

    if(solving_procedure == SolvingProcedure::CyclicReduction)
    {
        solution = cyclic_reduction(1 / _space_grid_step_size / _space_grid_step_size * _B, rhs_copy);
    }
    else
    {
        do
        {
            switch(solving_procedure)
            {
                case SolvingProcedure::PCBiCGStab:
                    std::cerr << "EquidistantBlock1D does not support preconditioned BiCGStab. Abort.\n";
                    exit(-1);
                    break;

                case SolvingProcedure::BiCGStab:
                    std::cerr << "EquidistantBlock1D does not support BiCGStab. Abort.\n";
                    exit(-1);
                    break;

                case SolvingProcedure::Jacobi:
                    euclideanError = jacobi_iteration(static_cast<floating>(1.0), rs_rhs, steps_per_iteration, accuracy,
                                                      solution);
                    std::cout << euclideanError << std::endl;
                    break;

                case SolvingProcedure::PCJacobi:
                    euclideanError = multigrid(2, rs_rhs, steps_per_iteration, accuracy, solution);

                    // TODO: REMOVE
                    static size_t totalCounter2 = 0;
                    ++totalCounter2;
                    std::cout << "   " << totalCounter2 << ": " << euclideanError << std::endl;

                    break;

                default:
                    std::cerr << "Unknown SolvingProcedure found. Abort.\n";
                    exit(-1);
            }

            ++n;
        }
        // check for the two breaking conditions:
        // 1. if max_num_iterations has a feasible value, repeat until max_num_iterations is reached
        // 2. if maxNormResidual has a feasible value, repeat until relative error is smaller than desired accuracy
        while ((max_num_iterations <= 0 || n < max_num_iterations) && (accuracy <= 0 || euclideanError >= accuracy));
    }

    return solution;
}

// private:

template<class floating>
EquidistantBlock1D<floating>::EquidistantBlock1D(const SizeType block_dim,
                                                 const AlgebraicMatrix<floating> &B, const AlgebraicMatrix<floating> &D, const AlgebraicMatrix<floating> &M,
                                                 const floating space_grid_step_size, const floating alpha, const floating time_grid_step_size,
                                                 const AlgebraicMatrix<floating> &A, const ProcessingUnit<floating> processing_unit)
        : _block_dim(block_dim), _B(B), _D(D), _M(M), _alpha(alpha), _time_grid_step_size(time_grid_step_size), _space_grid_step_size(space_grid_step_size), _A(A),
          _container_factory(processing_unit), _coarse_system(initialize_coarse_system())
{
    assert(_A.isSquare() && _B.isSquare() && _M.isSquare());
    assert(_A.getNrows() == _B.getNrows());
    assert(_A.getNcols() == _B.getNcols());
    assert(_A.getNrows() == _M.getNrows());
    assert(_A.getNcols() == _M.getNcols());

    _A.moveTo(processing_unit);
    _B.moveTo(processing_unit);
    _D.moveTo(processing_unit);
    _M.moveTo(processing_unit);
}

template<class floating>
floating EquidistantBlock1D<floating>::get_system_coeff(const floating alpha, const floating time_grid_step_size)
{
    floating coeff = pow(time_grid_step_size, -alpha) / tgamma(4 - alpha);
    return coeff;
}

template<class floating>
AlgebraicMatrix<floating> EquidistantBlock1D<floating>::initialize_a(const AlgebraicMatrix<floating> &B, const AlgebraicMatrix<floating> &D, const AlgebraicMatrix<floating> &M, const floating space_grid_step_size, const floating alpha, const floating time_grid_step_size) const
{
    const SizeType N = B.getNrows();
    auto A = static_cast<floating>(2) / space_grid_step_size / space_grid_step_size * M - get_system_coeff(alpha, time_grid_step_size) * D;

    // operator() can only be invoked for CPU objects
    auto Mcopy(M);
    A.moveTo(std::make_shared<CPU<floating>>());
    Mcopy.moveTo(std::make_shared<CPU<floating>>());

    const SizeType colIndices[3] = {0, 1, N-1};
    for (SizeType col  : colIndices)
    {
        for (SizeType row = 0; row < N; ++row)
        {
            A(col, row) = Mcopy(col, row);
        }
    }

    return A;
}

template<class floating>
ProcessingUnit<floating> EquidistantBlock1D<floating>::get_processing_unit() const
{
    return _container_factory.getProcessingUnit();
}

template<class floating>
BlockVector<floating> EquidistantBlock1D<floating>::cyclic_reduction(const AlgebraicMatrix<floating> &scaled_b, BlockVector<floating> const &f) const
{
    std::cout << "   cyclic_reduction:   " << this->get_block_dim() << "  " << this->get_block_dim() << '\n';

    auto returnMatrix = *get_container_factory().createMatrix();

    if (get_block_dim() <= 3)
    {
        EquidistantBlock1D<floating> const K0(3, scaled_b, _D, _M, _space_grid_step_size, _alpha, _time_grid_step_size, _A,
                                              get_processing_unit());
        AlgebraicMatrix<floating> const K = K0.get_dense_representation();

        auto returnMatrix = K / f.flat();
        returnMatrix.resize(f.getNrows(), f.getNcols());

        return returnMatrix;
    }
    else
    {
        SizeType const cnr = (get_block_dim() + 1) / 2;
        BlockVector<floating> const _B0  = scaled_b * (_A / scaled_b);

        const EquidistantBlock1D<floating> Bs(cnr, _B0, _D, _M, _space_grid_step_size, _alpha, _time_grid_step_size,
                                               _A - static_cast<floating>(2) * _B0, get_processing_unit());

        returnMatrix = cr_prolongation(scaled_b, f, Bs.cyclic_reduction(_B0, cr_restriction(scaled_b, f)));
    }

    return returnMatrix;
}

template<class floating>
BlockVector<floating> EquidistantBlock1D<floating>::cr_prolongation(const AlgebraicMatrix<floating> &scaled_b, const BlockVector<floating> &fine_rhs, const BlockVector<floating> &coarse_solution) const
{
    // TODO: write GPU kernel
    const SizeType fnr = get_block_dim();
    const SizeType cnr = (get_block_dim() + 1) / 2;

    assert(fine_rhs.getNrows() == _A.getNcols());
    assert(coarse_solution.getNrows() == _A.getNcols());

    const SizeType nrow = fine_rhs.getNrows();

    BlockVector<floating> uf  = *get_container_factory().createMatrix(nrow, fnr);

    constexpr SizeType BLOCKSIZE = 64;

    uf[0] = coarse_solution[0];

#pragma omp parallel for
    for (SizeType ib = 1; ib < cnr; ib += BLOCKSIZE) {
        BlockVector<floating> bmt = *get_container_factory().createMatrix(nrow, BLOCKSIZE);
        BlockVector<floating> icv = *get_container_factory().createMatrix(nrow, BLOCKSIZE);
        const SizeType iend  = std::min(ib + BLOCKSIZE, cnr);

        SizeType jt = 0;
        for (SizeType i = ib; i < iend; ++i, ++jt)
        {
            bmt[jt] = coarse_solution[i - 1] + coarse_solution[i];
            icv[jt] = fine_rhs[2 * i - 1];
            uf[2 * i] = coarse_solution[i];
        }

        bmt = _A/(scaled_b * bmt + icv);
        jt = 0;
        for (SizeType i = ib; i < iend; ++i, ++jt) {
            uf[2 * i - 1] = bmt[jt];
        }
    }

    return uf;
}
template<class floating>
BlockVector<floating> EquidistantBlock1D<floating>::cr_restriction(const AlgebraicMatrix<floating> &scaled_b, const BlockVector<floating> &fine_rhs) const
{
    const SizeType fnr = get_block_dim();
    assert(fnr % 2 == 1); // odd number of block rows required
    const SizeType cnr = (get_block_dim() + 1) / 2;
    assert(fine_rhs.getNrows() == _A.getNcols());

    const SizeType nrow = fine_rhs.getNrows();
    AlgebraicMatrix <floating> fc = *get_container_factory().createMatrix(nrow, cnr);

    fc[0] = fine_rhs[0];

    constexpr SizeType BLOCKSIZE = 64;

#pragma omp parallel for
    for (SizeType ib = 2; ib < fnr - 1; ib += 2 * BLOCKSIZE) {
        AlgebraicMatrix <floating> bmt = *get_container_factory().createMatrix(nrow, BLOCKSIZE);
        const SizeType iend = std::min(ib + 2 * BLOCKSIZE, fnr - 1);
        SizeType jt = 0;

        for (SizeType i = ib; i < iend; i += 2, ++jt) {
            fc[i / 2] = fine_rhs[i];
            bmt[jt] = fine_rhs[i - 1] + fine_rhs[i + 1];
        }
        bmt = scaled_b * (_A / bmt);

        fc.updateAdd(ib / 2, iend / 2, bmt);
    }

    fc[cnr - 1] = fine_rhs[fnr - 1];
    return fc;
}

template<class floating>
floating EquidistantBlock1D<floating>::multigrid(const unsigned num_smoothing_steps, const BlockVector<floating> &f, const size_t max_num_iterations, const floating accuracy, BlockVector<floating> &solution) const
{
    floating omega = 2/3.0;
    const SizeType N = get_num_blocks();
    const SizeType M = get_block_dim();

    const SizeType coarseM = get_coarse_block_dim();

    BlockVector<floating> coarseSolution = *get_container_factory().createMatrix(N, coarseM);

    const auto &coarseSystem = get_coarse_system();

    assert(solution.getNrows() == N && solution.getNcols() == M && "ERROR: Dimension mismatch.");

    floating euclideanError = 1;

    if (M == 3)
    {
        jacobi_iteration(1.0, f, max_num_iterations, accuracy, solution);
    }
    else
    {

        // Step 1: smoothing
        smooth(omega, f, 2, solution);

        // Step 2:
        auto residual = calculate_residual(solution, f);

        // Step 3:
        auto ffOnCoarseGrid = restriction(residual);

        // Step 4: solve on ffOnCoarseGrid:
        coarseSystem.multigrid(num_smoothing_steps, ffOnCoarseGrid, max_num_iterations, accuracy, coarseSolution);

        // Step 5: correction:
        solution += prolongation(coarseSolution);

        // Step 6: smoothing
        euclideanError = jacobi_iteration(omega, f, num_smoothing_steps, accuracy, solution);
    }

    return euclideanError;
}

template<class floating>
BlockVector<floating> EquidistantBlock1D<floating>::prolongation(const BlockVector<floating> &coarse_rhs) const
{
    const SizeType N = get_num_blocks();
    const SizeType M = get_block_dim();

    assert(coarse_rhs.getNrows() == N && coarse_rhs.getNcols() == (M + 1) / 2 && "ERROR: Wrong dimensions.");

    auto ffOnFineGrid = *get_container_factory().createMatrix(N, M);

    ffOnFineGrid[0] = coarse_rhs[0];
    for (SizeType i = 1; i < get_block_dim() - 1; i += 2)
    {
        ffOnFineGrid[i+1] = coarse_rhs[(i + 1) / 2];
        ffOnFineGrid[i] = static_cast<floating>(1/2.) * (ffOnFineGrid[i-1] + ffOnFineGrid[i+1]);
    }

    return ffOnFineGrid;
}

template<class floating>
BlockVector<floating> EquidistantBlock1D<floating>::restriction(const BlockVector<floating> &fine_rhs) const
{
    const SizeType N = get_num_blocks();
    const SizeType M = get_block_dim();
    assert(get_block_dim() % 2 == 1 && "ERROR: Need odd number of rows in each step.");

    const SizeType coarseGridSize = (get_block_dim() + 1) / 2;

    auto ffOnCoarseGrid = *get_container_factory().createMatrix(N, coarseGridSize);

// Full restriction
#pragma omp parallel for
    for (SizeType i = 2; i < M-1; i+=2)
    {
        ffOnCoarseGrid[i/2] = static_cast<floating>(1/2.) * (fine_rhs[i - 1] + fine_rhs[i + 1]) + fine_rhs[i];
    }
    return ffOnCoarseGrid;
}

template<class floating>
floating EquidistantBlock1D<floating>::get_reduced_grid_step_size() const
{
    return 2. * _space_grid_step_size;
}

template<class floating>
typename EquidistantBlock1D<floating>::SizeType EquidistantBlock1D<floating>::get_coarse_block_dim() const
{
    return (get_block_dim() + 1) / 2;
}

template<class floating>
std::unique_ptr<EquidistantBlock1D<floating>> EquidistantBlock1D<floating>::initialize_coarse_system() const
{
    if (get_block_dim() <= 3)
    {
        return nullptr;
    }
    else
    {
        return std::make_unique<EquidistantBlock1D<floating>>(get_coarse_block_dim(), _B, _D, _M, get_reduced_grid_step_size(), _alpha, _time_grid_step_size,
                                                              get_processing_unit());
    }
}

template<class floating>
const EquidistantBlock1D<floating>& EquidistantBlock1D<floating>::get_coarse_system() const
{
    return *_coarse_system;
}

template<class floating>
floating EquidistantBlock1D<floating>::jacobi_iteration(const floating omega, const BlockVector<floating> &rhs, const size_t max_num_iterations, const floating , BlockVector<floating> &solution) const
{
    const SizeType M = rhs.getNcols();

    const floating inv_scale_factor = 1 / _space_grid_step_size / 2;

    assert(solution.getNrows() == get_num_blocks() && solution.getNcols() == M && "ERROR: Dimension mismatch.");

    floating euclideanNormResidual = 0;

    size_t n = 0;
    do
    {
        euclideanNormResidual = 0;
        auto residual0 = calculate_row_residual(0, solution, rhs);
        euclideanNormResidual += scalarProduct(residual0, residual0);

        solution[0] = _M / rhs[0];

        for (unsigned int i = 1; i < M-1; ++i)
        {
            auto residual = calculate_row_residual(i, solution, rhs);
            euclideanNormResidual += scalarProduct(residual, residual);

            // scale residual: omega * (D * C)^-1 * r_i = omega * C^-1 * (D^-1 * r_i)
            residual = (inv_scale_factor)*residual;
            solution[i] += omega*(_A/residual);
        }

        residual0 = calculate_row_residual(M - 1, solution, rhs);
        euclideanNormResidual += scalarProduct(residual0, residual0);
        solution[M-1] = _M / rhs[M - 1];

        euclideanNormResidual = sqrt(euclideanNormResidual);
        ++n;
    }
        // check for the two breaking conditions:
        // 1. if max_num_iterations has a feasible value, repeat until max_num_iterations is reached
        // 2. if maxNormResidual has a feasible value, repeat until relative error is smaller than desired accuracy
    while ((max_num_iterations <= 0 || n < max_num_iterations) /*&& (accuracy <= 0 || euclideanNormResidual >= accuracy)*/);

    return euclideanNormResidual;
}

template<class floating>
void EquidistantBlock1D<floating>::smooth(const floating omega, const BlockVector<floating> &rhs, const size_t max_num_iterations, BlockVector<floating> &solution) const
{
    const SizeType M = rhs.getNcols();

    const floating inv_scale_factor = 1 / _space_grid_step_size / 2;

    assert(solution.getNrows() == get_num_blocks() && solution.getNcols() == M && "ERROR: Dimension mismatch.");

    size_t n = 0;
    do
    {
        solution[0] = _M / rhs[0];

        for (unsigned int i = 1; i < M-1; ++i) {
            // scale residual: omega * (D * C)^-1 * r_i = omega * C^-1 * (D^-1 * r_i)
            solution[i] += inv_scale_factor * omega * (_A / calculate_row_residual(i, solution, rhs));
        }

        solution[M-1] = _M / rhs[M - 1];
        ++n;
    }
        // check for the breaking condition:
        // repeat until max_num_iterations is reached
    while (n < max_num_iterations);
}


template<class floating>
BlockVector<floating> EquidistantBlock1D<floating>::calculate_residual(const BlockVector<floating> &beta, const BlockVector<floating> &rhs) const
{
    const SizeType N = get_num_blocks();
    const SizeType M = rhs.getNcols();


    assert(beta.getNrows() == N && beta.getNcols() == M && "ERROR: Dimension mismatch.");
    auto residual = *get_container_factory().createMatrix(N, M);  // GH: das dürfte bei Unfied memory langsam sein, diesen Speicher benötigen wir nur auf der GPU.

    for (unsigned int i = 0; i < M; ++i)
    {
        residual[i] = calculate_row_residual(i, beta, rhs);
    }

    return residual;
}

template<class floating>
AlgebraicVector<floating> EquidistantBlock1D<floating>::calculate_row_residual(const SizeType i, const BlockVector<floating> &beta_i, const BlockVector<floating> &rhs) const
{
    const SizeType M = rhs.getNcols();

    auto residual = rhs[i];

    if (i == 0 || i == M-1)
    {
        residual.add(_M * beta_i[i], -1.0);
    }
    else
    {
        const floating coeff_left = 2 / _space_grid_step_size;
        const floating coeff_right = 2 / _space_grid_step_size;
        const floating scale_factor = 2 * _space_grid_step_size;

        residual += (_B*((coeff_left * beta_i[i - 1]).add(beta_i[i + 1], coeff_right))).add(_A * beta_i[i], -scale_factor);
    }

    return residual;
}

template<class floating>
BlockVector<floating> EquidistantBlock1D<floating>::rescale_rhs(const BlockVector<floating> &rhs) const
{
    const SizeType M = get_block_dim();
    auto scaled_rhs = rhs;

    for (SizeType j = 1; j < M-1; ++j)
    {
        scaled_rhs[j] = 2 * _space_grid_step_size * scaled_rhs[j];
    }

    return scaled_rhs;
}
