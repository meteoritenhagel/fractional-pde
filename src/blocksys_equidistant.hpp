template<class floating>
EquidistantBlock_1D<floating>::EquidistantBlock_1D(const SizeType bdim, const AlgebraicMatrix <floating> &B,
                                                   const AlgebraicMatrix <floating> &D,
                                                   const AlgebraicMatrix <floating> &M, const floating h,
                                                   const floating alpha,
                                                   const floating timeGridStepSize,
                                                   const ProcessingUnit <floating> processingUnit)
        : EquidistantBlock_1D(bdim, B, D, M, h, alpha, timeGridStepSize,
                              initializeA(B, D, M, h, alpha, timeGridStepSize), processingUnit) {}

template<class floating>
EquidistantBlock_1D<floating>::EquidistantBlock_1D(const SizeType bdim, const AlgebraicMatrix <floating> &B,
                                                   const CoefficientMatrix <floating> &D,
                                                   const AlgebraicMatrix <floating> &M, const floating h,
                                                   const floating alpha,
                                                   const floating timeGridStepSize,
                                                   const ProcessingUnit <floating> processingUnit)
        : EquidistantBlock_1D(bdim, B, D.copyToDense(), M, h, alpha, timeGridStepSize,
                              initializeA(B, D.copyToDense(), M, h, alpha, timeGridStepSize), processingUnit) {}

template<class floating>
typename EquidistantBlock_1D<floating>::SizeType EquidistantBlock_1D<floating>::getNdim() const
{
    return _B.getNrows();
}

template<class floating>
typename EquidistantBlock_1D<floating>::SizeType EquidistantBlock_1D<floating>::getBlockDim() const
{
    return _bdim;
}

template<class floating>
typename EquidistantBlock_1D<floating>::SizeType EquidistantBlock_1D<floating>::getDenseDim() const
{
    return getBlockDim() * getNdim();
}

template<class floating>
ContainerFactory<floating> EquidistantBlock_1D<floating>::getMatrixFactory() const
{
    return _colMatrixFactory;
}

template<class floating>
AlgebraicMatrix<floating> EquidistantBlock_1D<floating>::copyToDense() const
{
    const SizeType loc_rows = _B.getNrows();
    const SizeType glob_rows = getDenseDim();

    auto cpu = std::make_shared<CPU<floating>>();

    auto host_A(_A);
    auto host_B(_B);
    auto host_M(_M);

    host_A.moveTo(cpu);
    host_B.moveTo(cpu);
    host_M.moveTo(cpu);

    auto DD = *host_M.getMatrixFactory().createMatrix(glob_rows, glob_rows);

    const floating scale_factor = 1.0;
    const floating scale_b = 1.0;

    // Now, copy the data
    for (SizeType j = 0; j < loc_rows; j++)
        for (SizeType i = 0; i < loc_rows; i++)
            DD(i,j) = host_M(i,j);

    // TODO: exchange loops for i and j (column major storage) ?
    for (SizeType k = 1; k < getBlockDim() - 1; ++k)
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

    DD.moveTo(getProcessingUnit());
    return DD;
}

template<class floating>
BlockVector<floating> EquidistantBlock_1D<floating>::solve(const BlockVector<floating> &rhs) const
{
    return solve(rhs, 0, 0, 0.0, SolvingProcedure::CyclicReduction);
}

template<class floating>
BlockVector<floating> EquidistantBlock_1D<floating>::solve(const BlockVector<floating> &rhs,
    const size_t maxNumberOfIterations, const size_t stepsPerIteration, const floating accuracy,
    const SolvingProcedure solvingProcedure) const
{
    const SizeType N = getNdim();
    const SizeType M = getBlockDim();

    auto rhs_copy(rhs);
    rhs_copy.moveTo(getProcessingUnit());
    auto rs_rhs = rescale_rhs(rhs_copy);

    BlockVector<floating> solution = *getMatrixFactory().createMatrix(N, M);
    auto x = calculateResidual(solution, rs_rhs);
    std::cout << "initial system error: " << x.getEuclidean() << std::endl;

    if (solvingProcedure == SolvingProcedure::PCRichardson)
    {
        // calculate inverses for multigrid
        _A.getInverse();
        _M.getInverse();
    }

    floating euclideanError = 0;

    size_t n = 0;

    if(solvingProcedure == SolvingProcedure::CyclicReduction)
    {
        solution = cyclicReduction(1/_h/_h * _B, rhs_copy);
    }
    else
    {
        do
        {
            switch(solvingProcedure)
            {
                case SolvingProcedure::PCBiCGStab:
                    std::cerr << "EquidistantBlock_1D does not support preconditioned BiCGStab. Abort.\n";
                    exit(-1);
                    break;

                case SolvingProcedure::BiCGStab:
                    std::cerr << "EquidistantBlock_1D does not support BiCGStab. Abort.\n";
                    exit(-1);
                    break;

                case SolvingProcedure::Richardson:
                    euclideanError = jacobiIteration(static_cast<floating>(1.0), rs_rhs, stepsPerIteration, accuracy, solution);
                    std::cout << euclideanError << std::endl;
                    break;

                case SolvingProcedure::PCRichardson:
                    euclideanError = multigrid(2, rs_rhs, stepsPerIteration, accuracy, solution);

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
        // 1. if maxNumberOfIterations has a feasible value, repeat until maxNumberOfIterations is reached
        // 2. if maxNormResidual has a feasible value, repeat until relative error is smaller than desired accuracy
        while ((maxNumberOfIterations <= 0 || n < maxNumberOfIterations) && (accuracy <= 0 || euclideanError >= accuracy));
    }

    return solution;
}

// private:

template<class floating>
EquidistantBlock_1D<floating>::EquidistantBlock_1D(const SizeType bdim,
                                                   const AlgebraicMatrix<floating> &B, const AlgebraicMatrix<floating> &D, const AlgebraicMatrix<floating> &M,
                                                   const floating h, const floating alpha, const floating timeGridStepSize,
                                                   const AlgebraicMatrix<floating> &A, const ProcessingUnit<floating> processingUnit)
        : _bdim(bdim), _B(B), _D(D), _M(M), _alpha(alpha), _timeGridStepSize(timeGridStepSize), _h(h), _A(A),
        _colMatrixFactory(processingUnit), _coarseSystemPtr(initializeCoarseSystemPtr())
{
    assert(_A.isSquare() && _B.isSquare() && _M.isSquare());
    assert(_A.getNrows() == _B.getNrows());
    assert(_A.getNcols() == _B.getNcols());
    assert(_A.getNrows() == _M.getNrows());
    assert(_A.getNcols() == _M.getNcols());

    _A.moveTo(processingUnit);
    _B.moveTo(processingUnit);
    _D.moveTo(processingUnit);
    _M.moveTo(processingUnit);
}

template<class floating>
floating EquidistantBlock_1D<floating>::getSystemCoeff(const floating alpha, const floating timeGridStepSize)
{
    floating coeff = pow(timeGridStepSize, -alpha)/tgamma(4-alpha);
    return coeff;
}

template<class floating>
AlgebraicMatrix<floating> EquidistantBlock_1D<floating>::initializeA(const AlgebraicMatrix<floating> &B, const AlgebraicMatrix<floating> &D, const AlgebraicMatrix<floating> &M, const floating h, const floating alpha, const floating timeGridStepSize) const
{
    const SizeType N = B.getNrows();
    auto A = static_cast<floating>(2)/h/h * M - getSystemCoeff(alpha, timeGridStepSize) * D;

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
ProcessingUnit<floating> EquidistantBlock_1D<floating>::getProcessingUnit() const
{
    return _colMatrixFactory.getProcessingUnit();
}

template<class floating>
BlockVector<floating> EquidistantBlock_1D<floating>::cyclicReduction(const AlgebraicMatrix<floating> &scaledB, BlockVector<floating> const &f) const
{
    std::cout << "   cyclicReduction:   " <<  this->getBlockDim() << "  " << this->getBlockDim() << '\n';

    auto returnMatrix = *getMatrixFactory().createMatrix();

    if (getBlockDim() <= 3)
    {
        EquidistantBlock_1D<floating> const K0(3, scaledB, _D, _M, _h, _alpha, _timeGridStepSize, _A, getProcessingUnit());
        AlgebraicMatrix<floating> const K = K0.copyToDense();

        auto returnMatrix = K / f.flat();
        returnMatrix.resize(f.getNrows(), f.getNcols());

        return returnMatrix;
    }
    else
    {
        SizeType const cnr = (getBlockDim() + 1) / 2;
        BlockVector<floating> const _B0  = scaledB * (_A / scaledB);

        const EquidistantBlock_1D<floating> Bs(cnr, _B0, _D, _M, _h, _alpha, _timeGridStepSize,
                                               _A - static_cast<floating>(2) * _B0, getProcessingUnit());

        returnMatrix = CRProlongation(scaledB, f, Bs.cyclicReduction(_B0, CRRestriction(scaledB, f)));
    }

    return returnMatrix;
}

template<class floating>
BlockVector<floating> EquidistantBlock_1D<floating>::CRProlongation(const AlgebraicMatrix<floating> &scaledB, const BlockVector<floating> &ff, const BlockVector<floating> &uc) const
{
    // TODO: write GPU kernel
    const SizeType fnr = getBlockDim();
    const SizeType cnr = (getBlockDim() + 1) / 2;

    assert(ff.getNrows() == _A.getNcols());
    assert(uc.getNrows() == _A.getNcols());

    const SizeType nrow = ff.getNrows();

    BlockVector<floating> uf  = *getMatrixFactory().createMatrix(nrow, fnr);

    constexpr SizeType BLOCKSIZE = 64;

    uf[0] = uc[0];

#pragma omp parallel for
    for (SizeType ib = 1; ib < cnr; ib += BLOCKSIZE) {
        BlockVector<floating> bmt = *getMatrixFactory().createMatrix(nrow, BLOCKSIZE);
        BlockVector<floating> icv = *getMatrixFactory().createMatrix(nrow, BLOCKSIZE);
        const SizeType iend  = std::min(ib + BLOCKSIZE, cnr);

        SizeType jt = 0;
        for (SizeType i = ib; i < iend; ++i, ++jt)
        {
            bmt[jt] = uc[i-1] + uc[i];
            icv[jt] = ff[2*i - 1];
            uf[2 * i] = uc[i];
        }

        bmt = _A/(scaledB*bmt+icv);
        jt = 0;
        for (SizeType i = ib; i < iend; ++i, ++jt) {
            uf[2 * i - 1] = bmt[jt];
        }
    }

    return uf;
}
template<class floating>
BlockVector<floating> EquidistantBlock_1D<floating>::CRRestriction(const AlgebraicMatrix<floating> &scaledB, const BlockVector<floating> &ff) const
{
    const SizeType fnr = getBlockDim();
    assert(fnr % 2 == 1); // odd number of block rows required
    const SizeType cnr = (getBlockDim() + 1) / 2;
    assert(ff.getNrows() == _A.getNcols());

    const SizeType nrow = ff.getNrows();
    AlgebraicMatrix <floating> fc = *getMatrixFactory().createMatrix(nrow, cnr);

    fc[0] = ff[0];

    constexpr SizeType BLOCKSIZE = 64;

#pragma omp parallel for
    for (SizeType ib = 2; ib < fnr - 1; ib += 2 * BLOCKSIZE) {
        AlgebraicMatrix <floating> bmt = *getMatrixFactory().createMatrix(nrow, BLOCKSIZE);
        const SizeType iend = std::min(ib + 2 * BLOCKSIZE, fnr - 1);
        SizeType jt = 0;

        for (SizeType i = ib; i < iend; i += 2, ++jt) {
            fc[i / 2] = ff[i];
            bmt[jt] = ff[i-1] + ff[i+1];
        }
        bmt = scaledB * (_A / bmt);

        fc.updateAdd(ib / 2, iend / 2, bmt);
    }

    fc[cnr - 1] = ff[fnr - 1];
    return fc;
}

template<class floating>
floating EquidistantBlock_1D<floating>::multigrid(const unsigned numberOfSmoothingSteps, const BlockVector<floating> &f, const size_t maxNumberOfIterations, const floating accuracy, BlockVector<floating> &solution) const
{
    floating omega = 2/3.0;
    const SizeType N = getNdim();
    const SizeType M = getBlockDim();

    const SizeType coarseM = getCoarseDim();

    BlockVector<floating> coarseSolution = *getMatrixFactory().createMatrix(N, coarseM);

    const auto &coarseSystem = getCoarseSystem();

    assert(solution.getNrows() == N && solution.getNcols() == M && "ERROR: Dimension mismatch.");

    floating euclideanError = 1;

    if (M == 3)
    {
        jacobiIteration(1.0, f, maxNumberOfIterations, accuracy, solution);
    }
    else
    {

        // Step 1: smoothing
        smooth(omega, f, 2, solution);

        // Step 2:
        auto residual = calculateResidual(solution, f);

        // Step 3:
        auto ffOnCoarseGrid = restriction(residual);

        // Step 4: solve on ffOnCoarseGrid:
        coarseSystem.multigrid(numberOfSmoothingSteps, ffOnCoarseGrid, maxNumberOfIterations, accuracy, coarseSolution);

        // Step 5: correction:
        solution += prolongation(coarseSolution);

        // Step 6: smoothing
        euclideanError = jacobiIteration(omega, f, numberOfSmoothingSteps, accuracy, solution);
    }

    return euclideanError;
}

template<class floating>
BlockVector<floating> EquidistantBlock_1D<floating>::prolongation(const BlockVector<floating> &ff) const
{
    const SizeType N = getNdim();
    const SizeType M = getBlockDim();

    assert(ff.getNrows() == N && ff.getNcols() == (M+1)/2 && "ERROR: Wrong dimensions.");

    auto ffOnFineGrid = *getMatrixFactory().createMatrix(N, M);

    ffOnFineGrid[0] = ff[0];
    for (SizeType i = 1; i < getBlockDim()-1; i += 2)
    {
        ffOnFineGrid[i+1] = ff[(i+1)/2];
        ffOnFineGrid[i] = static_cast<floating>(1/2.) * (ffOnFineGrid[i-1] + ffOnFineGrid[i+1]);
    }

    return ffOnFineGrid;
}

template<class floating>
BlockVector<floating> EquidistantBlock_1D<floating>::restriction(const BlockVector<floating> &ff) const
{
    const SizeType N = getNdim();
    const SizeType M = getBlockDim();
    assert(getBlockDim() % 2 == 1 && "ERROR: Need odd number of rows in each step.");

    const SizeType coarseGridSize = (getBlockDim()+1)/2;

    auto ffOnCoarseGrid = *getMatrixFactory().createMatrix(N, coarseGridSize);

// Full restriction
#pragma omp parallel for
    for (SizeType i = 2; i < M-1; i+=2)
    {
        ffOnCoarseGrid[i/2] = static_cast<floating>(1/2.) * (ff[i-1] + ff[i+1]) + ff[i];
    }
    return ffOnCoarseGrid;
}

template<class floating>
floating EquidistantBlock_1D<floating>::getReducedGrid() const
{
    return 2.*_h;
}

template<class floating>
typename EquidistantBlock_1D<floating>::SizeType EquidistantBlock_1D<floating>::getCoarseDim() const
{
    return (getBlockDim()+1)/2;
}

template<class floating>
std::unique_ptr<EquidistantBlock_1D<floating>> EquidistantBlock_1D<floating>::initializeCoarseSystemPtr() const
{
    if (getBlockDim() <= 3)
    {
        return nullptr;
    }
    else
    {
        return std::make_unique<EquidistantBlock_1D<floating>>(getCoarseDim(), _B, _D, _M, getReducedGrid(), _alpha, _timeGridStepSize, getProcessingUnit());
    }
}

template<class floating>
const EquidistantBlock_1D<floating>& EquidistantBlock_1D<floating>::getCoarseSystem() const
{
    return *_coarseSystemPtr;
}

template<class floating>
floating EquidistantBlock_1D<floating>::jacobiIteration(const floating omega, const BlockVector<floating> &f, const size_t maxNumberOfIterations, const floating , BlockVector<floating> &solution) const
{
    const SizeType M = f.getNcols();

    const floating inv_scale_factor = 1/_h/2;

    assert(solution.getNrows() == getNdim() && solution.getNcols() == M && "ERROR: Dimension mismatch.");

    floating euclideanNormResidual = 0;

    size_t n = 0;
    do
    {
        euclideanNormResidual = 0;
        auto residual0 = calculateRowResidual(0, solution, f);
        euclideanNormResidual += scalarProduct(residual0, residual0);

        solution[0] = _M / f[0];

        for (unsigned int i = 1; i < M-1; ++i)
        {
            auto residual = calculateRowResidual(i, solution, f);
            euclideanNormResidual += scalarProduct(residual, residual);

            // scale residual: omega * (D * C)^-1 * r_i = omega * C^-1 * (D^-1 * r_i)
            residual = (inv_scale_factor)*residual;
            solution[i] += omega*(_A/residual);
        }

        residual0 = calculateRowResidual(M-1, solution, f);
        euclideanNormResidual += scalarProduct(residual0, residual0);
        solution[M-1] = _M / f[M-1];

        euclideanNormResidual = sqrt(euclideanNormResidual);
        ++n;
    }
        // check for the two breaking conditions:
        // 1. if maxNumberOfIterations has a feasible value, repeat until maxNumberOfIterations is reached
        // 2. if maxNormResidual has a feasible value, repeat until relative error is smaller than desired accuracy
    while ((maxNumberOfIterations <= 0 || n < maxNumberOfIterations) /*&& (accuracy <= 0 || euclideanNormResidual >= accuracy)*/);

    return euclideanNormResidual;
}

template<class floating>
void EquidistantBlock_1D<floating>::smooth(const floating omega, const BlockVector<floating> &f, const size_t maxNumberOfIterations, BlockVector<floating> &solution) const
{
    const SizeType M = f.getNcols();

    const floating inv_scale_factor = 1/_h/2;

    assert(solution.getNrows() == getNdim() && solution.getNcols() == M && "ERROR: Dimension mismatch.");

    size_t n = 0;
    do
    {
        solution[0] = _M / f[0];

        for (unsigned int i = 1; i < M-1; ++i) {
            // scale residual: omega * (D * C)^-1 * r_i = omega * C^-1 * (D^-1 * r_i)
            solution[i] += inv_scale_factor * omega * (_A / calculateRowResidual(i, solution, f));
        }

        solution[M-1] = _M / f[M-1];
        ++n;
    }
        // check for the breaking condition:
        // repeat until maxNumberOfIterations is reached
    while (n < maxNumberOfIterations);
}


template<class floating>
BlockVector<floating> EquidistantBlock_1D<floating>::calculateResidual(const BlockVector<floating> &u, const BlockVector<floating> &f) const
{
    const SizeType N = getNdim();
    const SizeType M = f.getNcols();


    assert(u.getNrows() == N && u.getNcols() == M && "ERROR: Dimension mismatch.");
    auto residual = *getMatrixFactory().createMatrix(N, M);  // GH: das dürfte bei Unfied memory langsam sein, diesen Speicher benötigen wir nur auf der GPU.

    for (unsigned int i = 0; i < M; ++i)
    {
        residual[i] = calculateRowResidual(i, u, f);
    }

    return residual;
}

template<class floating>
AlgebraicVector<floating> EquidistantBlock_1D<floating>::calculateRowResidual(const SizeType i, const BlockVector<floating> &u, const BlockVector<floating> &f) const
{
    const SizeType M = f.getNcols();

    auto residual = f[i];

    if (i == 0 || i == M-1)
    {
        residual.add(_M*u[i], -1.0);
    }
    else
    {
        const floating coeff_left =  2/_h;
        const floating coeff_right =  2/_h;
        const floating scale_factor = 2*_h;

        residual += (_B*((coeff_left * u[i-1]).add(u[i+1], coeff_right))).add(_A*u[i], -scale_factor);
    }

    return residual;
}

template<class floating>
BlockVector<floating> EquidistantBlock_1D<floating>::rescale_rhs(const BlockVector<floating> &rhs) const
{
    const SizeType M = getBlockDim();
    auto scaled_rhs = rhs;

    for (SizeType j = 1; j < M-1; ++j)
    {
        scaled_rhs[j] = 2 * _h * scaled_rhs[j];
    }

    return scaled_rhs;
}
