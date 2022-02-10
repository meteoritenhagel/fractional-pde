#include "statusbar/statusbar.h"

#ifndef CPU_ONLY
#include "gpukernels.cuh"
#endif

// TODO: REMOVE
#include <fstream>
#include <string>

// public:

template<class floating>
NonEquidistantBlock_1D<floating>::NonEquidistantBlock_1D(const SizeType bdim,
                                                         const AlgebraicMatrix<floating> &B,
                                                         const CoefficientMatrix<floating> &D,
                                                         const AlgebraicMatrix<floating> &M,
                                                         const AlgebraicVector<floating> &h, const floating alpha,
                                                         const floating timeGridStepSize,
                                                         const ProcessingUnit<floating> processingUnit)
        : NonEquidistantBlock_1D(bdim, B, D.copyToDense(), M, h, alpha, timeGridStepSize, processingUnit) {}

template<class floating>
NonEquidistantBlock_1D<floating>::NonEquidistantBlock_1D(const SizeType bdim,
                                                         const AlgebraicMatrix<floating> &B, const AlgebraicMatrix<floating> &D,
                                                         const AlgebraicMatrix<floating> &M,
                                                         const AlgebraicVector<floating> &h, const floating alpha,
                                                         const floating timeGridStepSize,
                                                         const ProcessingUnit<floating> processingUnit)
        : _bdim(bdim), _B(B), _D(D), _M(M), _alpha(alpha), _timeGridStepSize(timeGridStepSize), _h(h),
          _C(initializeC()), _colMatrixFactory(processingUnit), _coarseSystemPtr(std::move(initializeCoarseSystemPtr())),
          _vectorBuffer(*getMatrixFactory().createMatrix(getNdim(), 10)),
          _host_h(*ContainerFactory(static_cast<ProcessingUnit<floating>>(std::make_shared<CPU<floating>>())).createColumn(getNdim())),
          _buffer1(*getMatrixFactory().createMatrix(getNdim(), getBlockDim())),
          _buffer2(*getMatrixFactory().createMatrix(getNdim(), getBlockDim())),
          _buffer3(*getMatrixFactory().createMatrix(getNdim(), getBlockDim())),
          _buffer4(*getMatrixFactory().createMatrix(getNdim(), getBlockDim())),
          _buffer5(*getMatrixFactory().createMatrix(getNdim(), getBlockDim())),
          _buffer6(*getMatrixFactory().createMatrix(getNdim(), getBlockDim())),
          _coarseBuffer1(*getMatrixFactory().createMatrix(getNdim(), getCoarseDim())),
          _coarseBuffer2(*getMatrixFactory().createMatrix(getNdim(), getCoarseDim()))
{
    //_h = h;
    assert(_B.isSquare() && _M.isSquare());
    assert(_B.getNrows() == _D.getNrows());
    assert(_B.getNrows() == _M.getNrows());
    assert(_B.getNcols() == _M.getNcols());

    assert(_h.size() == _bdim-1);

//    std::cout << "CPU_B " << _B.data() << " / " << _B.data() + _B.getNelems() - 1 << std::endl;
//    std::cout << "CPU_C " << _C.data() << " / " << _C.data() + _C.getNelems() - 1 << std::endl;
//    std::cout << "CPU_D " << _D.data() << " / " << _D.data() + _D.getNelems() - 1 << std::endl;
//    std::cout << "CPU_M " << _M.data() << " / " << _M.data() + _M.getNelems() - 1 << std::endl << std::endl;
//
//    _B.getProcessingUnit()->display();

    // Move everything to device:
    _B.moveTo(processingUnit);
    _C.moveTo(processingUnit);
    _D.moveTo(processingUnit);
    _M.moveTo(processingUnit);
//
//    _B.getProcessingUnit()->display();
//
//    std::cout << "_B " << _B.data() << " / " << _B.data() + _B.getNelems() - 1 << std::endl;
//    std::cout << "_C " << _C.data() << " / " << _C.data() + _C.getNelems() - 1 << std::endl;
//    std::cout << "_D " << _D.data() << " / " << _D.data() + _D.getNelems() - 1 << std::endl;
//    std::cout << "_M " << _M.data() << " / " << _M.data() + _M.getNelems() - 1 << std::endl;
//    std::cout << "_h " << _h.data() << " / " << _h.data() + _h.size() - 1 << std::endl;

    // _h is used in coefficients only, which must be on the CPU
    _h.moveTo(std::make_shared<CPU<floating>>());
    _host_h = _h;
    _h.moveTo(processingUnit);

    _C.getInverse(); // calculate inverse for multigrid
    _M.getInverse();
}

template<class floating>
typename NonEquidistantBlock_1D<floating>::SizeType NonEquidistantBlock_1D<floating>::getNdim() const
{
    return _B.getNrows();
}

template<class floating>
typename NonEquidistantBlock_1D<floating>::SizeType NonEquidistantBlock_1D<floating>::getBlockDim() const
{
    return _bdim;
}

template<class floating>
typename NonEquidistantBlock_1D<floating>::SizeType NonEquidistantBlock_1D<floating>::getDenseDim() const
{
    return getBlockDim() * getNdim();
}

template<class floating>
ContainerFactory<floating> NonEquidistantBlock_1D<floating>::getMatrixFactory() const
{
    return _colMatrixFactory;
}

template<class floating>
BlockVector<floating> NonEquidistantBlock_1D<floating>::solve(BlockVector<floating> &rhs, const size_t maxNumberOfIterations, const size_t stepsPerIteration, const floating accuracy, const SolvingProcedure solvingProcedure) const
{
    const SizeType N = getNdim();
    const SizeType M = getBlockDim();
    rhs.moveTo(getProcessingUnit());

#ifndef UNSYMMETRIZED
    // Since we work with the symmetrised system, also rhs must be rescaled appropiately
    rescale_rhs(rhs);
#endif

    const floating relativeAccuracy = accuracy * rhs.getEuclidean();
    std::cout << "relative Accuracy: " << relativeAccuracy << std::endl;

    BlockVector<floating> solution = *getMatrixFactory().createMatrix(N, M);

    auto residual{solution};
    calculateResidual(solution, rhs, residual);

    std::cout << "initial error: " << residual.getEuclidean() << std::endl;

    // TODO: Uncomment
    solution[0] = _M / rhs[0];
    solution[M-1] = _M / rhs[M-1];

    floating euclideanNorm = 0;

    CHRONO_Timer stopwatch;
    size_t totalCounter = 0;

    size_t n = 0;
    do
    {
        switch(solvingProcedure)
        {
            case SolvingProcedure::CyclicReduction:
                throw std::runtime_error("EquidistantBlock_1D does not support Cyclic Reduction. Abort.");
            case SolvingProcedure::PCBiCGStab:
                euclideanNorm = PCbiCGStab(rhs, stepsPerIteration, relativeAccuracy, solution);
                break;

            case SolvingProcedure::BiCGStab:
                euclideanNorm = biCGStab(rhs, stepsPerIteration, relativeAccuracy, solution);
                break;

            case SolvingProcedure::Richardson:
                smooth(static_cast<floating>(1.0), rhs, stepsPerIteration, solution);
                calculateResidual(solution, rhs, residual);
                euclideanNorm = residual.getEuclidean();

                ++totalCounter;
                std::cout << "   " << totalCounter << ": " << euclideanNorm << std::endl;
#ifdef PRINT
                {
                    std::ofstream filestream1;
                    filestream1.open("stepwise_tests/" + std::to_string(getNdim()-3) + "_" + std::to_string(getBlockDim()-1) + "test_richardson.txt", std::fstream::in | std::fstream::out | std::fstream::app);
                    filestream1 << totalCounter << " " << euclideanNorm << std::endl;
                    filestream1.close();
                }
#endif
                break;

                break;

            case SolvingProcedure::PCRichardson:
                multigrid(2, rhs, stepsPerIteration, relativeAccuracy, solution);
                
                calculateResidual(solution, rhs, residual);
                euclideanNorm = residual.getEuclidean();
                ++totalCounter;
                std::cout << "   " << totalCounter << ": " << euclideanNorm << std::endl;
#ifdef PRINT
                {
                    std::ofstream filestream2;
                    filestream2.open("stepwise_tests/" + std::to_string(getNdim()-3) + "_" + std::to_string(getBlockDim()-1) + "test_pcrichardson.txt", std::fstream::in | std::fstream::out | std::fstream::app);
                    filestream2 << totalCounter << " " << euclideanNorm << std::endl;
                    filestream2.close();
                }
#endif
                break;
            }

        // TODO: REMOVE:
//        if (totalCounter == 9)
//            break;
//        euclideanNorm = 100;



        //std::cout << maxNormResidual << std::endl;
        ++n;
    }
    // check for the two breaking conditions:
    // 1. if maxNumberOfIterations has a feasible value, repeat until maxNumberOfIterations is reached
    // 2. if maxNormResidual has a feasible value, repeat until relative error is smaller than desired accuracy
    while ((maxNumberOfIterations <= 0 || n < maxNumberOfIterations) && (accuracy <= 0 || euclideanNorm >= relativeAccuracy));

    stopwatch.stop();
    getProcessingUnit()->_multigridTime += stopwatch.elapsedTime();

    return solution;
}

// private:

template<class floating>
AlgebraicMatrix<floating> NonEquidistantBlock_1D<floating>::copyToDense() const
{
    const SizeType loc_rows = _B.getNrows();      // #rows per block
    // dimensions of dense matrix
    const SizeType glob_rows = getDenseDim();

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

    for (SizeType k = 1; k < getBlockDim() - 1; ++k)
    {
        const SizeType startIndex = k * loc_rows;
        const SizeType endIndex = (k+1) * loc_rows;

        for (SizeType i = startIndex; i < endIndex; i++)
        {
            const SizeType blockNumber = i / loc_rows - 1;

#ifdef UNSYMMETRIZED
            // unsymmetrised system
            const floating coeff_left = - 2/_host_h[blockNumber]/(_host_h[blockNumber] + _host_h[blockNumber+1]);
            const floating coeff_middle = 2/_host_h[blockNumber]/_host_h[blockNumber+1];
            const floating coeff_right = - 2/_host_h[blockNumber + 1]/(_host_h[blockNumber] + _host_h[blockNumber+1]);
            const floating scale_factor = 1.0;
#else
            // Symmetrised system
            const floating coeff_left =  - 2/_host_h[blockNumber];
            const floating coeff_middle = 2*(_host_h[blockNumber] + _host_h[blockNumber+1])/_host_h[blockNumber]/_host_h[blockNumber+1];
            const floating coeff_right =  - 2/_host_h[blockNumber+1];
            const floating scale_factor = (_host_h[blockNumber] + _host_h[blockNumber+1]);
#endif
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
                    DD(i,j) = coeff_middle * host_M(i - startIndex, j - startIndex) - scale_factor*getSystemCoeff() * host_D(i - startIndex, j - startIndex);
                }
            }

            for (SizeType j = endIndex; j < endIndex + loc_rows; j++) DD(i,j) = coeff_right * host_B(i - startIndex, j - endIndex);
        }
    }


    for (SizeType j = glob_rows - loc_rows; j < glob_rows; j++)
        for (SizeType i = glob_rows - loc_rows; i < glob_rows; i++)
            DD(i,j) = host_M(i - glob_rows + loc_rows, j - glob_rows + loc_rows);

    DD.moveTo(getProcessingUnit());
    return DD;
}

template<class floating>
void NonEquidistantBlock_1D<floating>::mult(const BlockVector<floating> &u, BlockVector<floating> &result) const
{
    const SizeType N = getNdim();
    const SizeType M = u.getNcols();
    assert(u.getNrows() == N && M == getBlockDim() && "ERROR: Dimension mismatch.");

    //result[0] <- _M*u[0];
    getProcessingUnit()->xgemv(OperationType::Identical, _M.getNrows(), _M.getNcols(),
                               static_cast<floating>(1.), _M.data(), _M.getNrows(), u[0].data(), 1,
                               static_cast<floating>(0.), result[0].data(), 1);

    for (SizeType i = 1; i < getBlockDim()-1; ++i)
    {

#ifdef UNSYMMETRIZED
        // Plain system
        const floating coeff_left =  2/_host_h[i-1]/(_host_h[i-1] + _host_h[i]);
        const floating coeff_middle = 2/_host_h[i-1]/_host_h[i];
        const floating coeff_right =  2/_host_h[i]/(_host_h[i-1] + _host_h[i]);
        const floating scale_factor = 1.0;
#else
        // Symmetrised system
        const floating coeff_left =  2/_host_h[i-1];
        const floating coeff_middle = 2*(_host_h[i-1] + _host_h[i])/_host_h[i-1]/_host_h[i];
        const floating coeff_right =  2/_host_h[i];
        const floating scale_factor = (_host_h[i-1] + _host_h[i]);
#endif

        // WE WANT: scaled -_B * u[i-1] + _A * u[i] + -_B * u[i+1]
        // Note that _A = (c1 * _M - c2 * _D) except for rows 1, 2, N, where it is _M

        // result[i] <- scaled _M * u[i]
        getProcessingUnit()->xgemv(OperationType::Identical, _M.getNrows(), _M.getNcols(),
                                   scale_factor, _M.data(), _M.getNrows(), u[i].data(), 1,
                                   static_cast<floating>(0), result[i].data(), 1);

        _vectorBuffer[0] = result[i];

        // _vectorBuffer[0] holds scaled _A * u[i] except in the rows 1, 2, N (!!! because of this we cannot work without buffer)
        getProcessingUnit()->xgemv(OperationType::Identical, _D.getNrows(), _D.getNcols(),
                                   -scale_factor*getSystemCoeff(), _D.data(), _D.getNrows(), u[i].data(), 1,
                                   coeff_middle/scale_factor, _vectorBuffer[0].data(), 1);


        // result[i] holds scaled _A * u[i]
        getProcessingUnit()->getMemoryManager()->copy(result[i].data()+2, _vectorBuffer[0].data() + 2, (N - 3) * sizeof(floating));


        // _vectorBuffer[1] holds scaled _B * (u[i-1] + u[i+1])
        _vectorBuffer[1] = u[i - 1];
        getProcessingUnit()->xaxpy(_vectorBuffer[1].size(), coeff_right / coeff_left, u[i + 1].data(), 1, _vectorBuffer[1].data(), 1);

        // result[i] holds scaled -_B * u[i-1] + _A * u[i] + -_B * u[i+1], which is what we want
        getProcessingUnit()->xgemv(OperationType::Identical, _B.getNrows(), _B.getNcols(),
                                   -coeff_left, _B.data(), _B.getNrows(), _vectorBuffer[1].data(), 1,
                                   static_cast<floating>(1.), result[i].data(), 1);
    }

    //result[M-1] <- _M*u[M-1];
    getProcessingUnit()->xgemv(OperationType::Identical, _M.getNrows(), _M.getNcols(),
                               static_cast<floating>(1.), _M.data(), _M.getNrows(), u[M-1].data(), 1,
                               static_cast<floating>(0.), result[M-1].data(), 1);
}

template<class floating>
floating NonEquidistantBlock_1D<floating>::biCGStab(const BlockVector<floating> &f, const size_t maxNumberOfIterations, const floating accuracy, BlockVector<floating> &solution) const
{
    const auto numberOfIterations = std::min(getDenseDim(), static_cast<SizeType>(maxNumberOfIterations));


    auto residual{solution};
    calculateResidual(solution, f, residual);

    const auto residual_0 = residual;

    floating rho = 1;
    floating alpha = 1;
    floating omega = 1;

    auto v = *f.getMatrixFactory().createMatrix(f.getNrows(), f.getNcols());
    auto p = v;

    // reserve space for calculation below
    auto s = v;
    auto t = s;
    auto solutionBuffer = solution;

    //StatusBar<floating> statusBar(50);

    floating euclideanError = residual.getEuclidean();

    static int totalCounter = 1;

#ifdef PRINT
    std::ofstream filestream;
    filestream.open("stepwise_tests/" + std::to_string(getNdim()-3) + "_" + std::to_string(getBlockDim()-1) + "test_bicgstab.txt", std::fstream::in | std::fstream::out | std::fstream::app);
#endif

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

        // TODO: REMOVE
        std::cout << totalCounter << " " << euclideanError << std::endl;
        ++totalCounter;


    }
    while (iterate && counter < numberOfIterations);

    // TODO: REMOVE
#ifdef PRINT
    std::cout << "   " << totalCounter << std::endl;
    filestream.close();
#endif

    return euclideanError;
}

template<class floating>
floating NonEquidistantBlock_1D<floating>::PCbiCGStab(const BlockVector<floating> &f, const size_t maxNumberOfIterations, const floating accuracy, BlockVector<floating> &solution) const
{

    const unsigned numberOfSmoothingSteps = 2;
    const unsigned numberOfMultiGridSteps = 20; //3000; //200;

    const auto numberOfIterations = std::min(getDenseDim(), static_cast<SizeType>(maxNumberOfIterations));

    auto &residual = _buffer6;
    calculateResidual(solution, f, residual);

    const auto residual_0 = residual;

    floating rho = 1;
    floating alpha = 1;
    floating omega = 1;

    auto v = *f.getMatrixFactory().createMatrix(f.getNrows(), f.getNcols());
    auto p = v;

    auto s = v;
    auto t = v;
    auto z = v;
    auto y = v;

    auto solutionBuffer = y;
    floating euclideanError = residual.getEuclidean();

    //StatusBar<floating> statusBar(50);

    static int totalCounter = 1;
#ifdef PRINT
    std::ofstream filestream;
    filestream.open("stepwise_tests/" + std::to_string(getNdim()-3) + "_" + std::to_string(getBlockDim()-1) + "test_pcbicgstab.txt", std::fstream::in | std::fstream::out | std::fstream::app);
#endif

    bool iterate = true;
    SizeType counter = 0;
    do
    {
        const auto rho_new = scalarProduct(residual_0, residual);
        const auto beta = (rho_new/rho)*(alpha/omega);
        rho = rho_new;

        //p = residual + beta*(p - omega * v);
        getProcessingUnit()->xaxpy(p.getNelems(), -omega, v.data(), 1, p.data(), 1);
        p.scale(beta);
        getProcessingUnit()->xaxpy(p.getNelems(), static_cast<floating>(1.0), residual.data(), 1, p.data(), 1);
        y.add(y, static_cast<floating>(-1.0)); // y <- 0
        multigrid(numberOfSmoothingSteps, p, numberOfMultiGridSteps, accuracy, y);
        this->mult(y, v);
        alpha = rho/scalarProduct(residual_0, v);

        //s = residual - alpha*v;
        s = residual;
        getProcessingUnit()->xaxpy(s.getNelems(), -alpha, v.data(), 1, s.data(), 1);
        z.add(z, static_cast<floating>(-1.0)); // z <- 0
        multigrid(numberOfSmoothingSteps, s, numberOfMultiGridSteps, accuracy, z);
        this->mult(z, t);

        omega = scalarProduct(t, s)/scalarProduct(t, t);

        //solution = solution + omega*z + alpha*y;
        getProcessingUnit()->xaxpy(solution.getNelems(), omega, z.data(), 1, solution.data(), 1);
        getProcessingUnit()->xaxpy(solution.getNelems(), alpha, y.data(), 1, solution.data(), 1);

        //residual = s - omega*t;
        residual = s;
        getProcessingUnit()->xaxpy(residual.getNelems(), -omega, t.data(), 1, residual.data(), 1);

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
        //statusBar.draw(counter, numberOfIterations, currentError, 0, 10);
        
        // TODO: REMOVE
        std::cout << totalCounter << " " << euclideanError << std::endl;

#ifdef PRINT
        filestream << totalCounter << " " << euclideanError << std::endl;
#endif
        ++totalCounter;
    }
    while (iterate && counter < numberOfIterations);

    // TODO: REMOVE
#ifdef PRINT
    filestream.close();
#endif

    return euclideanError;
}


template<class floating>
void NonEquidistantBlock_1D<floating>::multigrid(const unsigned numberOfSmoothingSteps, const BlockVector<floating> &f, const size_t maxNumberOfIterations, const floating accuracy, BlockVector<floating> &solution) const
{
// TODO: REMOVE
//auto stopwatch = std::make_shared<CHRONO_Timer>();
//stopwatch->start();
//auto stopwatch1 = std::make_shared<CHRONO_Timer>();
//auto stopwatch2 = std::make_shared<CHRONO_Timer>();
//auto stopwatch3 = std::make_shared<CHRONO_Timer>();
//auto stopwatch4 = std::make_shared<CHRONO_Timer>();
//auto stopwatch5 = std::make_shared<CHRONO_Timer>();
//auto stopwatch6 = std::make_shared<CHRONO_Timer>();
//auto stopwatch7 = std::make_shared<CHRONO_Timer>();

//stopwatch1->start();
    const SizeType N = getNdim();
    const SizeType M = getBlockDim();

    BlockVector<floating> &coarseSolution = _coarseBuffer1;
    initializeMemory(getProcessingUnit()->getMemoryManager(), coarseSolution.data(), coarseSolution.getNelems(), static_cast<floating>(0.0));

    BlockVector<floating> &ffOnCoarseGrid = _coarseBuffer2;
    initializeMemory(getProcessingUnit()->getMemoryManager(), ffOnCoarseGrid.data(), ffOnCoarseGrid.getNelems(), static_cast<floating>(0.0));

    BlockVector<floating> &residual = _buffer1;
    initializeMemory(getProcessingUnit()->getMemoryManager(), residual.data(), residual.getNelems(), static_cast<floating>(0.0));

    const auto &coarseSystem = getCoarseSystem();

    assert(solution.getNrows() == N && solution.getNcols() == M && "ERROR: Dimension mismatch.");

    floating euclideanError = 1;

    // TODO: Find right omega
    const floating omega = 2/3.0;

    //TODO: REMOVE
    //std::cout << "    N = " << N << ", M = " << M << '\n';
//stopwatch1->stop();

    if (M == 3)
    {
        // exact solution on the coarsest grid with 3 points
        smooth(1.0, f, maxNumberOfIterations, solution);
    }
    else
    {
        // TODO: REMOVE
//        copyToDense().display("sys");
//        solution.display("init solution");
//        f.display("rhs");

//stopwatch2->start();
        // Step 1: smoothing
        smooth(omega, f, numberOfSmoothingSteps, solution);
//stopwatch2->stop();

        // TODO: Remove
//        solution.display("after smoothing: ");

//stopwatch3->start();
        // Step 2:
        calculateResidual(solution, f, residual);
//stopwatch3->stop();

        // TODO: Remove
//        residual.display("defect after smoothing: ");

        //exit(0);

//stopwatch4->start();
        // Step 3:
        restriction(residual, ffOnCoarseGrid);
//stopwatch4->stop();

        // TODO: Remove
//        ffOnCoarseGrid.display("restricted defect");


        // Step 4: solve on ffOnCoarseGrid:

//        coarseSolution[0] = _M / ffOnCoarseGrid[0];
//        coarseSolution[coarseM-1] = _M / ffOnCoarseGrid[coarseM-1];

//stopwatch5->start();
        coarseSystem.multigrid(numberOfSmoothingSteps, ffOnCoarseGrid, maxNumberOfIterations, accuracy, coarseSolution);
//stopwatch5->stop();

        // TODO: Remove
        //coarseSolution.display("solution after multigrid");

        // Step 5: correction:

        // TODO: Remove
        //prolongation(coarseSolution).display("interpolation of solution");

//        stopwatch6->start();
        prolongation(coarseSolution, solution);
//        stopwatch6->stop();
        // Step 6: smoothing

//        stopwatch7->start();
        //euclideanError = jacobiIteration(omega, f, numberOfSmoothingSteps, accuracy, solution);
        smooth(omega, f, numberOfSmoothingSteps, solution);
//        stopwatch7->stop();
    }
//stopwatch->stop();

//    std::cout << "Multigrid time: " << stopwatch->elapsedTime() << "s." << std::endl
//              << "   init       : " << stopwatch1->elapsedTime() << "s. (" << stopwatch1->elapsedTime()/stopwatch->elapsedTime() * 100. << "%)" << std::endl
//              << "   pre-smooth : " << stopwatch2->elapsedTime() << "s. (" << stopwatch2->elapsedTime()/stopwatch->elapsedTime() * 100. << "%)" << std::endl
//              << "   residual   : " << stopwatch3->elapsedTime() << "s. (" << stopwatch3->elapsedTime()/stopwatch->elapsedTime() * 100. << "%)" << std::endl
//              << "   restriction: " << stopwatch4->elapsedTime() << "s. (" << stopwatch4->elapsedTime()/stopwatch->elapsedTime() * 100. << "%)" << std::endl
//              << "   small MG   : " << stopwatch5->elapsedTime() << "s. (" << stopwatch5->elapsedTime()/stopwatch->elapsedTime() * 100. << "%)" << std::endl
//              << "   prolongatio: " << stopwatch6->elapsedTime() << "s. (" << stopwatch6->elapsedTime()/stopwatch->elapsedTime() * 100. << "%)" << std::endl
//              << "   post-smooth: " << stopwatch7->elapsedTime() << "s. (" << stopwatch7->elapsedTime()/stopwatch->elapsedTime() * 100. << "%)" << std::endl << std::endl;
    //exit(-1);
    //return euclideanError;
}

template<class floating>
void NonEquidistantBlock_1D<floating>::prolongation(const BlockVector<floating> &ff, BlockVector<floating> &solution) const
{
    const SizeType N = getNdim();
    const SizeType M = getBlockDim();

    assert(ff.getNrows() == N && ff.getNcols() == (M+1)/2 && "ERROR: Wrong dimensions.");

    if (typeid(*this->getProcessingUnit()) == typeid(*std::make_shared<CPU<floating>>()))
    {
        solution[0] += ff[0];
#pragma omp parallel for  if (M>127)               // GH
        for (SizeType i = 1; i < getBlockDim()-1; i += 2)
        {
            const auto coeff1 = _h[i] / (_h[i - 1] + _h[i]);
            const auto coeff2 = _h[i] / (_h[i - 1] + _h[i]);
            solution[i + 1] += ff[(i + 1) / 2];
            //solution[i] += coeff1 * ff[(i - 1) / 2];
            getProcessingUnit()->xaxpy(N, coeff1, ff[(i-1)/2].data(), 1, solution[i].data(), 1);
            //solution[i] += coeff2 * ff[(i + 1) / 2];
            getProcessingUnit()->xaxpy(N, coeff2, ff[(i+1)/2].data(), 1, solution[i].data(), 1);
        }
    }
    else
    {
#ifndef CPU_ONLY
        deviceProlongation(N, M, _h.data(), ff.data(), solution.data());
#endif
    }
}

template<class floating>
void NonEquidistantBlock_1D<floating>::restriction(const BlockVector<floating> &ff, BlockVector<floating> &ffOnCoarseGrid) const
{
    const SizeType N = getNdim();
    const SizeType M = getBlockDim();
    assert(getBlockDim() % 2 == 1 && "ERROR: Need odd number of rows in each step.");

    // injection
//    for (SizeType i = 0; i < coarseGridSize; ++i)
//        ffOnCoarseGrid[i] = ff[2*i];

#ifdef UNSYMMETRIZED
    // TODO: Full weighted restriction
////    ffOnCoarseGrid[0] = (_h[0]+_h[1])/(_h[0] + 2*_h[1])*ff[0] + _h[1]/(_h[0] + 2*_h[1])*ff[1];
    for (SizeType i = 2; i < M-1; i+=2)
    {
        const floating divisor = (_h[i-1]*_h[i] + 2*_h[i-2]*_h[i] + 2*_h[i-1]*_h[i+1] + 3*_h[i-2]*_h[i+1]);
        ffOnCoarseGrid[i/2] = (_h[i-2]*(_h[i] + _h[i+1])/divisor * ff[i-1] + (_h[i-2] + _h[i])*(_h[i] + _h[i+1])/divisor * ff[i] + _h[i+1]*(_h[i-2] + _h[i-1])/divisor * ff[i+1]);
    }
////    ffOnCoarseGrid[coarseGridSize-1] = _h[M-3]/(2*_h[M-3] + _h[M-2])*ff[M-2] + (_h[M-3] + _h[M-2])/(2*_h[M-3] + _h[M-2])*ff[M-1]
#else
// TODO: Full restriction
    if (typeid(*this->getProcessingUnit()) == typeid(*std::make_shared<CPU<floating>>()))
    {
#pragma omp parallel for  if (M>127)               // GH
        for (SizeType j = 2; j < M - 1; j += 2)
        {
            const auto coeff1 = _h[j - 2] / (_h[j - 2] + _h[j - 1]);
            const auto coeff2 = _h[j + 1] / (_h[j] + _h[j + 1]);
            //ffOnCoarseGrid[j/2] = coeff1 * ff[j-1] + ff[j] + coeff2 * ff[j+1];
            ffOnCoarseGrid[j / 2] = ff[j];
            getProcessingUnit()->xaxpy(getNdim(), coeff1, ff[j - 1].data(), 1, ffOnCoarseGrid[j / 2].data(), 1, false);
            getProcessingUnit()->xaxpy(getNdim(), coeff2, ff[j + 1].data(), 1, ffOnCoarseGrid[j / 2].data(), 1, false);
        }
    }
    else
    {
#ifndef CPU_ONLY
        deviceRestriction(N, M, _h.data(), ff.data(), ffOnCoarseGrid.data());
#endif
    }
#endif
}

template<class floating>
AlgebraicVector<floating> NonEquidistantBlock_1D<floating>::getReducedGrid() const
{
    assert(_h.size() % 2 == 0 && "ERROR: Dimension mismatch. Grid size has to be even number.");

    auto coarseGrid = *ContainerFactory<floating>(std::make_shared<CPU<floating>>()).createColumn(_h.size()/2);

    SizeType const Mf = coarseGrid.size();                // GH

    if (typeid(*this->getProcessingUnit()) == typeid(*std::make_shared<CPU<floating>>()))
    {
#pragma omp parallel for  //if (Mf>127)
        for (SizeType i = 0; i < Mf; ++i)      // GH
            coarseGrid[i] = _h[2*i] + _h[2*i+1];
    }
    else
    {
#ifndef CPU_ONLY
        // TODO: REMOVE
        _h.moveTo(getProcessingUnit());
        coarseGrid.moveTo(getProcessingUnit());
        deviceGetReducedGrid(Mf, _h.data(), coarseGrid.data());
#endif
    }

    return coarseGrid;
}

template<class floating>
typename NonEquidistantBlock_1D<floating>::SizeType NonEquidistantBlock_1D<floating>::getCoarseDim() const
{
    return (getBlockDim()+1)/2;
}

template<class floating>
std::unique_ptr<NonEquidistantBlock_1D<floating>> NonEquidistantBlock_1D<floating>::initializeCoarseSystemPtr() const
{
    if (getBlockDim() <= 3)
        return nullptr;
    else
        return std::make_unique<NonEquidistantBlock_1D<floating>>(getCoarseDim(), _B, _D, _M, getReducedGrid(),
                _alpha, _timeGridStepSize, getProcessingUnit());
}

template<class floating>
const NonEquidistantBlock_1D<floating>& NonEquidistantBlock_1D<floating>::getCoarseSystem() const
{
    return *_coarseSystemPtr;
}

template<class floating>
floating NonEquidistantBlock_1D<floating>::jacobiIteration(const floating omega, const BlockVector<floating> &f, const size_t maxNumberOfIterations, const floating , BlockVector<floating> &solution) const
{
    //assert((maxNumberOfIterations > 0 || accuracy > 0) && "ERROR: No valid termination criterion given.");

    const SizeType N = getNdim();
    const SizeType M = f.getNcols();

    assert(solution.getNrows() == N && solution.getNcols() == M && "ERROR: Dimension mismatch.");

    //StatusBar<floating> statusBar(50);

    floating euclideanNormResidual = 0;

    auto &residual = _vectorBuffer[2];
    auto &outputBuffer = _vectorBuffer[3];

    size_t n = 0;
    do
    {
        euclideanNormResidual = 0;

        // Since these two lines are invariants of the algortithm, we assume that they already
        // have been computed!
        //solution[0] = _M / f[0];
        //solution[M-1] = _M / f[M-1];

// GH: Hier muss über die Blöcke parallelisiert werden.
// GH: Hauptaufwand ist die Operation  _C/residual welche damit parallelisiert wird.
// GH: Noch nicht korrekt, bei 2 Threads kommt ein Segmentation fault; Gelöst durch GH:(*)
#pragma omp parallel for reduction(+:euclideanNormResidual) if (M>63) //schedule(dynamic)
        for (unsigned int i = 1; i < M-1; ++i)
        {

#ifdef UNSYMMETRIZED
            // Plain system
            const floating inv_scale_factor = 1.0;
            const floating inv_coeff_middle = _h[i-1]*_h[i]/2;
#else
            // Symmetrised system
            const floating inv_scale_factor = 1/(_h[i-1] + _h[i]);
            const floating inv_coeff_middle = 1/(2*(_h[i-1] + _h[i])/_h[i-1]/_h[i]);
#endif

            calculateRowResidual(i, solution, f, residual);

            euclideanNormResidual += scalarProduct(residual, residual);

            // scale residual: omega * (D * C)^-1 * r_i = omega * C^-1 * (D^-1 * r_i)
            getProcessingUnit()->xscal(N, inv_scale_factor, residual.data(), 1);
            getProcessingUnit()->xscal(N-3, inv_coeff_middle/inv_scale_factor, residual.data()+2, 1);

            //solution[i] += omega*(_C/residual);
            _C.invTimes(residual, outputBuffer);
            getProcessingUnit()->xaxpy(N, omega, outputBuffer.data(), 1, solution[i].data(), 1);
        }

        euclideanNormResidual = sqrt(euclideanNormResidual);

        // TODO: REMOVE
        //solution.display("solution");

        // TODO: REMOVE
        //std::cout << "jacobi residual norm: " << euclideanNormResidual << '\n';
        ++n;
        // TODO: ENABLE/DISABLE
        //statusBar.draw(n, maxNumberOfIterations, maxNormResidual, accuracy, 10);       
    }
        // check for the two breaking conditions:
        // 1. if maxNumberOfIterations has a feasible value, repeat until maxNumberOfIterations is reached
        // 2. if maxNormResidual has a feasible value, repeat until relative error is smaller than desired accuracy
    while ((maxNumberOfIterations <= 0 || n < maxNumberOfIterations) /*&& (accuracy <= 0 || euclideanNormResidual >= accuracy)*/);

    return euclideanNormResidual;
}


template<class floating>
void NonEquidistantBlock_1D<floating>::smooth(const floating omega, const BlockVector<floating> &f, const size_t maxNumberOfIterations, BlockVector<floating> &solution) const {
    const SizeType N = getNdim();
    const SizeType M = f.getNcols();

    assert(solution.getNrows() == N && solution.getNcols() == M && "ERROR: Dimension mismatch.");

    size_t n = 0;
    // Since these two lines are invariants of the algortithm, we assume that they already
    // have been computed!
    //solution[0] = _M / f[0];
    //solution[M - 1] = _M / f[M-1];

    // #################### GAUSS SEIDEL ##################
#ifdef GAUSS_SEIDEL
    auto &residual = _vectorBuffer[4];
    auto &outputBuffer = _vectorBuffer[5];

    do {

// Gauss-Seidel cannot be parallelized
        for (unsigned int i = 1; i < M - 1; ++i) {
            calculateRowResidual(i, solution, f, residual);

#ifdef UNSYMMETRIZED
            // Plain system
            //const floating coeff_middle = 2/_h[i-1]/_h[i];
            const floating inv_scale_factor = 1.0;
            const floating inv_coeff_middle = _h[i-1]*_h[i]/2;

            getProcessingUnit()->xscal(N, inv_scale_factor, residual.data(), 1);
            getProcessingUnit()->xscal(N-3, inv_coeff_middle/inv_scale_factor, residual.data()+2, 1);
#else
            // scale residual: omega * (D * C)^-1 * r_i = omega * C^-1 * (D^-1 * r_i)
            if (typeid(*this->getProcessingUnit()) == typeid(*std::make_shared<CPU<floating>>()))
            {
                // Symmetrised system
                const floating inv_scale_factor = 1/(_h[i-1] + _h[i]);
                const floating inv_coeff_middle = 1/(2*(_h[i-1] + _h[i])/_h[i-1]/_h[i]);
                getProcessingUnit()->xscal(N, inv_scale_factor, residual.data(), 1);
                getProcessingUnit()->xscal(N-3, inv_coeff_middle/inv_scale_factor, residual.data()+2, 1);
            }
            else
            {
#ifndef CPU_ONLY
                deviceSmoothScale(N, i, _h.data(), residual.data());
#endif
            }
#endif

            //solution[i] += omega*(_C/residual);
            _C.invTimes(residual, outputBuffer);
            getProcessingUnit()->xaxpy(N, omega, outputBuffer.data(), 1, solution[i].data(), 1);
        }

        ++n;
    }
        // check for the breaking conditions:
        // repeat until maxNumberOfIterations is reached
    while (n < maxNumberOfIterations);

#else // ######################### JACOBI ###########################

    #ifdef UNSYMMETRIZED
    std::cerr << "Jacobi-type smoothing function is not supported for unsymmetrized system yet. Use Gauss-Seidel instead."
    exit(-1);
    #endif

    auto &fullResidual = _buffer2;

    do {
        calculateResidual(solution, f, fullResidual);
        // scale residual: omega * (D * C)^-1 * r_i = omega * C^-1 * (D^-1 * r_i)


        if (typeid(*this->getProcessingUnit()) == typeid(*std::make_shared<CPU<floating>>()))
        {
#pragma omp parallel for if (M>63) //schedule(dynamic)
            for (unsigned int j = 1; j < M - 1; ++j) {
                // Symmetrised system
                const floating inv_scale_factor = 1 / (_h[j - 1] + _h[j]);
                const floating inv_coeff_middle = 1 / (2 * (_h[j - 1] + _h[j]) / _h[j - 1] / _h[j]);
                getProcessingUnit()->xscal(N, inv_scale_factor, fullResidual[j].data(), 1);
                getProcessingUnit()->xscal(N - 3, inv_coeff_middle / inv_scale_factor, fullResidual[j].data() + 2, 1);
            }
        }
        else
        {
#ifndef CPU_ONLY
            deviceSmoothFullScale(N, M, _h.data(), fullResidual.data());
#endif
        }

#ifndef PLU
        const auto Cinv = _C.accessInverse().data();
        // solution *= omega * C^-1 * residual
        getProcessingUnit()->xgemm(OperationType::Identical, OperationType::Identical,
                                   N, M-2, N,
                                   omega, Cinv, N,
                                   fullResidual[1].data(), N,
                                   static_cast<floating>(1.0), solution[1].data(), N);
#else
        int info(0);
        const int NN = N;
        const int NRHS = M-2;
        getProcessingUnit()->xgetrs(OperationType::Identical, &NN, &NRHS, _C.accessInverse().data(), &NN, _C._ipiv.data(), fullResidual[1].data(), &NN, &info);
        getProcessingUnit()->xaxpy(NN * NRHS, omega, fullResidual[1].data(), 1, solution[1].data(), 1);
#endif
        ++n;
    }
        // check for the breaking conditions:
        // repeat until maxNumberOfIterations is reached
    while (n < maxNumberOfIterations);
#endif
}

template<class floating>
void NonEquidistantBlock_1D<floating>::calculateResidual(const BlockVector<floating> &u, const BlockVector<floating> &f, BlockVector<floating> &residual) const
{                 // GH: Wird nur selten benutzt.
    const SizeType N = getNdim();
    const SizeType M = f.getNcols();

    assert(u.getNrows() == N && u.getNcols() == M && "ERROR: Dimension mismatch.");

//    residual.moveTo(std::make_shared<CPU<floating>>());
//    residual.display("res");

#pragma omp parallel for  //if (M>127)                // GH faster
    for (unsigned int i = 0; i < M; ++i)
    {
        calculateRowResidual(i, u, f, residual[i]);
    }
}

template<class floating>
void NonEquidistantBlock_1D<floating>::calculateRowResidual(const SizeType i, const BlockVector<floating> &u, const BlockVector<floating> &f, AlgebraicVector<floating> &residual) const
{
    const SizeType N = getNdim();
    const SizeType M = f.getNcols();

    assert(u.getNrows() == N && u.getNcols() == M && "ERROR: Dimension mismatch.");

    //residual = f[i];
    getProcessingUnit()->xcopy(residual.size(), f[i].data(), 1, residual.data(), 1);

    if (i == 0 || i == M-1)
    {
//        std::cout << "!!!!" << std::endl;
//        std::cout << "M: " << _M.data() << std::endl;
//        std::cout << "u: " << u.data() << std::endl;
//        std::cout << "f: " << f.data() << std::endl;
//        std::cout << "residual: " << residual.data() << std::endl;

        //residual.add(_M*u[i], -1.0);
        getProcessingUnit()->xgemv(OperationType::Identical, _M.getNrows(), _M.getNcols(), -1.0, _M.data(), _M.getNrows(), u[i].data(), 1, 1.0, residual.data(), 1);
    }
    else
    {
        //std::cout << "    i=1!!!!!" << std::endl;
#ifdef UNSYMMETRIZED
        // Plain system
        const floating coeff_left =  2/_host_h[i-1]/(_host_h[i-1] + _host_h[i]);
        const floating coeff_middle = 2/_host_h[i-1]/_host_h[i];
        const floating coeff_right =  2/_host_h[i]/(_host_h[i-1] + _host_h[i]);
        const floating scale_factor = 1.0;
#else
        // Symmetrised system
        const floating coeff_left =  2/_host_h[i-1];
        const floating coeff_middle = 2*(_host_h[i-1] + _host_h[i])/_host_h[i-1]/_host_h[i];
        const floating coeff_right =  2/_host_h[i];
        const floating scale_factor = (_host_h[i-1] + _host_h[i]);
#endif
        // WE WANT: f[i] - (scaled -_B * u[i-1] + _A * u[i] + -_B * u[i+1])
        // Note that _A = (c1 * _M - c2 * _D) except for rows 1, 2, N, where it is _M

        // result[i] <- scaled _M * u[i]

//        auto &b1 = _vectorBuffer[7];
//        auto &b2 = _vectorBuffer[8];
//        auto &b3 = _vectorBuffer[6];

        auto &buf1 = _buffer4;
        auto &buf2 = _buffer5;
        auto &buf3 = _buffer3;

        auto &b1 = buf1[i];
        auto &b2 = buf2[i];
        auto &b3 = buf3[i];

        getProcessingUnit()->xgemv(OperationType::Identical, _M.getNrows(), _M.getNcols(),
                                   scale_factor, _M.data(), _M.getNrows(), u[i].data(), 1,
                                   static_cast<floating>(0), b3.data(), 1);

        //b1 = b3;
        getProcessingUnit()->xcopy(b1.size(), b3.data(), 1, b1.data(), 1);

        // _b1 holds scaled _A * u[i] except in the rows 1, 2, N (!!! because of this we cannot work without buffer)
        getProcessingUnit()->xgemv(OperationType::Identical, _D.getNrows(), _D.getNcols(),
                                   -scale_factor*getSystemCoeff(), _D.data(), _D.getNrows(), u[i].data(), 1,
                                   coeff_middle/scale_factor, b1.data(), 1);


        // b3 holds scaled _A * u[i]
        //getProcessingUnit()->getMemoryManager()->copy(b3.data()+2, b1.data()+2, (N-3)*sizeof(floating));
        getProcessingUnit()->xcopy(N-3, b1.data()+2, 1, b3.data()+2, 1);


        // b2 holds scaled _B * (u[i-1] + u[i+1])
        // b2 = u[i-1];
        getProcessingUnit()->xcopy(b2.size(), u[i-1].data(), 1, b2.data(), 1);
        getProcessingUnit()->xaxpy(b2.size(), coeff_right/coeff_left, u[i+1].data(), 1, b2.data(), 1);

        // b3 holds scaled -_B * u[i-1] + _A * u[i] + -_B * u[i+1], which is what we want
        getProcessingUnit()->xgemv(OperationType::Identical, _B.getNrows(), _B.getNcols(),
                                   -coeff_left, _B.data(), _B.getNrows(), b2.data(), 1,
                                   static_cast<floating>(1.), b3.data(), 1);

        // residual holds what we want
        residual.add(b3, -1.0);
    }
}

template<class floating>
floating NonEquidistantBlock_1D<floating>::getSystemCoeff() const
{
    floating coeff = pow(_timeGridStepSize, -_alpha)/tgamma(4-_alpha);
    return coeff;
}

template<class floating>
AlgebraicMatrix<floating> NonEquidistantBlock_1D<floating>::initializeC() const
{
    const SizeType N = getNdim();

    floating max_h = get_h_preconditioner(_h);

    auto C = _M - max_h * max_h / 2 * getSystemCoeff() * _D;

    const SizeType rowIndices[3] = {0, 1, N-1};

#pragma omp parallel for  if (N>127)
    for (SizeType col = 0; col < N; ++col)
    {
        for (SizeType row  : rowIndices)
        {
            C(row, col) = _M(row, col);
        }
        //C[col] = _M[col];
    }

    return C;
}

template<class floating>
ProcessingUnit<floating> NonEquidistantBlock_1D<floating>::getProcessingUnit() const
{
    return _colMatrixFactory.getProcessingUnit();
}

template<class floating>
floating NonEquidistantBlock_1D<floating>::get_h_preconditioner(const AlgebraicVector<floating> &h) const
{
    return h.getMaximum();
}

template<class floating>
void NonEquidistantBlock_1D<floating>::rescale_rhs(BlockVector<floating> &rhs) const
{
    const SizeType N = getNdim();
    const SizeType M = getBlockDim();
    //assert(rhs.getNrows() == N && rhs.getNcols() == M && "ERROR: Dimension mismatch.");

    if (typeid(*this->getProcessingUnit()) == typeid(*std::make_shared<CPU<floating>>()))
    {
#pragma omp parallel for  if (M>127)
        for (SizeType j = 1; j < M-1; ++j)
        {
            rhs[j].scale(_h[j] + _h[j-1]);
        }
    }
    else
    {
#ifndef CPU_ONLY
        deviceRescaleRhs(N, M, _h.data(), rhs.data());
#endif
    }

}
