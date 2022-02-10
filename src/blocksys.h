#ifndef FILE_BLOCKSYS
#define FILE_BLOCKSYS

#include "algebraiccontainers/algebraiccontainers.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <vector>

#include "auxiliary.h"
#include "algebraiccontainers/containerfactory.h"

enum class SolvingProcedure
{
    CyclicReduction,
    PCBiCGStab,
    BiCGStab,
    Richardson,
    PCRichardson
};

/** 	Matrix with block tridiagonal structure.
 *      Each block has the same dimension and it is a dense matrix.
 *
 *      |    M   -B   0   ...           0   |
 *      |   -B    A  -B                 .   |
 *      |    0   -B   A   -B                |
 *      |    .        .    .   .            |
 *      |    .             .   .   .        |
 *      |                     -B   A   -B   |
 *      |                          -B   M   |
 *
 *      @f$ \begin{pmatrix}
 *          M &&&& \\ -B & A &-B \\  & -B & A &-B \\ && -B & A &-B \\  &&&& M
 *         \end{pmatrix}_{n_b\times n_b} @f$
 *
 *      Matrix A is computed via A:=2M-h^2 g D  with dense matrix D
 *
 */
// For matrix in Latex, see https://stackoverflow.com/questions/25833856/doxygen-latex-output-using-mathjax-to-create-a-matrix-definition-produces-an-err
// In 'Doxyfile':  EXTRA_PACKAGES         = amsmath
template<class floating>
class EquidistantBlock_1D// : public BlockSys<floating>
{
public:
    using SizeType = typename AlgebraicMatrix<floating>::SizeType;

    /** 	Constructor. A and B have to be square with same dimensions.
     *
     * 	@param[in] bdim  number of row respectively column blocks
     * 	@param[in] M  matrix regarding Dirichlet conditions for all time steps
     * 	@param[in] A  matrix for diagonal blocks
     * 	@param[in] B  matrix for off diagonal blocks
     *
     */
    EquidistantBlock_1D(const SizeType bdim,
                        const AlgebraicMatrix<floating> &B, const AlgebraicMatrix<floating> &D, const AlgebraicMatrix<floating> &M, const floating h,
                        const floating alpha, const floating timeGridStepSize, const ProcessingUnit<floating> processingUnit);

    /** 	Constructor. A and B have to be square with same dimensions.
     *
     * 	@param[in] bdim  number of row respectively column blocks
     * 	@param[in] M  matrix regarding Dirichlet conditions for all time steps
     * 	@param[in] A  matrix for diagonal blocks
     * 	@param[in] B  matrix for off diagonal blocks
     *
     */
    EquidistantBlock_1D(const SizeType bdim,
                        const AlgebraicMatrix<floating> &B, const CoefficientMatrix<floating> &D, const AlgebraicMatrix<floating> &M, const floating h,
                        const floating alpha, const floating timeGridStepSize, const ProcessingUnit<floating> processingUnit);
    ~EquidistantBlock_1D() = default;

    /** 	Matrix row dimension.
     * @return Dimension of each block.
     */
    SizeType getNdim() const;

    /** 	Block row dimension.
     * @return Number of block rows.
     */
    SizeType getBlockDim() const;

    /** 	Dimension of square matrix which is obtained by copying the block
     *      matrix to a dense AlgebraicMatrix.
     * @return dimension.
     */
    SizeType getDenseDim() const;

    ContainerFactory<floating> getMatrixFactory() const;

    /** 	Copies the block matrix to a dense matrix.
     *
     * @return Dense matrix representation stored columnwise.
     */
    AlgebraicMatrix<floating> copyToDense() const;

    BlockVector<floating> solve(const BlockVector<floating> &rhs) const;
    BlockVector<floating> solve(const BlockVector<floating> &rhs, const size_t maxNumberOfIterations, const size_t stepsPerIteration, const floating accuracy, const SolvingProcedure solvingProcedure) const;


private:
    SizeType _bdim; //!< number of block rows and block columns
    AlgebraicMatrix<floating> _B;
    AlgebraicMatrix<floating> _D;
    AlgebraicMatrix<floating> _M;
    floating _alpha;
    floating _timeGridStepSize;
    floating _h;
    AlgebraicMatrix<floating> _A;
    ContainerFactory<floating> _colMatrixFactory; //!< factory for creating AlgebraicMatrix using suitable processing unit (i.e. CPU or GPU)
    std::unique_ptr<EquidistantBlock_1D<floating>> _coarseSystemPtr;

    // TODO: CHANGE
public:
    EquidistantBlock_1D(const SizeType bdim,
                        const AlgebraicMatrix<floating> &B, const AlgebraicMatrix<floating> &D, const AlgebraicMatrix<floating> &M,
                        const floating h, const floating alpha, const floating timeGridStepSize,
                        const AlgebraicMatrix<floating> &A, const ProcessingUnit<floating> processingUnit);

private:

    floating getSystemCoeff(const floating alpha, const floating timeGridStepSize) const;
    AlgebraicMatrix<floating> initializeA(const AlgebraicMatrix<floating> &B, const AlgebraicMatrix<floating> &D, const AlgebraicMatrix<floating> &M, const floating h, const floating alpha, const floating timeGridStepSize) const;
    ProcessingUnit<floating> getProcessingUnit() const;

    /** 	Solves the current system using cyclic reduction with RHS f.
     *
     * @return solution.
     */
    BlockVector<floating> cyclicReduction(const AlgebraicMatrix<floating> &scaledB, const BlockVector<floating> &f) const;

    /**  prolongates coarse solution vector (matrix) @p uc to the fine vector.
     *
     * The restriction uses the simple 1D formula
     * @f$ u_f[2*i-1] =  A^{-1}\cdot \left({ f_f[2i-1] + B\cdot (u_c[2i-2]+u_c[2i])  }\right)\enspace,@f$
     * and the simple copy  @f$ u_f[2*i] = u_c[2*i]@f$ for fine nodes with coarse indeces.
     *
     * @param[in] ff rhs vector (matrix) on fine grid
     * @param[in] uc solution vector (matrix) on coarse grid
     *
     * @return fine vector (matrix) @p uf.
     */
    AlgebraicMatrix<floating> CRProlongation(const AlgebraicMatrix<floating> &scaledB, const AlgebraicMatrix<floating> &ff, const AlgebraicMatrix<floating> &uc) const;

    /**  Restricts fine rhs vector (matrix) @p ff to the coarse vector.
     *
     * The restriction uses the simple 1D formula
     * @f$ f_c[i] = f_f[2i]+B\cdot A^{-1}\cdot (f_f[2i-1]+f_f[2i+1])@f$  and b.c.
     *
     * @param[in] ff vector (matrix) on fine grid
     *
     * @return coarse rhs vector (matrix) @p fc.
     */
    // fc[i/2] = fi[i]+B*A^(-1)*(fi[i-1]+fi[i+1]) and b.c.
    AlgebraicMatrix<floating> CRRestriction(const AlgebraicMatrix<floating> &scaledB, const AlgebraicMatrix<floating> &ff) const;

    floating multigrid(const unsigned numberOfSmoothingSteps, const BlockVector<floating> &f, const size_t maxNumberOfIterations, const floating accuracy, BlockVector<floating> &solution) const;
    BlockVector<floating> prolongation(const BlockVector<floating> &ff) const;
    BlockVector<floating> restriction(const BlockVector<floating> &ff) const;
    floating getReducedGrid() const;
    SizeType getCoarseDim() const;
    std::unique_ptr<EquidistantBlock_1D<floating>> initializeCoarseSystemPtr() const;
    const EquidistantBlock_1D<floating>& getCoarseSystem() const;
    floating jacobiIteration(const floating omega, const BlockVector<floating> &f, const size_t maxNumberOfIterations, const floating accuracy, BlockVector<floating> &solution) const;
    void smooth(const floating omega, const BlockVector<floating> &f, const size_t maxNumberOfIterations, BlockVector<floating> &solution) const;

    BlockVector<floating> calculateResidual(const BlockVector<floating> &u, const BlockVector<floating> &f) const;
    AlgebraicVector<floating> calculateRowResidual(const SizeType i, const BlockVector<floating> &u, const BlockVector<floating> &f) const;

    // TODO: REMOVE?
    BlockVector<floating> rescale_rhs(const BlockVector<floating> &rhs) const;
};

template<class floating>
class NonEquidistantBlock_1D
{
public:
    using SizeType = typename AlgebraicMatrix<floating>::SizeType;

    NonEquidistantBlock_1D(const SizeType bdim,
                           const AlgebraicMatrix<floating> &B, const CoefficientMatrix<floating> &D,
                           const AlgebraicMatrix<floating> &M,
                           const AlgebraicVector<floating> &h, const floating alpha, const floating timeGridStepSize,
                           const ProcessingUnit<floating> processingUnit);

    NonEquidistantBlock_1D(const SizeType bdim,
                           const AlgebraicMatrix<floating> &B, const AlgebraicMatrix<floating> &D, const AlgebraicMatrix<floating> &M,
                           const AlgebraicVector<floating> &h, const floating alpha, const floating timeGridStepSize,
                           const ProcessingUnit<floating> processingUnit);
    ~NonEquidistantBlock_1D() = default;

    SizeType getNdim() const;
    SizeType getBlockDim() const;
    SizeType getDenseDim() const;

    ContainerFactory<floating> getMatrixFactory() const;

    /** 	Copies the block matrix to a dense matrix.
     *
     * @return Dense matrix representation stored columnwise.
     */
    AlgebraicMatrix<floating> copyToDense() const;

    void mult(const BlockVector<floating> &u, BlockVector<floating> &result) const;

    BlockVector<floating> solve(BlockVector<floating> &rhs, const size_t maxNumberOfIterations, const size_t stepsPerIteration, const floating accuracy, const SolvingProcedure solvingProcedure = SolvingProcedure::PCBiCGStab) const;

private:
    SizeType _bdim;
    AlgebraicMatrix<floating> _B;
    AlgebraicMatrix<floating> _D;
    AlgebraicMatrix<floating> _M;
    floating _alpha;
    floating _timeGridStepSize;
    mutable AlgebraicVector<floating> _h; // TODO: REMOVE MUTABLE
    AlgebraicMatrix<floating> _C;
    ContainerFactory<floating> _colMatrixFactory;
    std::unique_ptr<NonEquidistantBlock_1D<floating>> _coarseSystemPtr;
    mutable AlgebraicMatrix<floating> _vectorBuffer;
    mutable AlgebraicVector<floating> _host_h;
    mutable AlgebraicMatrix<floating> _buffer1;
    mutable AlgebraicMatrix<floating> _buffer2;
    mutable AlgebraicMatrix<floating> _buffer3;
    mutable AlgebraicMatrix<floating> _buffer4;
    mutable AlgebraicMatrix<floating> _buffer5;
    mutable AlgebraicMatrix<floating> _buffer6;
    mutable AlgebraicMatrix<floating> _coarseBuffer1;
    mutable AlgebraicMatrix<floating> _coarseBuffer2;
    mutable AlgebraicVector<floating> _fineGridBuffer;

    floating biCGStab(const BlockVector<floating> &f, const size_t maxNumberOfIterations, const floating accuracy, BlockVector<floating> &solution) const;
    floating PCbiCGStab(const BlockVector<floating> &f, const size_t maxNumberOfIterations, const floating accuracy, BlockVector<floating> &solution) const;

    void multigrid(const unsigned numberOfSmoothingSteps, const BlockVector<floating> &f, const size_t maxNumberOfIterations, const floating accuracy, BlockVector<floating> &solution) const;
    void prolongation(const BlockVector<floating> &ff, BlockVector<floating> &solution) const;
    void restriction(const BlockVector<floating> &ff, BlockVector<floating> &ffOnCoarseGrid) const;
    AlgebraicVector<floating> getReducedGrid() const;
    SizeType getCoarseDim() const;
    std::unique_ptr<NonEquidistantBlock_1D<floating>> initializeCoarseSystemPtr() const;
    const NonEquidistantBlock_1D<floating>& getCoarseSystem() const;
    floating jacobiIteration(const floating omega, const BlockVector<floating> &f, const size_t maxNumberOfIterations, const floating accuracy, BlockVector<floating> &solution) const;
    void smooth(const floating omega, const BlockVector<floating> &f, const size_t maxNumberOfIterations, BlockVector<floating> &solution) const;

    void calculateResidual(const BlockVector<floating> &u, const BlockVector<floating> &f, BlockVector<floating> &residual) const;
    void calculateRowResidual(const SizeType i, const BlockVector<floating> &u, const BlockVector<floating> &f, AlgebraicVector<floating> &residual) const;
    floating getSystemCoeff() const;
    AlgebraicMatrix<floating> initializeC() const;
    ProcessingUnit<floating> getProcessingUnit() const;

    floating get_h_preconditioner(const AlgebraicVector<floating> &h) const;

    NonEquidistantBlock_1D() = default;

    // Todo: change?
public:
    void rescale_rhs(BlockVector<floating> &rhs) const;
};

#include "blocksys_equidistant.hpp"
#include "blocksys_nonequidistant.hpp"
#endif
