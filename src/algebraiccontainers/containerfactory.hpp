#include "containerfactory.h"

#include <array>
#include <vector>

template<class floating>
ContainerFactory<floating>::ContainerFactory(const ProcessingUnit<floating> processingUnit)
: _processingUnit(processingUnit)
{}

template<class floating>
typename ContainerFactory<floating>::ColMatrixPointer ContainerFactory<floating>::createMatrix() const
{
    DeviceMatrix<floating> A(0, 0, 0, _processingUnit->get_memory_manager());
    auto ptr = std::make_unique<AlgebraicMatrix<floating>>(_processingUnit, A);
    return ptr;
}

template<class floating>
typename ContainerFactory<floating>::ColMatrixPointer ContainerFactory<floating>::createMatrix(
        const SizeType nrow, const SizeType ncol, const floating val) const
{
    DeviceMatrix<floating> A(nrow, ncol, val, _processingUnit->get_memory_manager());
    auto ptr = std::make_unique<AlgebraicMatrix<floating>>(_processingUnit, A);
    return ptr;
}

template<class floating>
typename ContainerFactory<floating>::ColMatrixPointer ContainerFactory<floating>::createMatrix(
        const SizeType nrow, std::vector<floating> const &u) const
{
    DeviceMatrix<floating> A(nrow, u);
    A.moveTo(_processingUnit->get_memory_manager());
    auto ptr = std::make_unique<AlgebraicMatrix<floating>>(_processingUnit, A);
    return ptr;
}


template<class floating>
typename ContainerFactory<floating>::ColMatrixColumnPointer ContainerFactory<floating>::createColumn(const SizeType size, const floating val) const
{
    auto arrayPointer = std::make_shared<DeviceArray<floating>>(size, val, _processingUnit->get_memory_manager());
    auto ptr = std::make_shared<AlgebraicVector<floating>>(_processingUnit, arrayPointer);
    return ptr;
}

template<class floating>
typename ContainerFactory<floating>::CoefficientMatrixPointer ContainerFactory<floating>::createCoefficientMatrix(const SizeType size, const floating alpha) const
{
    auto ptr = std::make_unique<CoefficientMatrix<floating>>(_processingUnit, size, alpha);
    return ptr;
}
template<class floating>
ProcessingUnit<floating> ContainerFactory<floating>::getProcessingUnit() const
{
    return _processingUnit;
}
