#include "containerfactory.h"

#include <array>
#include <vector>

template<class floating>
ContainerFactory<floating>::ContainerFactory(const ProcessingUnit<floating> processing_unit)
: _processing_unit(processing_unit)
{}

template<class floating>
typename ContainerFactory<floating>::ColMatrixPointer ContainerFactory<floating>::create_matrix() const
{
    DeviceMatrix<floating> A(0, 0, 0, _processing_unit->get_memory_manager());
    auto ptr = std::make_unique<AlgebraicMatrix<floating>>(_processing_unit, A);
    return ptr;
}

template<class floating>
typename ContainerFactory<floating>::ColMatrixPointer ContainerFactory<floating>::create_matrix(
        const SizeType num_rows, const SizeType num_cols, const floating val) const
{
    DeviceMatrix<floating> A(num_rows, num_cols, val, _processing_unit->get_memory_manager());
    auto ptr = std::make_unique<AlgebraicMatrix<floating>>(_processing_unit, A);
    return ptr;
}

template<class floating>
typename ContainerFactory<floating>::ColMatrixPointer ContainerFactory<floating>::create_matrix(
        const SizeType num_rows, std::vector<floating> const &vec) const
{
    DeviceMatrix<floating> A(num_rows, vec);
    A.move_to(_processing_unit->get_memory_manager());
    auto ptr = std::make_unique<AlgebraicMatrix<floating>>(_processing_unit, A);
    return ptr;
}


template<class floating>
typename ContainerFactory<floating>::ColMatrixColumnPointer ContainerFactory<floating>::create_array(const SizeType size, const floating val) const
{
    auto arrayPointer = std::make_shared<DeviceArray<floating>>(size, val, _processing_unit->get_memory_manager());
    auto ptr = std::make_shared<AlgebraicVector<floating>>(_processing_unit, arrayPointer);
    return ptr;
}

template<class floating>
typename ContainerFactory<floating>::CoefficientMatrixPointer ContainerFactory<floating>::create_coefficient_matrix(const SizeType N, const floating alpha) const
{
    auto ptr = std::make_unique<CoefficientMatrix<floating>>(_processing_unit, N, alpha);
    return ptr;
}
template<class floating>
ProcessingUnit<floating> ContainerFactory<floating>::get_processing_unit() const
{
    return _processing_unit;
}
