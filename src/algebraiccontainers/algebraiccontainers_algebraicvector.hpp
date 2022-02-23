#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include "containerfactory.h"

template<class floating>
AlgebraicVector<floating> operator* (const floating scalar, const AlgebraicVector<floating> &B)
{
    AlgebraicVector<floating> tmp(B);
    tmp.scale(scalar);
    return tmp;
}

// GH
//template<class floating>
//AlgebraicVector<floating>&& operator* (const floating alpha, AlgebraicVector<floating> &&B)
//{
    //B.scale(alpha);
    //return B;
//}


// public:

template<class floating>
AlgebraicVector<floating>::AlgebraicVector(const ProcessingUnit<floating> processing_unit, ArrayPointerType data)
: _container_factory(processing_unit), _data(data)
{}

template<class floating>
AlgebraicVector<floating>::AlgebraicVector(AlgebraicVector<floating> const &other)
: _container_factory(other._container_factory), _data(initialize_pointer_by_copying(other))
{}

template<class floating>
AlgebraicVector<floating>& AlgebraicVector<floating>::move_to(const ProcessingUnit<floating> processing_unit)
{
    _container_factory = ContainerFactory<floating>(processing_unit);
    _data->move_to(processing_unit->get_memory_manager());
    return *this;
}

template<class floating>
AlgebraicVector<floating>& AlgebraicVector<floating>::operator=(const AlgebraicVector<floating> &other)
{
    assert(typeid(*this->get_processing_unit()) == typeid(*other.get_processing_unit()) && "Processing Units must be identical.");
    assert(is_valid() && "ERROR: Object not initialized");
    _container_factory = other._container_factory;
    access_array() = other.access_array();

    return *this;
}

template<class floating>
AlgebraicVector<floating>& AlgebraicVector<floating>::operator=(AlgebraicVector<floating> &&other)
{
    if (is_valid())
    {
        assert(typeid(*this->get_processing_unit()) == typeid(*other.get_processing_unit()) && "Processing Units must be identical.");
        access_array() = std::move(other.access_array());
        other._data = nullptr;
    }
    else
    {
        _container_factory = std::move(other._container_factory);
        _data = std::move(other._data);
    }
    return *this;
}

template<class floating>
floating* AlgebraicVector<floating>::data()
{
    return access_array().data();
}

template<class floating>
const floating* AlgebraicVector<floating>::data() const
{
    return access_array().data();
}

template<class floating>
floating& AlgebraicVector<floating>::operator[](const SizeType index)
{
#ifndef NDEBUG
    assert(typeid(*get_processing_unit()) == typeid(*std::make_shared<Cpu<floating>>()) && "Must be on Cpu");
#endif
    return access_array()[index];
}

template<class floating>
floating const & AlgebraicVector<floating>::operator[](const SizeType index) const
{
#ifndef NDEBUG
    assert(typeid(*get_processing_unit()) == typeid(*std::make_shared<Cpu<floating>>()) && "Must be on Cpu");
#endif
    return access_array()[index];
}

template<class floating>
typename AlgebraicVector<floating>::SizeType AlgebraicVector<floating>::size() const
{
    return access_array().size();
}


template<class floating>
ContainerFactory<floating> AlgebraicVector<floating>::get_container_factory() const
{
    ContainerFactory<floating> factory(get_processing_unit());
    return factory;
}

template<class floating>
ProcessingUnit<floating> AlgebraicVector<floating>::get_processing_unit() const
{
    return _container_factory.get_processing_unit();
}

template<class floating>
floating AlgebraicVector<floating>::get_euclidean_norm() const
{
    const floating sum = scalarProduct(*this, *this);
    return sqrt(sum);
}

template<class floating>
floating AlgebraicVector<floating>::get_maximum_norm() const
{
    const auto index = get_processing_unit()->ixamax(size(), data(), 1);
    DeviceScalar<floating> maximum(data()[index], get_processing_unit()->get_memory_manager());
    maximum.move_to(std::make_shared<CpuManager>());
    return maximum.value();
}

template<class floating>
std::string AlgebraicVector<floating>::display(std::string name) const
{
    return access_array().display(name);
}

template<class floating>
AlgebraicVector<floating>& AlgebraicVector<floating>::add(AlgebraicVector<floating> const &B, floating const scalar)
{
    assert( this->size() == B.size() ); // identical dimensions?
    assert(typeid(*this->get_processing_unit()) == typeid(*B.get_processing_unit()) && "Processing Units must be identical.");

    unsigned int const N = this->size();
    floating const * const b_arraystart = B.data();
    floating * const c_arraystart = this->data();
    unsigned int const inc = 1;

    get_processing_unit()->xaxpy(N, scalar, b_arraystart, inc, c_arraystart, inc);
    return *this;
}

template<class floating>
AlgebraicVector<floating> AlgebraicVector<floating>::operator+(const AlgebraicVector &B) const
{
    assert(typeid(*this->get_processing_unit()) == typeid(*B.get_processing_unit()) && "Processing Units must be identical.");
    auto tmp = *this;
    return tmp.add(B);
}

template<class floating>
AlgebraicVector<floating>& AlgebraicVector<floating>::operator+=(const AlgebraicVector &B)
{
    assert(typeid(*this->get_processing_unit()) == typeid(*B.get_processing_unit()) && "Processing Units must be identical.");
    return this->add(B);
}

template<class floating>
AlgebraicVector<floating> AlgebraicVector<floating>::operator-(const AlgebraicVector &B) const
{
    assert(typeid(*this->get_processing_unit()) == typeid(*B.get_processing_unit()) && "Processing Units must be identical.");
    auto tmp = *this;
    return tmp.add(B, -1.0);
}

template<class floating>
AlgebraicVector<floating> AlgebraicVector<floating>::operator*(const AlgebraicMatrix<floating> &A) const
{
    assert(typeid(*this->get_processing_unit()) == typeid(*A.get_processing_unit()) && "Processing Units must be identical.");
    auto res = *A.get_container_factory().create_array(A.get_num_cols());
    floating const *const pA = A.data();
    floating const *const pu = this->data();          // pointer to memory of input vector
    floating *const       pf = res.data();         // pointer to memory of output vector

    floating const alpha = 1.0;
    floating const beta = 0.0;
    unsigned int const M = A.get_num_rows();
    unsigned int const N = A.get_num_cols();
    unsigned int const LDA = M; // since transposed

    get_processing_unit()->xgemv(OperationType::Transposed, M, N, alpha, pA, LDA, pu, 1, beta, pf, 1);

    return res;
}

template<class floating>
void AlgebraicVector<floating>::scale(floating const scalar)
{
    unsigned int const N = this->size();
    floating* arraystart = this->data();
    unsigned int const incx = 1; // spacing between elements = 1

    get_processing_unit()->xscal(N, scalar, arraystart, incx);

    return;
}

// private:

template<class floating>
bool AlgebraicVector<floating>::is_valid() const
{
    return (_data != nullptr);
}

template<class floating>
typename AlgebraicVector<floating>::ArrayDataType& AlgebraicVector<floating>::access_array()
{
    assert(is_valid() && "ERROR: Object not initialized!");
    return *_data;
}

template<class floating>
typename AlgebraicVector<floating>::ArrayDataType const & AlgebraicVector<floating>::access_array() const
{
    assert(is_valid() && "ERROR: Object not initialized!");
    return *_data;
}

template<class floating>
typename AlgebraicVector<floating>::ArrayPointerType AlgebraicVector<floating>::initialize_pointer_by_copying(const AlgebraicVector<floating>& other) const
{
    assert(typeid(*this->get_processing_unit()) == typeid(*other.get_processing_unit()) && "Processing Units must be identical.");
    auto ptr = std::make_shared<DeviceArray<floating>>(other.access_array());
    return ptr;
}
