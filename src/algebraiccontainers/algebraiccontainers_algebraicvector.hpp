#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include "containerfactory.h"

template<class floating>
AlgebraicVector<floating> operator* (const floating alpha, const AlgebraicVector<floating> &B)
{
    AlgebraicVector<floating> tmp(B);
    tmp.scale(alpha);
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
AlgebraicVector<floating>::AlgebraicVector(const ProcessingUnit<floating> processingUnit, ArrayPointerType A)
: _colMatrixFactory(processingUnit), _Aptr(A)
{}

template<class floating>
AlgebraicVector<floating>::AlgebraicVector(AlgebraicVector<floating> const &other)
: _colMatrixFactory(other._colMatrixFactory), _Aptr(initializePointerCopying(other))
{}

template<class floating>
AlgebraicVector<floating>& AlgebraicVector<floating>::moveTo(const ProcessingUnit<floating> processingUnit)
{
    _colMatrixFactory = ContainerFactory<floating>(processingUnit);
    _Aptr->moveTo(processingUnit->getMemoryManager());
    return *this;
}

template<class floating>
AlgebraicVector<floating>& AlgebraicVector<floating>::operator=(const AlgebraicVector<floating> &other)
{
    assert(typeid(*this->get_processing_unit()) == typeid(*other.get_processing_unit()) && "Processing Units must be identical.");
    assert(isValid() && "ERROR: Object not initialized");
    _colMatrixFactory = other._colMatrixFactory;
    accessArray() = other.accessArray();

    return *this;
}

template<class floating>
AlgebraicVector<floating>& AlgebraicVector<floating>::operator=(AlgebraicVector<floating> &&other)
{
    if (isValid())
    {
        assert(typeid(*this->get_processing_unit()) == typeid(*other.get_processing_unit()) && "Processing Units must be identical.");
        accessArray() = std::move(other.accessArray());
        other._Aptr = nullptr;
    }
    else
    {
        _colMatrixFactory = std::move(other._colMatrixFactory);
        _Aptr = std::move(other._Aptr);
    }
    return *this;
}

template<class floating>
floating* AlgebraicVector<floating>::data()
{
    return accessArray().data();
}

template<class floating>
const floating* AlgebraicVector<floating>::data() const
{
    return accessArray().data();
}

template<class floating>
floating& AlgebraicVector<floating>::operator[](const SizeType col)
{
#ifndef NDEBUG
    assert(typeid(*get_processing_unit()) == typeid(*std::make_shared<CPU<floating>>()) && "Must be on CPU");
#endif
    return accessArray()[col];
}

template<class floating>
floating const & AlgebraicVector<floating>::operator[](const SizeType col) const
{
#ifndef NDEBUG
    assert(typeid(*get_processing_unit()) == typeid(*std::make_shared<CPU<floating>>()) && "Must be on CPU");
#endif
    return accessArray()[col];
}

template<class floating>
typename AlgebraicVector<floating>::SizeType AlgebraicVector<floating>::size() const
{
    return accessArray().size();
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
    return _colMatrixFactory.getProcessingUnit();
}

template<class floating>
floating AlgebraicVector<floating>::getEuclidean() const
{
    const floating sum = scalarProduct(*this, *this);
    return sqrt(sum);
}

template<class floating>
floating AlgebraicVector<floating>::getMaximum() const
{
    const auto index = get_processing_unit()->ixamax(size(), data(), 1);
    DeviceScalar<floating> maximum(data()[index], get_processing_unit()->getMemoryManager());
    maximum.moveTo(std::make_shared<CPU_Manager>());
    return maximum.value();
}

template<class floating>
std::string AlgebraicVector<floating>::display(std::string name) const
{
    return accessArray().display(name);
}

template<class floating>
AlgebraicVector<floating>& AlgebraicVector<floating>::add(AlgebraicVector<floating> const &B, floating const alpha)
{
    assert( this->size() == B.size() ); // identical dimensions?
    assert(typeid(*this->get_processing_unit()) == typeid(*B.get_processing_unit()) && "Processing Units must be identical.");

    unsigned int const N = this->size();
    floating const * const b_arraystart = B.data();
    floating * const c_arraystart = this->data();
    unsigned int const inc = 1;

    get_processing_unit()->xaxpy(N, alpha, b_arraystart, inc, c_arraystart, inc);
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
    auto res = *A.get_container_factory().createColumn(A.getNcols());
    floating const *const pA = A.data();
    floating const *const pu = this->data();          // pointer to memory of input vector
    floating *const       pf = res.data();         // pointer to memory of output vector

    floating const alpha = 1.0;
    floating const beta = 0.0;
    unsigned int const M = A.getNrows();
    unsigned int const N = A.getNcols();
    unsigned int const LDA = M; // since transposed

    get_processing_unit()->xgemv(OperationType::Transposed, M, N, alpha, pA, LDA, pu, 1, beta, pf, 1);

    return res;
}

template<class floating>
void AlgebraicVector<floating>::scale(floating const alpha)
{
    unsigned int const N = this->size();
    floating* arraystart = this->data();
    unsigned int const incx = 1; // spacing between elements = 1

    get_processing_unit()->xscal(N, alpha, arraystart, incx);

    return;
}

// private:

template<class floating>
bool AlgebraicVector<floating>::isValid() const
{
    return (_Aptr != nullptr);
}

template<class floating>
typename AlgebraicVector<floating>::ArrayDataType& AlgebraicVector<floating>::accessArray()
{
    assert(isValid() && "ERROR: Object not initialized!");
    return *_Aptr;
}

template<class floating>
typename AlgebraicVector<floating>::ArrayDataType const & AlgebraicVector<floating>::accessArray() const
{
    assert(isValid() && "ERROR: Object not initialized!");
    return *_Aptr;
}

template<class floating>
typename AlgebraicVector<floating>::ArrayPointerType AlgebraicVector<floating>::initializePointerCopying(const AlgebraicVector<floating>& other) const
{
    assert(typeid(*this->get_processing_unit()) == typeid(*other.get_processing_unit()) && "Processing Units must be identical.");
    auto ptr = std::make_shared<DeviceArray<floating>>(other.accessArray());
    return ptr;
}
