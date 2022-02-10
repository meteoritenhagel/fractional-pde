/*
 * unifieddata_unifiedmatrix.hpp
 *
 *  Created on: Jul 1, 2020
 *      Author: tristan
 */

#ifndef CPU_ONLY
#include "initializememory.cuh"
#endif

// public:

template<class T>
DeviceMatrix<T>::DeviceMatrix(const SizeType N, const SizeType M, const T value, const MemoryManager memoryManager)
: DeviceDataDevice<T>(memoryManager), _N(N), _M(M), _pointer(initializePointer()), _arrayOfPointers(initializeArray())
{
    initializeMemory(memoryManager, data(), this->size(), value);
}

template<class T>
DeviceMatrix<T>::DeviceMatrix(const SizeType N, std::vector<T> const &u, const MemoryManager memoryManager)
: DeviceDataDevice<T>(memoryManager), _N(N), _M(u.size() / N), _pointer(initializePointer()), _arrayOfPointers(initializeArray())
{
    assert(u.size() % N == 0);
    this->_memoryManager->copy(data(), u.data(), this->byteSize());
}

template<class T>
DeviceMatrix<T>::DeviceMatrix(const DeviceMatrix &other)
:  DeviceDataDevice<T>(other), _N(other._N), _M(other._M), _pointer(initializePointer()), _arrayOfPointers(initializeArray())
{
    this->_memoryManager->copy(data(), other.data(), this->byteSize());
}

template<class T>
DeviceMatrix<T>::DeviceMatrix(DeviceMatrix&& rhs)
: DeviceDataDevice<T>(rhs), _N(rhs._N), _M(rhs._M), _pointer(std::move(rhs._pointer)), _arrayOfPointers(std::move(rhs._arrayOfPointers))
{
    rhs._pointer = nullptr;
}

template<class T>
DeviceMatrix<T>& DeviceMatrix<T>::operator= (const DeviceMatrix &other)
{
    if (this != &other)
    {
        if(this->size() != other.size())
        {
            this->_N = other._N;
            this->_M = other._M;
            resetPointer();
        }
        this->DeviceDataDevice<T>::operator=(other);
        resetArray();
        this->_memoryManager->copy(data(), other.data(), this->byteSize());
    }
    return *this;
}

template<class T>
void DeviceMatrix<T>::moveTo(const MemoryManager targetDevice)
{
    //std::cout << "ByteSize before move: " << this->byteSize() << std::endl;
    // only move if targetDevice is different to current device
    if (typeid(*(this->_memoryManager)) != typeid(*targetDevice))
    {
        T* newPtr = static_cast<T*>(this->_memoryManager->copyTo(_pointer.get(), this->byteSize(), targetDevice));

        this->_memoryManager = targetDevice;

        const auto currentManager = this->_memoryManager;
        _pointer = PointerType(newPtr, [currentManager](T* ptrToMemory){ currentManager->free(ptrToMemory); });
        resetArray();
    }
}

template<class T>
DeviceMatrix<T>& DeviceMatrix<T>::operator=(DeviceMatrix&& rhs)
{
    this->_N = rhs._N;
    this->_M = rhs._M;

    if (this != &rhs)
    {
        _pointer = std::move(rhs._pointer);
        rhs._pointer = nullptr;
    }

    this->DeviceDataDevice<T>::operator=(std::move(rhs));

    // We have to initialize array again,
    // because trying to move rhs._arrayOfArrays to
    // this->_arrayOfArrays results results in deep copying!
    resetArray();
    return *this;
}

template<class T>
DeviceMatrix<T>& DeviceMatrix<T>::resize(const SizeType nrows, const SizeType ncols)
{
    // TODO: THIS IS POSSIBLY WRONG
    assert(nrows*ncols == getN() * getM() && "ERROR: Cannot resize. Dimension mismatch.");
    _N = nrows;
    _M = ncols;
    resetArray();
    
    return *this;
}

template<class T>
T* DeviceMatrix<T>::data()
{
    return _pointer.get();
}

template<class T>
T const * DeviceMatrix<T>::data() const
{
    return _pointer.get();
}

template<class T>
typename DeviceMatrix<T>::PointerToColumn DeviceMatrix<T>::getPointerToColumn(const SizeType m) const
{
    assert(m < getM() && "ERROR: Index out of bounds.");
    return _arrayOfPointers[m];
}

template<class T>
DeviceArray<T>& DeviceMatrix<T>::operator[](const SizeType m)
{
    //assert(m < getM() && "ERROR: Index out of bounds.");
    return *getPointerToColumn(m);
}

template<class T>
DeviceArray<T> const & DeviceMatrix<T>::operator[](const SizeType m) const
{
    //assert(m < getM() && "ERROR: Index out of bounds.");
    return *getPointerToColumn(m);
}

template<class T>
T& DeviceMatrix<T>::operator()(const SizeType i, const SizeType j)
{
    return (*this)[j][i];
}

template<class T>
T const & DeviceMatrix<T>::operator()(const SizeType i, const SizeType j) const
{
    return (*this)[j][i];
}

template<class T>
typename DeviceMatrix<T>::SizeType DeviceMatrix<T>::size() const
{
    return _N * _M;
}

template<class T>
typename DeviceMatrix<T>::SizeType DeviceMatrix<T>::getN() const
{
    return _N;
}

template<class T>
typename DeviceMatrix<T>::SizeType DeviceMatrix<T>::getM() const
{
    return _M;
}

template<class T>
bool DeviceMatrix<T>::isSquare() const
{
    return (getN() == getM());
}

template<class T>
void DeviceMatrix<T>::display(const std::string name) const
{
    std::cout << name << " = (" << std::endl;
    for (SizeType i = 0; i < getN(); ++i)
    {
        for (SizeType j = 0; j < getM(); ++j)
            std::cout << std::setprecision(5) << (*this)(i,j) << "  ";
        std::cout << std::endl;
    }
    std::cout << ")" << std::endl;
    return;
}

// private:

template<class T>
typename DeviceMatrix<T>::PointerType DeviceMatrix<T>::initializePointer()
{
    if (this->byteSize() != 0)
    {
        T* pointerToUnifiedMemory = static_cast<T*>(this->_memoryManager->allocate(this->byteSize()));
        const auto currentManager = this->_memoryManager;
        return PointerType(pointerToUnifiedMemory, [currentManager](T* ptrToUnifiedMemory){ currentManager->free(ptrToUnifiedMemory); });
    }
   else
    {
       return nullptr;
    }
}

template<class T>
typename DeviceMatrix<T>::ArrayOfPointers DeviceMatrix<T>::initializeArray()
{
    ArrayOfPointers arrayOfPointers(getM());
    for (SizeType i = 0; i < getM(); ++i)
    {
        arrayOfPointers[i] = std::make_shared<DeviceArray<T>>(this->_memoryManager);
        arrayOfPointers[i]->makeDependentOn(getN(), data()+i*getN());
    }
    return arrayOfPointers;
}

template<class T>
void DeviceMatrix<T>::resetArray()
{
    _arrayOfPointers = initializeArray();
}

template<class T>
void DeviceMatrix<T>::resetPointer()
{
    _pointer = initializePointer();
}
