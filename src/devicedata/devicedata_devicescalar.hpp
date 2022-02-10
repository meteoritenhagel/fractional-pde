/*
 * unifieddata_unifiedscalar.hpp
 *
 *  Created on: Jul 1, 2020
 *      Author: tristan
 */

// public:

template<class T>
DeviceScalar<T>::DeviceScalar(const T &value, const MemoryManager memoryManager)
: DeviceDataDevice<T>(memoryManager), _pointer(initializePointer(value))
{}

template<class T>
DeviceScalar<T>::DeviceScalar(DeviceScalar const &other)
: _pointer(initializePointer())
{
    this->memoryManager->copy(data(), other.data(), this->byteSize());
}

template<class T>
void DeviceScalar<T>::moveTo(const MemoryManager targetDevice)
{
    // only move if targetDevice is different to current device
    if (typeid(*(this->_memoryManager)) != typeid(*targetDevice))
    {
        T* newPtr = static_cast<T*>(this->_memoryManager->copyTo(data(), this->byteSize(), targetDevice));
        this->_memoryManager = targetDevice;

        const auto currentManager = this->_memoryManager;
        _pointer = PointerType(newPtr, [currentManager](T* ptrToMemory){ currentManager->free(ptrToMemory); });
    }
}

template<class T>
DeviceScalar<T>::DeviceScalar(DeviceScalar&& rhs)
: _pointer(rhs._pointer)
{
    rhs._pointer = nullptr;
}

template<class T>
DeviceScalar<T>& DeviceScalar<T>::operator= (DeviceScalar const &other)
{
    if (this != &other)
    {
        this->_memoryManager->copy(data(), other.data(), this->byteSize());
    }
    return *this;
}

template<class T>
DeviceScalar<T>& DeviceScalar<T>::operator=(DeviceScalar&& other)
{
    this->_memoryManager->copy(_pointer, other._pointer, this->byteSize());
    other._pointer = nullptr;
    return *this;
}

template<class T>
T* DeviceScalar<T>::data()
{
    return _pointer.get();
}

template<class T>
T const * DeviceScalar<T>::data() const
{
    return _pointer.get();
}

template<class T>
T DeviceScalar<T>::value() const
{
    return *_pointer;
}

template<class T>
typename DeviceScalar<T>::SizeType DeviceScalar<T>::size() const
{
    return 1;
}

template<class T>
void DeviceScalar<T>::display(const std::string name) const
{
    std::cout << name << " = " << value() << std::endl;
    return;
}

// private:

template<class T>
typename DeviceScalar<T>::PointerType DeviceScalar<T>::initializePointer(const T &value)
{
    T* pointerToMemory = static_cast<T*>(this->_memoryManager->allocate(this->byteSize()));
    this->_memoryManager->copy(pointerToMemory, &value, this->byteSize());
    const auto currentManager = this->_memoryManager;
    return PointerType(pointerToMemory, [currentManager](T* ptrToMemory){ currentManager->free(ptrToMemory); });
}

