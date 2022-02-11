#include <sstream>

template<class T>
DeviceArray<T>::DeviceArray(const MemoryManager memoryManager)
: DeviceDataDevice<T>(memoryManager), _size(0), _pointer(nullptr), _hasOwnMemoryManagement(false) {}

template<class T>
DeviceArray<T>::DeviceArray(const SizeType size, const T value, const MemoryManager memoryManager)
: DeviceDataDevice<T>(memoryManager), _size(size), _pointer(initializePointer()), _hasOwnMemoryManagement(true)
{
    initializeMemory(memoryManager, data(), this->size(), value);
}

template<class T>
DeviceArray<T>::DeviceArray(const DeviceArray &other)
: DeviceDataDevice<T>(other), _size(other.size()), _pointer(initializePointer()), _hasOwnMemoryManagement(true)
{
    this->_memoryManager->copy(data(), other.data(), this->byteSize());
}

template<class T>
void DeviceArray<T>::moveTo(const MemoryManager targetDevice)
{
    // only move if targetDevice is different to current device
    if (typeid(*(this->_memoryManager)) != typeid(*targetDevice))
    {
        T* newPtr = static_cast<T*>(this->_memoryManager->copyTo(_pointer.get(), this->byteSize(), targetDevice));
        this->_memoryManager = targetDevice;

        _hasOwnMemoryManagement = true;
        const auto currentManager = this->_memoryManager;
        _pointer = PointerType(newPtr, [currentManager](T* ptrToMemory){ currentManager->free(ptrToMemory); });
    }
}


template<class T>
DeviceArray<T>& DeviceArray<T>::operator= (const DeviceArray &other)
{
    if (this != &other)
    {
        if(hasOwnMemoryManagement())
        {
            resize(other.size());
        }
        else
        {
            assert(this->size() == other.size()
                    && "ERROR: Dimension mismatch. Cannot overwrite dependent DeviceArray.");
        }
        this->_memoryManager->copy(data(), other.data(), this->byteSize());
    }
    return *this;
}

template<class T>
DeviceArray<T>& DeviceArray<T>::operator=(DeviceArray&& other)
{
    if (isValid())
    {
        assert(this->size() == other.size()
                && "ERROR: Dimension mismatch. Cannot move to dependent DeviceArray.");
        this->_memoryManager->copy(data(), other.data(), this->byteSize());
        other._pointer = nullptr;
    }
    else
    {
        this->_size = other._size;
        this->_pointer = std::move(other._pointer);
        this->_hasOwnMemoryManagement = other._hasOwnMemoryManagement;
    }
    return *this;
}


template<class T>
T* DeviceArray<T>::data()
{
    return _pointer.get();
}

template<class T>
T const * DeviceArray<T>::data() const
{
    return _pointer.get();
}

template<class T>
T& DeviceArray<T>::operator[](const SizeType index)
{
    assert(index < size() && "ERROR: Index out of bounds.");
    return *(data() + index);
}

template<class T>
T const & DeviceArray<T>::operator[](const SizeType index) const
{
    assert(index < size() && "ERROR: Index out of bounds.");
    return *(data() + index);
}

template<class T>
DeviceArray<T> DeviceArray<T>::resize(SizeType newSize)
{
    if (size() != newSize)
    {
        _size = newSize;
        _pointer = initializePointer();
    }
    return (*this);
}

template<class T>
typename DeviceArray<T>::SizeType DeviceArray<T>::size() const
{
    return _size;
}

template<class T>
std::string DeviceArray<T>::display(const std::string name) const
{
    std::stringstream ss;
    ss << name << " = (" << std::endl;
    for (SizeType j = 0; j < size(); ++j)
        ss << std::setprecision(5) << (*this)[j] << "  ";
    ss << ")" << std::endl;
    return ss.str();
}

// private:

template<class T>
bool DeviceArray<T>::isValid()
{
    return (_pointer != nullptr);
}

template<class T>
typename DeviceArray<T>::PointerType DeviceArray<T>::initializePointer()
{
    if (this->byteSize() != 0)
    {
        T* pointer = static_cast<T*>(this->_memoryManager->allocate(this->byteSize()));
        _hasOwnMemoryManagement = true;
        const auto currentManager = this->_memoryManager;
        return PointerType(pointer, [currentManager](T* ptrToMemory){ currentManager->free(ptrToMemory); });
    }
    else
    {
       return nullptr;
    }
}

template<class T>
bool DeviceArray<T>::hasOwnMemoryManagement() const
{
    return _hasOwnMemoryManagement;
}

template<class T>
void DeviceArray<T>::makeDependentOn(const SizeType size, T * const pointer)
{
    //_pointer = PointerType(pointer, [](T* ptrToUnifiedMemory){ return; });
    _pointer = PointerType(pointer, [](T*){ return; }); // GH
    _size = size;
    _hasOwnMemoryManagement = false;
}
