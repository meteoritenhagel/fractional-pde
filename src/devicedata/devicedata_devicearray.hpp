#include <sstream>

template<class T>
DeviceArray<T>::DeviceArray(const MemoryManager& memory_manager)
: DeviceDataDevice<T>(memory_manager), _size(0), _data(nullptr), _has_own_memory_management(false) {}

template<class T>
DeviceArray<T>::DeviceArray(const SizeType size, const T value, const MemoryManager& memory_manager)
: DeviceDataDevice<T>(memory_manager), _size(size), _data(initialize_data()), _has_own_memory_management(true)
{
    initialize_memory(memory_manager, data(), this->size(), value);
}

template<class T>
DeviceArray<T>::DeviceArray(const DeviceArray &other)
: DeviceDataDevice<T>(other), _size(other.size()), _data(initialize_data()), _has_own_memory_management(true)
{
    this->_memory_manager->copy(data(), other.data(), this->byte_size());
}

template<class T>
void DeviceArray<T>::move_to(const MemoryManager& target_device)
{
    // only move if target_device is different to current device
    if (typeid(*(this->_memory_manager)) != typeid(*target_device))
    {
        T* newPtr = static_cast<T*>(this->_memory_manager->copy_to(_data.get(), this->byte_size(), target_device));
        this->_memory_manager = target_device;

        _has_own_memory_management = true;
        const auto currentManager = this->_memory_manager;
        _data = PointerType(newPtr, [currentManager](T* ptrToMemory){ currentManager->free(ptrToMemory); });
    }
}


template<class T>
DeviceArray<T>& DeviceArray<T>::operator= (const DeviceArray &other)
{
    if (this != &other)
    {
        if(has_own_memory_management())
        {
            resize(other.size());
        }
        else
        {
            assert(this->size() == other.size()
                    && "ERROR: Dimension mismatch. Cannot overwrite dependent DeviceArray.");
        }
        this->_memory_manager->copy(data(), other.data(), this->byte_size());
    }
    return *this;
}

template<class T>
DeviceArray<T>& DeviceArray<T>::operator=(DeviceArray&& other)
{
    if (is_valid())
    {
        assert(this->size() == other.size()
                && "ERROR: Dimension mismatch. Cannot move to dependent DeviceArray.");
        this->_memory_manager->copy(data(), other.data(), this->byte_size());
        other._data = nullptr;
    }
    else
    {
        this->_size = other._size;
        this->_data = std::move(other._data);
        this->_has_own_memory_management = other._has_own_memory_management;
    }
    return *this;
}


template<class T>
T* DeviceArray<T>::data()
{
    return _data.get();
}

template<class T>
T const * DeviceArray<T>::data() const
{
    return _data.get();
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
DeviceArray<T> DeviceArray<T>::resize(SizeType new_size)
{
    if (size() != new_size)
    {
        _size = new_size;
        _data = initialize_data();
    }
    return (*this);
}

template<class T>
typename DeviceArray<T>::SizeType DeviceArray<T>::size() const
{
    return _size;
}

template<class T>
std::string DeviceArray<T>::display(const std::string& name) const
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
bool DeviceArray<T>::is_valid()
{
    return (_data != nullptr);
}

template<class T>
typename DeviceArray<T>::PointerType DeviceArray<T>::initialize_data()
{
    if (this->byte_size() != 0)
    {
        T* pointer = static_cast<T*>(this->_memory_manager->allocate(this->byte_size()));
        _has_own_memory_management = true;
        const auto currentManager = this->_memory_manager;
        return PointerType(pointer, [currentManager](T* ptrToMemory){ currentManager->free(ptrToMemory); });
    }
    else
    {
       return nullptr;
    }
}

template<class T>
bool DeviceArray<T>::has_own_memory_management() const
{
    return _has_own_memory_management;
}

template<class T>
void DeviceArray<T>::make_dependent_on(const SizeType size, T * const pointer)
{
    //_data = PointerType(pointer, [](T* ptrToUnifiedMemory){ return; });
    _data = PointerType(pointer, [](T*){ return; }); // GH
    _size = size;
    _has_own_memory_management = false;
}
