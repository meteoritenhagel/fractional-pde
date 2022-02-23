#include <sstream>

template<class T>
DeviceScalar<T>::DeviceScalar(const T &value, const MemoryManager& memory_manager)
: DeviceDataDevice<T>(memory_manager), _data(initialize_data(value))
{}

template<class T>
DeviceScalar<T>::DeviceScalar(DeviceScalar const &other)
: DeviceDataDevice<T>(other._memory_manager), _data(initialize_data(0))
{
    this->_memory_manager->copy(data(), other.data(), this->byte_size());
}

template<class T>
void DeviceScalar<T>::move_to(const MemoryManager& target_device)
{
    // only move if target_device is different to current device
    if (typeid(*(this->_memory_manager)) != typeid(*target_device))
    {
        T* newPtr = static_cast<T*>(this->_memory_manager->copy_to(data(), this->byte_size(), target_device));
        this->_memory_manager = target_device;

        const auto currentManager = this->_memory_manager;
        _data = PointerType(newPtr, [currentManager](T* ptrToMemory){ currentManager->free(ptrToMemory); });
    }
}

template<class T>
DeviceScalar<T>::DeviceScalar(DeviceScalar&& rhs)
: _data(rhs._data)
{
    rhs._data = nullptr;
}

template<class T>
DeviceScalar<T>& DeviceScalar<T>::operator= (DeviceScalar const &other)
{
    if (this != &other)
    {
        this->_memory_manager->copy(data(), other.data(), this->byte_size());
    }
    return *this;
}

template<class T>
DeviceScalar<T>& DeviceScalar<T>::operator=(DeviceScalar&& other)
{
    this->_memory_manager->copy(_data, other._data, this->byte_size());
    other._data = nullptr;
    return *this;
}

template<class T>
T* DeviceScalar<T>::data()
{
    return _data.get();
}

template<class T>
T const * DeviceScalar<T>::data() const
{
    return _data.get();
}

template<class T>
T DeviceScalar<T>::value() const
{
    DeviceScalar<T> temp(*this);
    temp.move_to(std::make_unique<CpuManager>());

    return *temp._data;
}

template<class T>
typename DeviceScalar<T>::SizeType DeviceScalar<T>::size() const
{
    return 1;
}

template<class T>
std::string DeviceScalar<T>::display(const std::string& name) const
{
    std::stringstream ss;
    ss << name << " = " << value() << std::endl;
    return ss.str();
}

template<class T>
typename DeviceScalar<T>::PointerType DeviceScalar<T>::initialize_data(const T &value)
{
    T* pointerToMemory = static_cast<T*>(this->_memory_manager->allocate(this->byte_size()));
    this->_memory_manager->copy(pointerToMemory, &value, this->byte_size());
    const auto currentManager = this->_memory_manager;
    return PointerType(pointerToMemory, [currentManager](T* ptrToMemory){ currentManager->free(ptrToMemory); });
}

