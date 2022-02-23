#ifndef CPU_ONLY
#include "initializememory.cuh"
#endif

#include <sstream>

template<class T>
DeviceMatrix<T>::DeviceMatrix(const SizeType num_rows, const SizeType num_cols, const T value, const MemoryManager& memory_manager)
: DeviceDataDevice<T>(memory_manager), _num_rows(num_rows), _num_cols(num_cols), _data(initialize_data()), _array_of_columns(
        initialize_array_of_columns())
{
    initialize_memory(memory_manager, data(), this->size(), value);
}

template<class T>
DeviceMatrix<T>::DeviceMatrix(const SizeType num_rows, std::vector<T> const &vec)
: DeviceDataDevice<T>(std::make_shared<CpuManager>()), _num_rows(num_rows), _num_cols(vec.size() / num_rows), _data(
        initialize_data()), _array_of_columns(initialize_array_of_columns())
{
    assert(vec.size() % num_rows == 0);
    this->_memory_manager->copy(data(), vec.data(), this->byte_size());
}

template<class T>
DeviceMatrix<T>::DeviceMatrix(const DeviceMatrix &other)
:  DeviceDataDevice<T>(other), _num_rows(other._num_rows), _num_cols(other._num_cols), _data(initialize_data()), _array_of_columns(
        initialize_array_of_columns())
{
    this->_memory_manager->copy(data(), other.data(), this->byte_size());
}

template<class T>
DeviceMatrix<T>::DeviceMatrix(DeviceMatrix&& rhs)
: DeviceDataDevice<T>(rhs), _num_rows(rhs._num_rows), _num_cols(rhs._num_cols), _data(std::move(rhs._data)), _array_of_columns(std::move(rhs._array_of_columns))
{
    rhs._data = nullptr;
}

template<class T>
DeviceMatrix<T>& DeviceMatrix<T>::operator= (const DeviceMatrix &other)
{
    if (this != &other)
    {
        if(this->size() != other.size())
        {
            this->_num_rows = other._num_rows;
            this->_num_cols = other._num_cols;
            reset_data();
        }
        this->DeviceDataDevice<T>::operator=(other);
        reset_array_of_columns();
        this->_memory_manager->copy(data(), other.data(), this->byte_size());
    }
    return *this;
}

template<class T>
void DeviceMatrix<T>::move_to(const MemoryManager& target_device)
{
    // only move if target_device is different to current device
    if (typeid(*(this->_memory_manager)) != typeid(*target_device))
    {
        T* newPtr = static_cast<T*>(this->_memory_manager->copy_to(_data.get(), this->byte_size(), target_device));

        this->_memory_manager = target_device;

        const auto currentManager = this->_memory_manager;
        _data = PointerType(newPtr, [currentManager](T* ptrToMemory){ currentManager->free(ptrToMemory); });
        reset_array_of_columns();
    }
}

template<class T>
DeviceMatrix<T>& DeviceMatrix<T>::operator=(DeviceMatrix&& rhs)
{
    this->_num_rows = rhs._num_rows;
    this->_num_cols = rhs._num_cols;

    if (this != &rhs)
    {
        _data = std::move(rhs._data);
        rhs._data = nullptr;
    }

    this->DeviceDataDevice<T>::operator=(std::move(rhs));

    // We have to initialize array again,
    // because trying to move rhs._arrayOfArrays to
    // this->_arrayOfArrays results results in deep copying!
    reset_array_of_columns();
    return *this;
}

template<class T>
DeviceMatrix<T>& DeviceMatrix<T>::resize(const SizeType num_rows, const SizeType num_cols)
{
    // TODO: THIS IS POSSIBLY WRONG
    assert(num_rows * num_cols == get_num_rows() * get_num_cols() && "ERROR: Cannot resize. Dimension mismatch.");
    _num_rows = num_rows;
    _num_cols = num_cols;

    reset_array_of_columns();
    
    return *this;
}

template<class T>
T* DeviceMatrix<T>::data()
{
    return _data.get();
}

template<class T>
T const * DeviceMatrix<T>::data() const
{
    return _data.get();
}

template<class T>
typename DeviceMatrix<T>::PointerToColumn DeviceMatrix<T>::get_pointer_to_column(const SizeType m) const
{
    assert(m < get_num_cols() && "ERROR: Index out of bounds.");
    return _array_of_columns[m];
}

template<class T>
DeviceArray<T>& DeviceMatrix<T>::operator[](const SizeType m)
{
    //assert(m < get_num_cols() && "ERROR: Index out of bounds.");
    return *get_pointer_to_column(m);
}

template<class T>
DeviceArray<T> const & DeviceMatrix<T>::operator[](const SizeType m) const
{
    //assert(m < get_num_cols() && "ERROR: Index out of bounds.");
    return *get_pointer_to_column(m);
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
    return _num_rows * _num_cols;
}

template<class T>
typename DeviceMatrix<T>::SizeType DeviceMatrix<T>::get_num_rows() const
{
    return _num_rows;
}

template<class T>
typename DeviceMatrix<T>::SizeType DeviceMatrix<T>::get_num_cols() const
{
    return _num_cols;
}

template<class T>
bool DeviceMatrix<T>::is_square() const
{
    return (get_num_rows() == get_num_cols());
}

template<class T>
std::string DeviceMatrix<T>::display(const std::string& name) const
{
    std::stringstream ss;
    ss << name << " = (" << std::endl;
    for (SizeType i = 0; i < get_num_rows(); ++i)
    {
        for (SizeType j = 0; j < get_num_cols(); ++j)
            ss << std::setprecision(5) << (*this)(i,j) << "  ";
        ss<< std::endl;
    }
    ss << ")" << std::endl;
    return ss.str();
}

template<class T>
typename DeviceMatrix<T>::PointerType DeviceMatrix<T>::initialize_data()
{
    if (this->byte_size() != 0)
    {
        T* pointerToUnifiedMemory = static_cast<T*>(this->_memory_manager->allocate(this->byte_size()));
        const auto currentManager = this->_memory_manager;
        return PointerType(pointerToUnifiedMemory, [currentManager](T* ptrToUnifiedMemory){ currentManager->free(ptrToUnifiedMemory); });
    }
   else
    {
       return nullptr;
    }
}

template<class T>
typename DeviceMatrix<T>::ArrayOfPointers DeviceMatrix<T>::initialize_array_of_columns()
{
    ArrayOfPointers arrayOfPointers(get_num_cols());
    for (SizeType i = 0; i < get_num_cols(); ++i)
    {
        arrayOfPointers[i] = std::make_shared<DeviceArray<T>>(this->_memory_manager);
        arrayOfPointers[i]->make_dependent_on(get_num_rows(), data() + i * get_num_rows());
    }
    return arrayOfPointers;
}

template<class T>
void DeviceMatrix<T>::reset_array_of_columns()
{
    _array_of_columns = initialize_array_of_columns();
}

template<class T>
void DeviceMatrix<T>::reset_data()
{
    _data = initialize_data();
}
