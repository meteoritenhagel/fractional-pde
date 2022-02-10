#ifndef UNIFIEDDATA_H_
#define UNIFIEDDATA_H_

#include "memorymanager.h"

#ifndef CPU_ONLY
#include "initializememory.cuh"
#endif

#include <cassert>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

template<class T>
void initializeMemory(const MemoryManager memoryManager, T* data, const int size, const T value) {
    if (typeid(*memoryManager) == typeid(*std::make_shared<CPU_Manager>())) {
        hostInitializeMemory(data, size, value);
    } else {
#ifndef CPU_ONLY
        deviceInitializeMemory(data, size, value);
#endif
    }
}

template<class T>
void initializeIdentityMatrix(const MemoryManager memoryManager, T* data, const int N, const int M) {
    if (typeid(*memoryManager) == typeid(*std::make_shared<CPU_Manager>())) {
        hostInitializeIdentityMatrix(data, N, M);
    } else {
#ifndef CPU_ONLY
        deviceInitializeIdentityMatrix(data, N, M);
#endif
    }
}

template<class T>
class DeviceMatrix;

template<class T>
class DeviceArray;

template<class T>
class DeviceDataDevice {
public:
    using SizeType = unsigned int;

    DeviceDataDevice(const MemoryManager memoryManager)
    : _memoryManager(memoryManager) {}

    virtual ~DeviceDataDevice() = default;

    virtual SizeType size() const = 0;

    SizeType byteSize() const
    {
        return size() * sizeof(T);
    }

    virtual T* data() = 0;
    virtual T const * data() const = 0;

    virtual void moveTo(const MemoryManager targetDevice) = 0;

    virtual void display(const std::string name) const = 0;

    MemoryManager getMemoryManager() const
    {
        return _memoryManager;
    }

protected:
    MemoryManager _memoryManager = std::make_shared<CPU_Manager>();
};

template<class T>
class DeviceMatrix : public DeviceDataDevice<T> {
public:
    using SizeType = typename DeviceDataDevice<T>::SizeType;
    using PointerToColumn = std::shared_ptr<DeviceArray<T>>;
    using ArrayOfPointers = std::vector<PointerToColumn>;
    using PointerType = std::shared_ptr<T>;

    DeviceMatrix(const SizeType N, const SizeType M, const T value = T(), const MemoryManager memoryManager = std::make_shared<CPU_Manager>());
    DeviceMatrix(const SizeType N, std::vector<T> const &u, const MemoryManager memoryManager = std::make_shared<CPU_Manager>());
    DeviceMatrix(const DeviceMatrix &other);
    DeviceMatrix(DeviceMatrix&& rhs);

    ~DeviceMatrix() override = default;

    void moveTo(const MemoryManager targetDevice) override;

    DeviceMatrix& operator= (const DeviceMatrix &other);
    DeviceMatrix& operator=(DeviceMatrix&& rhs);
    
    DeviceMatrix<T>& resize(const SizeType nrows, const SizeType ncols);

    T* data() override;
    T const * data() const override;

    PointerToColumn getPointerToColumn(const SizeType m) const;

    DeviceArray<T>& operator[](const SizeType m);
    DeviceArray<T> const & operator[](const SizeType m) const;
    T& operator()(const SizeType i, const SizeType j);
    T const & operator()(const SizeType i, const SizeType j) const;

    SizeType size() const override;
    SizeType getN() const;
    SizeType getM() const;

    bool isSquare() const;

    void display(const std::string name) const override;

private:
    SizeType _N;
    SizeType _M;
    PointerType _pointer;
    ArrayOfPointers _arrayOfPointers;

    PointerType initializePointer();
    ArrayOfPointers initializeArray();
    void resetArray();
    void resetPointer();
};

template<class T>
class DeviceArray : public DeviceDataDevice<T> {
public:
    using SizeType = typename DeviceDataDevice<T>::SizeType;
    using PointerType = std::shared_ptr<T>;

    DeviceArray(const MemoryManager memoryManager = std::make_shared<CPU_Manager>());
    explicit DeviceArray(const SizeType size, const T value = T(), const MemoryManager memoryManager = std::make_shared<CPU_Manager>());
    DeviceArray(const DeviceArray &other);
    DeviceArray(DeviceArray&& rhs) = default;

    ~DeviceArray() override = default;

    void moveTo(const MemoryManager targetDevice) override;

    DeviceArray& operator= (const DeviceArray &other);
    DeviceArray& operator=(DeviceArray&& other);

    T* data() override;
    T const * data() const override;
    T& operator[](const SizeType index);
    T const & operator[](const SizeType index) const;

    void resize(SizeType newSize);

    SizeType size() const override;
    void display(const std::string name) const override;

private:
    SizeType _size;
    PointerType _pointer;
    bool _hasOwnMemoryManagement = true;

    bool isValid();
    PointerType initializePointer();
    bool hasOwnMemoryManagement() const;
    void makeDependentOn(const SizeType size, T * const pointer);

    friend class DeviceMatrix<T>;
};

template<class T>
class DeviceScalar : public DeviceDataDevice<T> {
public:
    using SizeType = typename DeviceDataDevice<T>::SizeType;
    using PointerType = std::shared_ptr<T>;

    explicit DeviceScalar(const T &value, const MemoryManager memoryManager = std::make_shared<CPU_Manager>());
    DeviceScalar(DeviceScalar const &other);
    DeviceScalar(DeviceScalar&& rhs);

    ~DeviceScalar() = default;

    void moveTo(const MemoryManager targetDevice) override;

    DeviceScalar& operator= (DeviceScalar const &other);
    DeviceScalar& operator=(DeviceScalar&& other);

    T* data();
    T const * data() const;
    T value() const;

    SizeType size() const;

    void display(const std::string name) const override;

private:
    PointerType _pointer;

    PointerType initializePointer(const T &value);
};

#include "devicedata.hpp"
#include "devicedata_devicematrix.hpp"
#include "devicedata_devicearray.hpp"
#include "devicedata_devicescalar.hpp"
#endif /* UNIFIEDDATA_H_ */
