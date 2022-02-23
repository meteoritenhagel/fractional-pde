#ifndef UNIFIEDDATA_H_
#define UNIFIEDDATA_H_

#include "memorymanager.h"

#ifndef CPU_ONLY
#include "initializememory.cuh"
#endif

#include <string>
#include <vector>

/** Given a classical C-style array, this function initializes @param size elements starting
 * from @param data with value @param value. The @param memoryManager indicates in which memory
 * (e.g. Cpu memory or Gpu memory) the assignment is going to take place.
 *
 * @tparam T data type
 * @param memoryManager determines where the memory assignment happens
 * @param data pointer to start of data
 * @param size number of elements to assign to
 * @param value the value with which the memory is overwritten
 */
template<class T>
void initializeMemory(const MemoryManager& memoryManager, T* data, const int size, const T value);

/**
 * Given a matrix in form of a classical C-style array, this function sets it to be an
 * identity matrix.
 *
 * @tparam T data type
 * @param memoryManager determines where the memory assignment happens
 * @param data pointer to start of data
 * @param N number of the matrix' rows
 * @param M number of the matrix' columns
 *
 * @warning Note that starting from @param data, at least N*M elements must be accessible
 */
template<class T>
void initializeIdentityMatrix(const MemoryManager& memoryManager, T* data, const int N, const int M);

template<class T>
class DeviceMatrix;

template<class T>
class DeviceArray;

/**
 * The class DeviceDataDevice serves as an abstract interface, from which individual
 * data types are derived. These data types contain elements on a specific device,
 * but can easily be shifted onto another device.
 *
 * @tparam T element data type
 */
template<class T>
class DeviceDataDevice {
public:
    using SizeType = unsigned int;

    /** Constructor
     *
     * @param memoryManager determining where the contained data is located to ensure right memory accesses
     */
    DeviceDataDevice(const MemoryManager& memoryManager);

    /**
     * Destructor
     */
    virtual ~DeviceDataDevice() = default;

    /**
     * Returns the total number of contained elements.
     * @return total number of contained elements
     */
    virtual SizeType size() const = 0;

    /**
     * Returns the number of bytes the contained elements occupy in memory.
     * @return number of bytes the contained elements occupy in memory.
     */
    SizeType byteSize() const;

    /**
     * Pointer to the first memory position of the contained elements.
     * @return Pointer to first element.
     */
    virtual T* data() = 0;

    /**
     * Const pointer to the first memory position of the contained elements.
     * @return Const pointer to first element.
     */
    virtual T const * data() const = 0;

    /**
     * Move all the elements contained to another device, e.g. from Cpu memory to Gpu memory.
     * @param targetDevice target device
     */
    virtual void moveTo(const MemoryManager& targetDevice) = 0;

    /**
     * Displays the contents in a human-readable format as a string.
     *
     * Display is roughly the following: "@param name = (<contents>)"
     * @param name Name to display
     *
     * @return string representation of current instance
     */
    virtual std::string display(const std::string& name) const = 0;

    /**
     * Returns the current instance's memory manager
     * @return current memory manager
     */
    MemoryManager get_memory_manager() const;

protected:
    MemoryManager _memoryManager = std::make_shared<CPU_Manager>(); //!< The memory manager to determine on which device memory is ccessed
};

/**
 * The class DeviceMatrix represents a matrix which is allocated on a specific device,
 * and provides some methods for element access.
 * Instances of this class can be easily shifted from one device to another.
 * @tparam T data type of contained elements
 */
template<class T>
class DeviceMatrix : public DeviceDataDevice<T> {
public:
    using SizeType = typename DeviceDataDevice<T>::SizeType;
    using PointerToColumn = std::shared_ptr<DeviceArray<T>>;
    using ArrayOfPointers = std::vector<PointerToColumn>;
    using PointerType = std::shared_ptr<T>;

    /**
     * Constructor, allocates a DeviceMatrix
     * @param N Number of matrix rows
     * @param M Number of matric columns
     * @param value value to initialize matrix elements
     * @param memoryManager memory manager
     */
    DeviceMatrix(const SizeType N, const SizeType M, const T value = T(), const MemoryManager& memoryManager = std::make_shared<CPU_Manager>());

    /**
     * Constructor, constructs a DeviceMatrix from a std::vector on the Cpu memory.
     * @param N number of matrix rows
     * @param u vector containing the column-wise stored matrix
     * @param memoryManager memory manager
     *
     * @warning the size of @param u must be a multiple of @param N
     */
    DeviceMatrix(const SizeType N, std::vector<T> const &u);

    /**
     * Copy constructor
     * @param other
     */
    DeviceMatrix(const DeviceMatrix &other);

    /**
     * Move constructor
     * @param rhs
     */
    DeviceMatrix(DeviceMatrix&& rhs);

    /**
     * Destructor
     */
    ~DeviceMatrix() override = default;

    void moveTo(const MemoryManager& targetDevice) override;

    /**
     * Copy assignment operator
     * @param other
     * @return
     */
    DeviceMatrix& operator= (const DeviceMatrix &other);

    /**
     * Move assignment operator
     * @param rhs
     * @return
     */
    DeviceMatrix& operator=(DeviceMatrix&& rhs);

    /**
     * Resize the matrix to new shape.
     * @param nrows number of rows
     * @param ncols number of columns
     * @return reference to current instance
     *
     * @warning nrows*ncols must still stay the original size
     */
    DeviceMatrix<T>& resize(const SizeType nrows, const SizeType ncols);

    T* data() override;
    T const * data() const override;

    /**
     * Returns a pointer to the DeviceArray forming the @param m-th column of the matrix.
     * Changing the underlying elements results in the matrix also being changed.
     * @param m index of column to return
     * @return pointer to m-th column
     */
    PointerToColumn getPointerToColumn(const SizeType m) const;

    /**
     * Returns a reference to the @param m-th column of the matrix as a mutable object,
     * e.g. changes to the underlying elements result in changes to the matrix itself.
     * @param m index of column to return
     * @return reference to @param m-th column
     */
    DeviceArray<T>& operator[](const SizeType m);

    /**
     * Returns a const reference to the @param m-th column of the matrix as a mutable object,
     * e.g. changes to the underlying elements result in changes to the matrix itself.
     * @param m index of column to return
     * @return const reference to @param m-th column
     */
    DeviceArray<T> const & operator[](const SizeType m) const;

    /**
     * Access operator. Changes to the underlying element result in changes to the matrix itself.
     * @param i row index
     * @param j column index
     * @return the element with matrix index (i,j)
     */
    T& operator()(const SizeType i, const SizeType j);

    /**
     * Const access operator.
     * @param i row index
     * @param j column index
     * @return the element with matrix index (i,j)
     */
    T const & operator()(const SizeType i, const SizeType j) const;

    SizeType size() const override;

    // TODO: here, the functions are called getN() and getM(), but for AlgebraicMatrix it is getNrows() and getNcols()
    /**
     * Get number of rows.
     * @return number of rows
     */
    SizeType getN() const;

    /**
     * Get number of columns.
     * @return number of columns
     */
    SizeType getM() const;

    /**
     * Checks whether the current matrix is square.
     * @return true if the matrix is square.
     */
    bool isSquare() const;

    std::string display(const std::string& name) const override;

private:
    SizeType _N; //!< number of the matrix's rows
    SizeType _M; //!< number of the matrix's columns
    PointerType _pointer; //!< pointer to the first element
    ArrayOfPointers _arrayOfPointers; //!< array of pointers to the individual columns of the matrix

    /**
     * Allocates the correct amount of memory on the correct device.
     * @return pointer to the first element
     */
    PointerType initializePointer();

    /**
     * Initializes the array of pointers to the matrix's individual columns.
     * This is needed to provide mutable access of the columns via reference.
     * @return correctly initialized array of such pointers
     */
    ArrayOfPointers initializeArray();

    /**
     * Has the same effect as initializeArray and then updating the corresponding member variable accordingly.
     */
    void resetArray();

    /**
     * Has the same effect as initializePointer and then updating the corresponding member variable accordingly.
     */
    void resetPointer();
};

/**
 * The class DeviceArray represents an array or a column of DeviceMatrix which is allocated on a specific device,
 * and provides some methods for element access.
 * Instances of this class can be easily shifted from one device to another.
 * @tparam T data type of contained elements
 */
template<class T>
class DeviceArray : public DeviceDataDevice<T> {
public:
    using SizeType = typename DeviceDataDevice<T>::SizeType;
    using PointerType = std::shared_ptr<T>;

    /**
     * Constructor, constructs empty array with given memoryManager
     * @param memoryManager memory manager
     */
    DeviceArray(const MemoryManager& memoryManager = std::make_shared<CPU_Manager>());

    /**
     * Constructor, constructs an array of given size, where each element is initialized with @param value.
     * @param size number of elements
     * @param value value to initialize the individual elements
     * @param memoryManager memory manager
     */
    explicit DeviceArray(const SizeType size, const T value = T(), const MemoryManager& memoryManager = std::make_shared<CPU_Manager>());

    /**
     * Copy constructor
     * @param other
     */
    DeviceArray(const DeviceArray &other);

    /**
     * Move constructor
     * @param rhs
     */
    DeviceArray(DeviceArray&& rhs) = default;

    /**
     * Destructor
     */
    ~DeviceArray() override = default;

    void moveTo(const MemoryManager& targetDevice) override;

    /**
     * Copy assignment operator.
     * @param other
     * @return
     */
    DeviceArray& operator= (const DeviceArray &other);

    /**
     * Move assignment operator.
     * @param other
     * @return
     */
    DeviceArray& operator=(DeviceArray&& other);

    T* data() override;
    T const * data() const override;

    /**
     * Returns a reference to the @param index-th element of the array as a mutable object,
     * e.g. changes to this element result in changes to the array itself.
     * @param index index of element to return
     * @return reference to @param index-th element
     */
    T& operator[](const SizeType index);

    /**
     * Returns a const reference to the @param index-th element of the array as a mutable object,
     * e.g. changes to this element result in changes to the array itself.
     * @param index index of element to return
     * @return const reference to @param index-th element
     */
    T const & operator[](const SizeType index) const;

    /**
    * Resize the array to new length.
    * @param newSize new length of array
    * @return reference to current instance
    *
    * @warning the elements previously contained get lost
    */
    DeviceArray<T> resize(SizeType newSize);

    SizeType size() const override;
    std::string display(const std::string& name) const override;

private:
    SizeType _size; //!< length of array
    PointerType _pointer; //!< pointer to start of array
    bool _hasOwnMemoryManagement = true; //!< determines whether the array is stand-alone or a column of an existing DeviceMatrix

    /**
     * Checks whether the array is valid.
     * @return true if the array is valid
     */
    bool isValid();

    /**
     * In case the array's length is >0, allocates the right amount of memory on the right device.
     * Otherwise, a nullptr is returned.
     * @return Pointer to start of array
     */
    PointerType initializePointer();

    /**
     * Checks whether the array is stand-alone and manages its own memory,
     * or if its contents are the column of a DeviceMatrix.
     * @return
     */
    bool hasOwnMemoryManagement() const;

    /**
     * Sets the array to not allocate or delete memory on its own,
     * as it is dependent on existing memory. This is important
     * for column-wise access of DeviceMatrix, where the individual
     * columns are accessed as references to DeviceArray.
     * @param size Number of elements
     * @param pointer Pointer to start of memory.
     */
    void makeDependentOn(const SizeType size, T * const pointer);

    friend class DeviceMatrix<T>; // DeviceMatrix needs to be a friend to access the makeDependentOn private member function.
};

/**
 * The class DeviceScalar represents an individual is allocated on a specific device.
 * Instances of this class can be easily shifted from one device to another.
 * @tparam T data type of contained elements
 */
template<class T>
class DeviceScalar : public DeviceDataDevice<T> {
public:
    using SizeType = typename DeviceDataDevice<T>::SizeType;
    using PointerType = std::shared_ptr<T>;

    /**
     * Constructor, constructs scalar value using the @param memoryManager
     * @param value value of the scalar value
     * @param memoryManager memory manager
     */
    explicit DeviceScalar(const T &value, const MemoryManager& memoryManager = std::make_shared<CPU_Manager>());

    /**
     * Copy constructor.
     * @param other
     */
    DeviceScalar(DeviceScalar const &other);

    /**
     * Move constructor.
     * @param rhs
     */
    DeviceScalar(DeviceScalar&& rhs);

    /**
     * Destructor
     */
    ~DeviceScalar() override = default;

    void moveTo(const MemoryManager& targetDevice) override;

    /**
     * Copy assignment operator.
     * @param other
     * @return
     */
    DeviceScalar& operator= (DeviceScalar const &other);

    /**
     * Move assignment operator.
     * @param other
     * @return
     */
    DeviceScalar& operator=(DeviceScalar&& other);

    T* data() override;
    T const * data() const override;

    /**
     * Returns the value of the contained element on Cpu memory.
     * @return value of contained element
     */
    T value() const;

    SizeType size() const override;
    std::string display(const std::string& name) const override;

private:
    PointerType _pointer; //<! Pointer to contained element

    /**
     * Allocates the right amount of memory for one instance of type T
     * and initializes it with value @param value.
     * @return Pointer to element
     */
    PointerType initializePointer(const T &value);
};

#include "devicedata.hpp"
#include "devicedata_devicematrix.hpp"
#include "devicedata_devicearray.hpp"
#include "devicedata_devicescalar.hpp"
#endif /* UNIFIEDDATA_H_ */
