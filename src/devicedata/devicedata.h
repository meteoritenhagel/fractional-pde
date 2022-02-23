#ifndef UNIFIEDDATA_H_
#define UNIFIEDDATA_H_

#include "memorymanager.h"

#ifndef CPU_ONLY
#include "initializememory.cuh"
#endif

#include <string>
#include <vector>

/** Given a classical C-style array, this function initializes @p size elements starting
 * from @p data with value @p value. The @p memory_manager indicates in which memory
 * (e.g. Cpu memory or Gpu memory) the assignment is going to take place.
 *
 * @tparam T data type
 * @param memory_manager determines where the memory assignment happens
 * @param data pointer to start of data
 * @param size number of elements to assign to
 * @param value the value with which the memory is overwritten
 */
template<class T>
void initialize_memory(const MemoryManager& memory_manager, T* data, const int size, const T value);

/**
 * Given a matrix in form of a classical C-style array, this function sets it to be an
 * identity matrix.
 *
 * @tparam T data type
 * @param memory_manager determines where the memory assignment happens
 * @param data pointer to start of data
 * @param num_rows number of the matrix' rows
 * @param num_cols number of the matrix' columns
 *
 * @warning Note that starting from @p data, at least num_rows*num_cols elements must be accessible
 */
template<class T>
void initialize_identity_matrix(const MemoryManager& memory_manager, T* data, const int num_rows, const int num_cols);

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
    SizeType byte_size() const;

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
     * @param target_device target device
     */
    virtual void move_to(const MemoryManager& target_device) = 0;

    /**
     * Displays the contents in a human-readable format as a string.
     *
     * Display is roughly the following: "@p name = (<contents>)"
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
    MemoryManager _memory_manager = std::make_shared<CpuManager>(); //!< The memory manager to determine on which device memory is ccessed
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
     * @param num_rows Number of matrix rows
     * @param num_cols Number of matric columns
     * @param value value to initialize matrix elements
     * @param memory_manager memory manager
     */
    DeviceMatrix(const SizeType num_rows, const SizeType num_cols, const T value = T(),
                 const MemoryManager& memory_manager = std::make_shared<CpuManager>());

    /**
     * Constructor, constructs a DeviceMatrix from a std::vector on the Cpu memory.
     * @param num_rows number of matrix rows
     * @param vec vector containing the column-wise stored matrix
     * @param memoryManager memory manager
     *
     * @warning the size of @p vec must be a multiple of @p num_rows
     */
    DeviceMatrix(const SizeType num_rows, std::vector<T> const &vec);

    /**
     * Copy constructor
     */
    DeviceMatrix(const DeviceMatrix &other);

    /**
     * Move constructor
     */
    DeviceMatrix(DeviceMatrix&& rhs);

    /**
     * Destructor
     */
    ~DeviceMatrix() override = default;

    void move_to(const MemoryManager& target_device) override;

    /**
     * Copy assignment operator
     */
    DeviceMatrix& operator= (const DeviceMatrix &other);

    /**
     * Move assignment operator
     */
    DeviceMatrix& operator=(DeviceMatrix&& rhs);

    /**
     * Resize the matrix to new shape.
     * @param num_rows number of rows
     * @param num_cols number of columns
     * @return reference to current instance
     *
     * @warning num_rows*num_cols must still stay the original size
     */
    DeviceMatrix<T>& resize(const SizeType num_rows, const SizeType num_cols);

    T* data() override;
    T const * data() const override;

    /**
     * Returns a pointer to the DeviceArray forming the @p m-th column of the matrix.
     * Changing the underlying elements results in the matrix also being changed.
     * @param m index of column to return
     * @return pointer to @p m-th column
     */
    PointerToColumn get_pointer_to_column(const SizeType m) const;

    /**
     * Returns a reference to the @p m-th column of the matrix as a mutable object,
     * e.g. changes to the underlying elements result in changes to the matrix itself.
     * @p m index of column to return
     * @return reference to @p m-th column
     */
    DeviceArray<T>& operator[](const SizeType m);

    /**
     * Returns a const reference to the @p m-th column of the matrix as a mutable object,
     * e.g. changes to the underlying elements result in changes to the matrix itself.
     * @p m index of column to return
     * @return const reference to @p m-th column
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

    /**
     * Get number of rows.
     * @return number of rows
     */
    SizeType get_num_rows() const;

    /**
     * Get number of columns.
     * @return number of columns
     */
    SizeType get_num_cols() const;

    /**
     * Checks whether the current matrix is square.
     * @return true if the matrix is square.
     */
    bool is_square() const;

    std::string display(const std::string& name) const override;

private:
    SizeType _num_rows; //!< number of the matrix's rows
    SizeType _num_cols; //!< number of the matrix's columns
    PointerType _data; //!< pointer to the first element
    ArrayOfPointers _array_of_columns; //!< array of pointers to the individual columns of the matrix

    /**
     * Allocates the correct amount of memory on the correct device.
     * @return pointer to the first element
     */
    PointerType initialize_data();

    /**
     * Initializes the array of pointers to the matrix's individual columns.
     * This is needed to provide mutable access of the columns via reference.
     * @return correctly initialized array of such pointers
     */
    ArrayOfPointers initialize_array_of_columns();

    /**
     * @copydoc initialize_array_of_columns()
     * Has the same effect as initialize_array_of_columns and then updating the corresponding member variable accordingly.
     */
    void reset_array_of_columns();

    /**
     * @copydoc reset_data()
     * Has the same effect as initialize_data and then updating the corresponding member variable accordingly.
     */
    void reset_data();
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
     * Constructor, constructs empty array with given @p memory_manager
     * @param memory_manager memory manager
     */
    DeviceArray(const MemoryManager& memory_manager = std::make_shared<CpuManager>());

    /**
     * Constructor, constructs an array of given size, where each element is initialized with @p value.
     * @param size number of elements
     * @param value value to initialize the individual elements
     * @param memory_manager memory manager
     */
    explicit DeviceArray(const SizeType size, const T value = T(),
                         const MemoryManager& memory_manager = std::make_shared<CpuManager>());

    /**
     * Copy constructor
     */
    DeviceArray(const DeviceArray &other);

    /**
     * Move constructor
     */
    DeviceArray(DeviceArray&&) = default;

    /**
     * Destructor
     */
    ~DeviceArray() override = default;

    void move_to(const MemoryManager& target_device) override;

    /**
     * Copy assignment operator.
     */
    DeviceArray& operator= (const DeviceArray &other);

    /**
     * Move assignment operator.
     */
    DeviceArray& operator=(DeviceArray&& other);

    T* data() override;
    T const * data() const override;

    /**
     * Returns a reference to the @p index-th element of the array as a mutable object,
     * e.g. changes to this element result in changes to the array itself.
     * @param index index of element to return
     * @return reference to @p index-th element
     */
    T& operator[](const SizeType index);

    /**
     * Returns a const reference to the @p index-th element of the array as a mutable object,
     * e.g. changes to this element result in changes to the array itself.
     * @param index index of element to return
     * @return const reference to @p index-th element
     */
    T const & operator[](const SizeType index) const;

    /**
    * Resize the array to new length.
    * @param new_size new length of array
    * @return reference to current instance
    *
    * @warning the elements previously contained get lost
    */
    DeviceArray<T> resize(SizeType new_size);

    SizeType size() const override;
    std::string display(const std::string& name) const override;

private:
    SizeType _size; //!< length of array
    PointerType _data; //!< pointer to start of array
    bool _has_own_memory_management = true; //!< determines whether the array is stand-alone or a column of an existing DeviceMatrix

    /**
     * Checks whether the array is valid.
     * @return true if the array is valid
     */
    bool is_valid();

    /**
     * In case the array's length is >0, allocates the right amount of memory on the right device.
     * Otherwise, a nullptr is returned.
     * @return Pointer to start of array
     */
    PointerType initialize_data();

    /**
     * Checks whether the array is stand-alone and manages its own memory,
     * or if its contents are the column of a DeviceMatrix.
     * @return true if the current instance manages memory on its own
     */
    bool has_own_memory_management() const;

    /**
     * Sets the array to not allocate or delete memory on its own,
     * as it is dependent on existing memory. This is important
     * for column-wise access of DeviceMatrix, where the individual
     * columns are accessed as references to DeviceArray.
     * @param size Number of elements
     * @param pointer Pointer to start of memory.
     */
    void make_dependent_on(const SizeType size, T * const pointer);

    friend class DeviceMatrix<T>; // DeviceMatrix needs to be a friend to access the make_dependent_on private member function.
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
     * Constructor, constructs scalar value using the @p memory_manager
     * @param value value of the scalar value
     * @param memory_manager memory manager
     */
    explicit DeviceScalar(const T &value, const MemoryManager& memory_manager = std::make_shared<CpuManager>());

    /**
     * Copy constructor.
     */
    DeviceScalar(DeviceScalar const &other);

    /**
     * Move constructor.
     */
    DeviceScalar(DeviceScalar&& rhs);

    /**
     * Destructor
     */
    ~DeviceScalar() override = default;

    void move_to(const MemoryManager& target_device) override;

    /**
     * Copy assignment operator.
     */
    DeviceScalar& operator=(DeviceScalar const &other);

    /**
     * Move assignment operator.
     */
    DeviceScalar& operator=(DeviceScalar&& other);

    T* data() override;
    T const * data() const override;

    /**
     * Returns the value of the contained element in RAM.
     * @return value of contained element
     */
    T value() const;

    SizeType size() const override;
    std::string display(const std::string& name) const override;

private:
    PointerType _data; //<! Pointer to contained element

    /**
     * Allocates the right amount of memory for one instance of type T
     * and initializes it with value @p value.
     * @return Pointer to freshly initialized element
     */
    PointerType initialize_data(const T &value);
};

#include "devicedata.hpp"
#include "devicedata_devicematrix.hpp"
#include "devicedata_devicearray.hpp"
#include "devicedata_devicescalar.hpp"
#endif /* UNIFIEDDATA_H_ */
