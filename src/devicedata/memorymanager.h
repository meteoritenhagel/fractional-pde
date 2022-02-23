#ifndef TR_HACK_DEVICEDATAPOINTER_H
#define TR_HACK_DEVICEDATAPOINTER_H

#ifndef CPU_ONLY
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <typeinfo>

/**
 * Given a classical C-style array in the Cpu memory, this function initializes @p size elements starting
 * from @p data with value @p value.
 *
 * @tparam T data type of array elements
 * @param host_memory[in,out] first element of array (in the Cpu memory)
 * @param size number of elements in the array
 * @param value value to initialize elements with
 */
template<class T>
void host_initialize_memory(T* host_memory, const int size, const T value);

/**
 * Given a matrix in form of a classical C-style array in the Cpu memory, this function sets it to be an
 * identity matrix, where the storage is assumed to be column-wise.
 *
 * @tparam T data type of matrix elements
 * @param host_memory[in,out] pointer to start of data (in the Cpu memory)
 * @param num_rows number of the matrix' rows
 * @param num_cols number of the matrix' columns
 *
 * @warning Note that starting from @p data, at least num_rows*num_cols elements must be accessible
 */
template<class T>
void host_initialize_identity_matrix(T* host_memory, const int num_rows, const int num_cols);

class MemoryManagerDevice;

using MemoryManager = std::shared_ptr<MemoryManagerDevice>;

/**
 * Class MemoryManagerDevice is an abstract interface for allocating, freeing and copying
 * memory on a certain device or between devices (e.g. from Gpu memory to Cpu memory or vice versa).
 * For each device present, a separate child class should be implemented.
 */
class MemoryManagerDevice {
public:
    /**
     * Constructor
     */
    MemoryManagerDevice() = default;

    /**
     * Destructor
     */
    virtual ~MemoryManagerDevice() = default;

    /**
     * Allocates @p byte_size bytes on the corresponding device.
     * @param byte_size number of bytes to allocate
     * @return pointer to start of allocated memory
     */
    virtual void * allocate(const size_t byte_size) const = 0;

    /**
     * Frees the memory allocated on position @p pointer_to_memory.
     * @param pointer_to_memory Pointer to memory block which should be freed.
     */
    virtual void free(void* pointer_to_memory) const = 0;

    /**
     * Copies @param byte_size bytes from the @p source to the @p destination pointer.
     *
     * @warning after the destination pointer, @p byte_size bytes have to be allocated
     * @param destination destination of copy process
     * @param source source of copy process
     * @param byte_size number of bytes being copied
     * @return destination pointer
     */
    virtual void* copy(void* destination, void const * source, const size_t byte_size) const = 0;

    /**
    * Returns the memory manager's name as a string.
    * @return string representation of current instance
    */
    virtual std::string display() const = 0;

    /**
     * Allocates @p byte_size memory on the device corresponding the other MemoryManager @p manager,
     * and copies @p byte_size bytes from @p source to new newly allocated
     * space.
     *
     * @param source pointer to source memory (must have been allocated on the device corresponding to the current instance)
     * @param byteSize number of bytes being copied
     * @param manager other manager defining the target of the copy process
     * @return destination pointer, start of the newly allocated memory
     */
    void* copy_to(void* source, const size_t byteSize, const MemoryManager manager) const;
};

class CpuManager : public MemoryManagerDevice {

    /**
     * @copydoc MemoryManagerDevice::allocate(const size_t) const
     */
    void* allocate(const size_t byte_size) const override;

    /**
     * @copydoc MemoryManagerDevice::free(void*) const
     */
    void free(void* ptr) const override;

    /**
     * @copydoc MemoryManagerDevice::copy(void*, void const *, const size_t) const
     */
    void* copy(void * destination, void const * source, const size_t byte_size) const override;

    /**
     * @copydoc MemoryManagerDevice::display() const
     */
    std::string display() const override;
};

#ifndef CPU_ONLY

class GpuManager : public MemoryManagerDevice {
    /**
    * @copydoc MemoryManagerDevice::allocate(const size_t) const
    */
    void* allocate(const size_t byte_size) const override;

    /**
     * @copydoc MemoryManagerDevice::free(void*) const
     */
    void free(void* ptr) const override;

    /**
     * @copydoc MemoryManagerDevice::copy(void*, void const *, const size_t) const
     */
    void* copy(void* destination, void const * source, const size_t byte_size) const override;

    /**
     * @copydoc MemoryManagerDevice::display() const
     */
    std::string display() const override;
};

class UnifiedManager : public MemoryManagerDevice {
    /**
    * @copydoc MemoryManagerDevice::allocate(const size_t) const
    */
    void* allocate(const size_t byte_size) const override;

    /**
     * @copydoc MemoryManagerDevice::free(void*) const
     */
    void free(void* ptr) const override;

    /**
     * @copydoc MemoryManagerDevice::copy(void*, void const *, const size_t) const
     */
    void* copy(void* destination, void const * source, const size_t byte_size) const override;

    /**
     * @copydoc MemoryManagerDevice::display() const
     */
    std::string display() const override;
};

#endif

#include "memorymanager.hpp"
#endif //TR_HACK_DEVICEDATAPOINTER_H
