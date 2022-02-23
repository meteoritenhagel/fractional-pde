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
 * @param hostMemory[in,out] first element of array (in the Cpu memory)
 * @param size number of elements in the array
 * @param value value to initialize elements with
 */
template<class T>
void hostInitializeMemory(T* hostMemory, const int size, const T value);

/**
 * Given a matrix in form of a classical C-style array in the Cpu memory, this function sets it to be an
 * identity matrix.
 *
 * @tparam T data type of matrix elements
 * @param hostMemory[in,out] pointer to start of data (in the Cpu memory)
 * @param N number of the matrix' rows
 * @param M number of the matrix' columns
 *
 * @warning Note that starting from @param data, at least N*M elements must be accessible
 */
template<class T>
void hostInitializeIdentityMatrix(T* hostMemory, const int N, const int M);

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
     * Allocates @param byteSize bytes on the corresponding device.
     * @param byteSize number of bytes to allocate
     * @return pointer to start of allocated memory
     */
    virtual void * allocate(const size_t byteSize) const = 0;

    /**
     * Frees the memory allocated on position @p pointerToMemory.
     * @param pointerToMemory Pointer to memory block which should be freed.
     */
    virtual void free(void* pointerToMemory) const = 0;

    /**
     * Copies @param byteSize bytes from the @p source to the @p destination pointer.
     *
     * @warning after the destination pointer, @p byteSize bytes have to be allocated
     * @param destination destination of copy process
     * @param source source of copy process
     * @param byteSize number of bytes being copied
     * @return destination pointer
     */
    virtual void* copy(void* destination, void const * source, const size_t byteSize) const = 0;

    /**
    * Returns the memory manager's name as a string.
    * @return string representation of current instance
    */
    virtual std::string display() const = 0;

    /**
     * Allocates @p byteSize memory on the device corresponding the other MemoryManager @p manager,
     * and copies @p byteSize bytes from @p pointerToMemory to new newly allocated
     * space.
     *
     * @param pointerToMemory pointer to source memory (must have been allocated on the device corresponding to the current instance)
     * @param byteSize number of bytes being copied
     * @param manager other manager defining the target of the copy process
     * @return destination pointer, start of the newly allocated memory
     */
    void* copyTo(void* pointerToMemory, const size_t byteSize, const MemoryManager manager) const;
};

class CPU_Manager : public MemoryManagerDevice {

    /**
     * @copydoc MemoryManagerDevice::allocate(const size_t) const
     */
    void* allocate(const size_t byteSize) const override;

    /**
     * @copydoc MemoryManagerDevice::free(void*) const
     */
    void free(void* pointerToMemory) const override;

    /**
     * @copydoc MemoryManagerDevice::copy(void*, void const *, const size_t) const
     */
    void* copy(void * destination, void const * source, const size_t byteSize) const override;

    /**
     * @copydoc MemoryManagerDevice::display() const
     */
    std::string display() const override;
};

#ifndef CPU_ONLY

class GPU_Manager : public MemoryManagerDevice {
    /**
    * @copydoc MemoryManagerDevice::allocate(const size_t) const
    */
    void* allocate(const size_t byteSize) const override;

    /**
     * @copydoc MemoryManagerDevice::free(void*) const
     */
    void free(void* pointerToMemory) const override;

    /**
     * @copydoc MemoryManagerDevice::copy(void*, void const *, const size_t) const
     */
    void* copy(void* destination, void const * source, const size_t byteSize) const override;

    /**
     * @copydoc MemoryManagerDevice::display() const
     */
    std::string display() const override;
};

class UnifiedManager : public MemoryManagerDevice {
    /**
    * @copydoc MemoryManagerDevice::allocate(const size_t) const
    */
    void* allocate(const size_t byteSize) const override;

    /**
     * @copydoc MemoryManagerDevice::free(void*) const
     */
    void free(void* pointerToMemory) const override;

    /**
     * @copydoc MemoryManagerDevice::copy(void*, void const *, const size_t) const
     */
    void* copy(void* destination, void const * source, const size_t byteSize) const override;

    /**
     * @copydoc MemoryManagerDevice::display() const
     */
    std::string display() const override;
};

#endif

#include "memorymanager.hpp"
#endif //TR_HACK_DEVICEDATAPOINTER_H
