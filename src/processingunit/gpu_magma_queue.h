#ifndef GpuMagma_QUEUE_H_
#define GpuMagma_QUEUE_H_

#include "magma_v2.h"

#include <iostream>

/**
 * The class GpuMagmaQueue is a wrapper for magma_queue_t,
 * which are needed for MAGMA use.
 * It ensures easy handling of the construction / destruction actions.
 */
class GpuMagmaQueue {
public:

    /**
     * Constructor
     */
    GpuMagmaQueue();

    /**
     * Destructor
     */
    ~GpuMagmaQueue();

    /**
     * Returns the MAGMA queue
     * @return the MAGMA queue
     */
    magma_queue_t const& get_magma_queue() const;

private:

    /**
     * Allocates and initializes a new MAGMA queue
     * @return the new MAGMA queue
     */
    magma_queue_t initializeMagmaQueue();

    magma_queue_t _magmaQueue; //!< MAGMA queue

    GpuMagmaQueue(const GpuMagmaQueue&) = delete;
    GpuMagmaQueue(GpuMagmaQueue&&) = delete;
    GpuMagmaQueue& operator=(const GpuMagmaQueue&) = delete;
    GpuMagmaQueue& operator=(GpuMagmaQueue&&) = delete;
};

#endif /* GpuMagma_QUEUE_H_ */
