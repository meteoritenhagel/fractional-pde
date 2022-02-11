#ifndef GPU_MAGMA_QUEUE_H_
#define GPU_MAGMA_QUEUE_H_

#include "magma_v2.h"

#include <iostream>

/**
 * The class GPU_MAGMA_Queue is a wrapper for magma_queue_t,
 * which are needed for MAGMA use.
 * It ensures easy handling of the construction / destruction actions.
 */
class GPU_MAGMA_Queue {
public

    /**
     * Constructor
     */
    GPU_MAGMA_Queue();

    /**
     * Destructor
     */
    ~GPU_MAGMA_Queue();

    /**
     * Returns the MAGMA queue
     * @return the MAGMA queue
     */
    magma_queue_t const& getMagmaQueue() const;

private:

    /**
     * Allocates and initializes a new MAGMA queue
     * @return the new MAGMA queue
     */
    magma_queue_t initializeMagmaQueue();

    magma_queue_t _magmaQueue; //!< MAGMA queue

    GPU_MAGMA_Queue(const GPU_MAGMA_Queue&) = delete;
    GPU_MAGMA_Queue(GPU_MAGMA_Queue&&) = delete;
    GPU_MAGMA_Queue& operator=(const GPU_MAGMA_Queue&) = delete;
    GPU_MAGMA_Queue& operator=(GPU_MAGMA_Queue&&) = delete;
};

#endif /* GPU_MAGMA_QUEUE_H_ */
