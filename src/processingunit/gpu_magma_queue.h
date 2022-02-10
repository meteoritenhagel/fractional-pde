#ifndef GPU_MAGMA_QUEUE_H_
#define GPU_MAGMA_QUEUE_H_

#include "magma_v2.h"

#include <iostream>

class GPU_MAGMA_Queue {
public:
    GPU_MAGMA_Queue();
    ~GPU_MAGMA_Queue();

    magma_queue_t const& getMagmaQueue() const;

private:
    magma_queue_t initializeMagmaQueue();

    GPU_MAGMA_Queue(const GPU_MAGMA_Queue&) = delete;
    GPU_MAGMA_Queue(GPU_MAGMA_Queue&&) = delete;
    GPU_MAGMA_Queue& operator=(const GPU_MAGMA_Queue&) = delete;
    GPU_MAGMA_Queue& operator=(GPU_MAGMA_Queue&&) = delete;

    magma_queue_t _magmaQueue;
};

#endif /* GPU_MAGMA_QUEUE_H_ */
