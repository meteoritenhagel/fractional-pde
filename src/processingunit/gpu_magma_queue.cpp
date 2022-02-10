/*
 * gpu_magma_queue.hpp
 *
 *  Created on: Jul 26, 2020
 *      Author: tristan
 */

#include "gpu_magma_queue.h"

//public:

GPU_MAGMA_Queue::GPU_MAGMA_Queue()
: _magmaQueue(initializeMagmaQueue())
{}

GPU_MAGMA_Queue::~GPU_MAGMA_Queue()
{
    magma_queue_destroy(_magmaQueue);
    magma_finalize();
}

magma_queue_t const& GPU_MAGMA_Queue::getMagmaQueue() const
{
    return _magmaQueue;
}

//private:

magma_queue_t GPU_MAGMA_Queue::initializeMagmaQueue()
{
    magma_queue_t magmaQueue;
    magma_init();
    magma_int_t dev = 0;
    magma_queue_create(dev, &magmaQueue);
    return magmaQueue;
}
