/*
 * GpuMagma_queue.hpp
 *
 *  Created on: Jul 26, 2020
 *      Author: tristan
 */

#include "GpuMagma_queue.h"

//public:

GpuMagma_Queue::GpuMagma_Queue()
: _magmaQueue(initializeMagmaQueue())
{}

GpuMagma_Queue::~GpuMagma_Queue()
{
    magma_queue_destroy(_magmaQueue);
    magma_finalize();
}

magma_queue_t const& GpuMagma_Queue::get_magma_queue() const
{
    return _magmaQueue;
}

//private:

magma_queue_t GpuMagma_Queue::initializeMagmaQueue()
{
    magma_queue_t magmaQueue;
    magma_init();
    magma_int_t dev = 0;
    magma_queue_create(dev, &magmaQueue);
    return magmaQueue;
}
