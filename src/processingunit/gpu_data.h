/*
 * gpu_data.h
 *
 *  Created on: Jul 1, 2020
 *      Author: tristan
 */

#ifndef PROCESSINGUNIT_GPU_DATA_H_
#define PROCESSINGUNIT_GPU_DATA_H_

#include <cuda_runtime.h>

template<class T>
class GPU_Data {
public:
	using SizeType = unsigned int;

	explicit GPU_Data(const SizeType size);

	GPU_Data(const GPU_Data &other) = delete;
	GPU_Data(GPU_Data &&other) = delete;

	~GPU_Data();

	GPU_Data& operator=(const GPU_Data &other) = delete;
	GPU_Data& operator=(GPU_Data &&other) = delete;

	T* data();
private:
	SizeType _size;
	T* _pointer;

	T* initializePointer();
};

#include "gpu_data.hpp"

#endif /* PROCESSINGUNIT_GPU_DATA_H_ */
