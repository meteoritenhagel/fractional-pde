/*
 * gpu_data.hpp
 *
 *  Created on: Jul 1, 2020
 *      Author: tristan
 */

//public:

template<class T>
GPU_Data<T>::GPU_Data(const SizeType size)
: _size(size), _pointer(initializePointer())
{}

template<class T>
GPU_Data<T>::~GPU_Data()
{
	cudaFree(_pointer);
}

template<class T>
T* GPU_Data<T>::data()
{
	return _pointer;
}

//private:
template<class T>
T* GPU_Data<T>::initializePointer()
{
	T* pointer;
	cudaMalloc(&pointer, _size*sizeof(T));
	return pointer;
}
