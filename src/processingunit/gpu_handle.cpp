/*
 * gpu_handle.hpp
 *
 *  Created on: Jul 26, 2020
 *      Author: tristan
 */

//public:

#include "gpu_handle.h"
#include <exception>

// TODO: This initialization might be prone to errors, since
// when the second member initalization fails, the cublas handle
// is not destroyed.
GPU_Handle::GPU_Handle()
: _cublasHandle(initializeCublasHandle()), _cusolverHandle(initializeCusolverHandle())
{
    cublasSetPointerMode(getCublasHandle(), CUBLAS_POINTER_MODE_HOST);
}

GPU_Handle::~GPU_Handle()
{
    cublasDestroy(_cublasHandle);
    cusolverDnDestroy(_cusolverHandle);
}

cublasHandle_t const& GPU_Handle::getCublasHandle() const
{
    return _cublasHandle;
}

cusolverDnHandle_t const& GPU_Handle::getCusolverHandle() const
{
    return _cusolverHandle;
}

//private:

cublasHandle_t GPU_Handle::initializeCublasHandle()
{
    cublasHandle_t handleptr;
    cublasStatus_t status = cublasCreate(&handleptr);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        throw std::runtime_error("GPU_Handle ERROR: cuBLAS initialization failed.");
    }
    return handleptr;
}

cusolverDnHandle_t GPU_Handle::initializeCusolverHandle()
{
    cusolverDnHandle_t handleptr;
    cusolverStatus_t status = cusolverDnCreate(&handleptr);
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        throw std::runtime_error("GPU_Handle ERROR: cuSOLVER initialization failed.");
        assert(status);
    }
    return handleptr;
}
