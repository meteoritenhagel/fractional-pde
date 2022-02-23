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
GpuHandle::GpuHandle()
: _cublas_handle(initialize_cublas_handle()), _cusolver_handle(initialize_cusolver_handle())
{
    cublasSetPointerMode(get_cublas_handle(), CUBLAS_POINTER_MODE_HOST);
}

GpuHandle::~GpuHandle()
{
    cublasDestroy(_cublas_handle);
    cusolverDnDestroy(_cusolver_handle);
}

cublasHandle_t const& GpuHandle::get_cublas_handle() const
{
    return _cublas_handle;
}

cusolverDnHandle_t const& GpuHandle::get_cusolver_handle() const
{
    return _cusolver_handle;
}

//private:

cublasHandle_t GpuHandle::initialize_cublas_handle()
{
    cublasHandle_t handleptr;
    cublasStatus_t status = cublasCreate(&handleptr);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        throw std::runtime_error("GpuHandle ERROR: cuBLAS initialization failed.");
    }
    return handleptr;
}

cusolverDnHandle_t GpuHandle::initialize_cusolver_handle()
{
    cusolverDnHandle_t handleptr;
    cusolverStatus_t status = cusolverDnCreate(&handleptr);
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        throw std::runtime_error("GpuHandle ERROR: cuSOLVER initialization failed.");
        assert(status);
    }
    return handleptr;
}
