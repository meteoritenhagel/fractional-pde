#ifndef GPU_HANDLE_H_
#define GPU_HANDLE_H_

#include <cublas_v2.h>
#include <cusolverDn.h>

#include <cassert>
#include <iostream>

/**
 * The class GpuHandle is a wrapper for cublasHandle_t and cusolverDnHandle_,
 * which are needed for cuBLAS and cuSOLVER use.
 * It ensures easy handling of the construction / destruction actions.
 */
class GpuHandle {
public:

    /**
     * Constructor
     */
    GpuHandle();

    /**
     * Destructor
     */
    ~GpuHandle();

    /**
     * Returns the handle for cuBLAS
     * @return cuBLAS handle
     */
    cublasHandle_t const& get_cublas_handle() const;

    /**
     * Returns the handle for cuSOLVER
     * @return cuSOLVER handle
     */
    cusolverDnHandle_t const& get_cusolver_handle() const;

private:
    /**
     * Allocates and initializes a new cuBLAS handle
     * @return the new cuBLAS handle
     */
    static cublasHandle_t initialize_cublas_handle();

    /**
     * Allocates and initializes a new cuSOLVER handle
     * @return the new cuSOLVER handle
     */
    static cusolverDnHandle_t initialize_cusolver_handle();

    cublasHandle_t _cublas_handle; //!< cuBLAS handle
    cusolverDnHandle_t _cusolver_handle; //!< cuSOLVER handle

    GpuHandle(const GpuHandle&) = delete;
    GpuHandle(GpuHandle&&) = delete;
    GpuHandle& operator=(const GpuHandle&) = delete;
    GpuHandle& operator=(GpuHandle&&) = delete;
};

#endif /* GPU_HANDLE_H_ */
