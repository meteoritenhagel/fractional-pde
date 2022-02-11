#ifndef GPU_HANDLE_H_
#define GPU_HANDLE_H_

#include <cublas_v2.h>
#include <cusolverDn.h>

#include <cassert>
#include <iostream>

/**
 * The class GPU_Handle is a wrapper for cublasHandle_t and cusolverDnHandle_,
 * which are needed for CUBLAS and cuSOLVER use.
 * It ensures easy handling of the construction / destruction actions.
 */
class GPU_Handle {
public:

    /**
     * Constructor
     */
    GPU_Handle();

    /**
     * Destructor
     */
    ~GPU_Handle();

    /**
     * Returns the handle for CUBLAS
     * @return CUBLAS handle
     */
    cublasHandle_t const& getCublasHandle() const;

    /**
     * Returns the handle for cuSOLVER
     * @return cuSOLVER handle
     */
    cusolverDnHandle_t const& getCusolverHandle() const;

private:
    /**
     * Allocates and initializes a new CUBLAS handle
     * @return the new CUBLAS handle
     */
    static cublasHandle_t initializeCublasHandle();

    /**
     * Allocates and initializes a new cuSOLVER handle
     * @return the new cuSOLVER handle
     */
    static cusolverDnHandle_t initializeCusolverHandle();

    cublasHandle_t _cublasHandle; //!< CUBLAS handle
    cusolverDnHandle_t _cusolverHandle; //!< cuSOLVER handle

    GPU_Handle(const GPU_Handle&) = delete;
    GPU_Handle(GPU_Handle&&) = delete;
    GPU_Handle& operator=(const GPU_Handle&) = delete;
    GPU_Handle& operator=(GPU_Handle&&) = delete;
};

#endif /* GPU_HANDLE_H_ */
