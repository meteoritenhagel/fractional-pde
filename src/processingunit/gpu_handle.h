#ifndef GPU_HANDLE_H_
#define GPU_HANDLE_H_

#include <cublas_v2.h>
#include <cusolverDn.h>

#include <cassert>
#include <iostream>

class GPU_Handle {
public:
    GPU_Handle();
    ~GPU_Handle();

    GPU_Handle(const GPU_Handle&) = delete;
    GPU_Handle(GPU_Handle&&) = delete;
    GPU_Handle& operator=(const GPU_Handle&) = delete;
    GPU_Handle& operator=(GPU_Handle&&) = delete;

    cublasHandle_t const& getCublasHandle() const;
    cusolverDnHandle_t const& getCusolverHandle() const;

private:
    static cublasHandle_t initializeCublasHandle();
    static cusolverDnHandle_t initializeCusolverHandle();

    cublasHandle_t _cublasHandle;
    cusolverDnHandle_t _cusolverHandle;
};

#endif /* GPU_HANDLE_H_ */
