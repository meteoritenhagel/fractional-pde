cmake_minimum_required(VERSION 3.21)
project(fractional_pde)

set(CMAKE_CXX_STANDARD 20)

### OpenBLAS include ###
set(BLA_VENDER OpenBLAS)
find_package(BLAS REQUIRED)
if(BLAS_FOUND)
    message("OpenBLAS found.")
    include_directories(/opt/OpenBLAS/include/)
    link_libraries(${BLAS_LIBRARIES})
endif(BLAS_FOUND)

### OpenMP include ###
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message("OpenMP found.")
    link_libraries(OpenMP::OpenMP_CXX)
endif(OpenMP_CXX_FOUND)


### CUDA include if possible ###
find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    if(NOT CMAKE_CUDA_COMPILER)
        # In case this happens, check and modify your compiler path
        SET(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
    endif()
    message("CUDA found.")
    enable_language(CUDA)
else()
    message("No CUDA found.")
endif()

set(COMMON_SOURCES
        src/main.cpp
        src/fractional_pde.h
        src/fractional_pde.hpp
        src/blocksys.h
        src/blocksys_equidistant.hpp
        src/blocksys_nonequidistant.hpp
        src/demo.h
        src/demo.hpp
        src/algebraiccontainers/algebraiccontainers.h
        src/algebraiccontainers/algebraiccontainers_algebraicmatrix.hpp
        src/algebraiccontainers/algebraiccontainers_algebraicvector.hpp
        src/algebraiccontainers/coefficientmatrix.h
        src/algebraiccontainers/coefficientmatrix.hpp
        src/algebraiccontainers/containerfactory.h
        src/algebraiccontainers/containerfactory.hpp
        src/devicedata/devicedata.h
        src/devicedata/devicedata.hpp
        src/devicedata/devicedata_devicearray.hpp
        src/devicedata/devicedata_devicematrix.hpp
        src/devicedata/devicedata_devicescalar.hpp
        src/devicedata/memorymanager.h
        src/devicedata/memorymanager.cpp
        src/processingunit/processingunit.h
        src/processingunit/processingunit.hpp
        src/processingunit/processingunit_cpu.hpp
        src/processingunit/timer.h
        src/processingunit/timer.cpp
)

set(GPU_SOURCES
        src/processingunit/gpu_handle.h
        src/processingunit/gpu_handle.cpp
        src/processingunit/processingunit_gpu.hpp
        src/processingunit/processingunit_gpu_mixed.hpp
)

set(MAGMA_SOURCES
        src/processingunit/gpu_magma_queue.h
        src/processingunit/gpu_magma_queue.cpp
        src/processingunit/processingunit_gpu_magma.hpp
)

set(CUDA_SOURCES
        src/gpukernels.cuh
        src/gpukernels.cu
        src/devicedata/initializememory.cuh
        src/devicedata/initializememory.cu
)


### Cpu only mode ###
add_executable(fpde_cpu
        ${COMMON_SOURCES}
)
target_compile_definitions(fpde_cpu PRIVATE CPU_ONLY)
target_compile_options(fpde_cpu PRIVATE
        -Wall -pedantic -Wextra -Weffc++ -Woverloaded-virtual -Wfloat-equal -Wshadow -Wredundant-decls
        -fmax-errors=1 -Wno-error=redundant-decls
        -ffast-math -O3
)
target_link_options(fpde_cpu PRIVATE -O3 -flto)

if(CUDAToolkit_FOUND)
    ### CUDA mode ###
    add_executable(fpde_cuda
            ${COMMON_SOURCES}
            ${GPU_SOURCES}
            ${CUDA_SOURCES}
    )
    target_link_libraries(fpde_cuda CUDA::cudart CUDA::cublas CUDA::cusolver)
    target_compile_options(fpde_cuda PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
            --resource-usage -src-in-ptx --restrict --Wreorder --ftemplate-backtrace-limit 1
            -use_fast_math -restrict
            >)

    ### CUDA + MAGMA mode ###
    add_executable(fpde_magma
            ${COMMON_SOURCES}
            ${GPU_SOURCES}
            ${CUDA_SOURCES}
            ${MAGMA_SOURCES}
            )

    ### Set the MAGMA directories manually ###
    set(MAGMA_DIR /usr/local/magma)
    set(MAGMA_INCDIR ${MAGMA_DIR}/include)
    set(MAGMA_LIBDIR ${MAGMA_DIR}/lib)
    target_compile_definitions(fpde_magma PUBLIC MAGMA)
    target_include_directories(fpde_magma PUBLIC ${MAGMA_INCDIR})
    target_link_directories(fpde_magma PUBLIC ${MAGMA_LIBDIR})
    target_link_libraries(fpde_magma CUDA::cudart CUDA::cublas CUDA::cusolver)
    target_compile_options(fpde_magma PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
            --resource-usage -src-in-ptx --restrict --Wreorder --ftemplate-backtrace-limit 1
            -use_fast_math -restrict
            >)
endif()
