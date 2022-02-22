cmake_minimum_required(VERSION 3.21)
project(fractional_pde)

set(CMAKE_CXX_STANDARD 20)

add_compile_definitions(CPU_ONLY)

add_executable(
        fractional_pde
        src/main.cpp
        src/auxiliary.h
        src/auxiliary.hpp
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

set(BLA_VENDER OpenBLAS)
find_package(BLAS REQUIRED)
if(BLAS_FOUND)
    message("OpenBLAS found.")
    include_directories(/opt/OpenBLAS/include/)
    target_link_libraries(fractional_pde ${BLAS_LIBRARIES})
endif(BLAS_FOUND)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message("OpenMP found.")
    target_link_libraries(fractional_pde OpenMP::OpenMP_CXX)
endif(OpenMP_CXX_FOUND)
