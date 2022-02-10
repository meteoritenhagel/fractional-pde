# fractional-pde

Solver for Caputo fractional PDEs over equidistant time and
arbitrary space grids.

Documentation of this repository is still work in progress.

The code in this repository has been written in the course of
my master's thesis "Enhancing a C++ Program Solving Time Fractional
PDEs" (https://unipub.uni-graz.at/obvugrhs/download/pdf/6408510),
in which an introduction to Caputo fractional PDEs is given. The thesis
also documents the development of the code and serves as a more in-depth
documentation of the code base.

The program has the following features:

* Calculation of solutions for Caputo fractional PDEs
* Both equidistant and non-equidistant space grids are supported. 
  Time grids have to be equidistant.
* Different algorithms for solving the PDE, the most recommended are Cyclic Reduction
  for equidistant space grids, and multigrid preconditioned BiCGStab for non-equidistant
  space grids.
* Templatized solving routines allowing to choose the precision of the floating point numbers
  used between `float` or `double`.
* Custom data types for scalars, arrays and matrices allowing to be allocated and transferred
  from the CPU's to GPU's memory and vice versa. They are contained in the folder
  `src/devicedata`.
* Additional data types for arrays and matrices supporting efficient linear algebra
  operations. The data types are contained in the folder `srd/algebraiccontainers`.
* These linear algebra operations are accelerated using an adapter generalizing BLAS/LAPACK,
  cuBLAS/cuSOLVER, and MAGMA, for calculation on the CPU or GPU. This adapter is contained
  in the folder `src/processingunit`.

The contents of the folders `src/devicedata`, `src/processingunit` and
`src/algebraiccontainers` can be used independently of the solver. `devicedata` is
stand-alone, which is a prerequisite for `processingunit`, which in turn is again
a prerequisite for `algebraiccontainers`.

An exemplary use of the solver is demonstrated in `main.cpp`.
