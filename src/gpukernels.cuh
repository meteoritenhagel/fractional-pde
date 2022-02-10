#ifndef TR_HACK_GPUKERNELS_CUH
#define TR_HACK_GPUKERNELS_CUH

template<class floating>
__global__ void prolongationKernel(const int N, const int M, const floating * const h, floating const * const ff, floating * const solution);

template<class floating>
__global__ void rescaleRhsKernel(const int N, const int M, const floating * const h, floating * const rhs);

template<class floating>
__global__ void restrictionKernel(const int N, const int M, const floating * const h, const floating * const ff, floating * const ffOnCoarseGrid);

template<class floating>
__global__ void getReducedGridKernel(const int coarseLen, const floating * const h, floating * const coarseH);

template<class floating>
__global__ void smoothScaleKernel(const int N, const int i, const floating * const h, floating * const x);

template<class floating>
__global__ void smoothFullScaleKernel(const int N, const int M, const floating * const h, floating * const X);

template<class floating>
void deviceProlongation(const int N, const int M, const floating * const h, floating const * const ff, floating * const solution);

template<class floating>
void deviceRescaleRhs(const int N, const int M, const floating * const h, floating * const rhs);

template<class floating>
void deviceRestriction(const int N, const int M, const floating * const h, const floating * const ff, floating * const ffOnCoarseGrid);

template<class floating>
void deviceGetReducedGrid(const int coarseLen, const floating * const h, floating * const coarseH);

template<class floating>
void deviceSmoothScale(const int N, const int i, const floating * const h, floating * const x);

template<class floating>
void deviceSmoothFullScale(const int N, const int M, const floating * const h, floating * const X);

#endif //TR_HACK_GPUKERNELS_CUH
