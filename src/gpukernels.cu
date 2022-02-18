#include "gpukernels.cuh"

template<class floating>
__global__ void prolongationKernel(const int N, const int M, const floating *const h, const floating *const ff,
                              floating *const solution) {
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < N*M)
    {
        if (idx < N) // 0th column
        {
            solution[idx] += ff[idx];
        }
        else // jth column
        {
            const size_t j = idx / N;
            const size_t i = idx - j*N;

            if (j % 2) // odd number of columns, i.e. 1st, 3rd, 5th etc.
            {
                const auto coeff1 = h[j] / (h[j - 1] + h[j]);
                const auto coeff2 = h[j-1] / (h[j - 1] + h[j]);
                solution[idx] += coeff1 * ff[(j-1)/2 * N + i] + coeff2*ff[(j+1)/2 * N + i];
            }
            else // even number of columns, i.e. 2nd, 4th, 6th etc.
            {
                solution[idx] += ff[(j + 1) / 2 * N + i];
            }
        }
    }
}

template<class floating>
__global__ void rescaleRhsKernel(const int N, const int M, const floating * const h, floating * const rhs)
{
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < N*M)
    {
        const size_t j = idx / N;

        if (1 <= j && j <= M-2)
        {
            rhs[idx] *= h[j] + h[j-1];
        }
    }
}

template<class floating>
__global__ void restrictionKernel(const int N, const int M, const floating * const h, const floating * const ff, floating * const ffOnCoarseGrid)
{
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;

    const size_t j = idx / N;
    const size_t i = idx - j*N;

    if (idx < N*M && (j%2 == 0) && 2 <= j && j < M-1)
    {
        const auto coeff1 = h[j - 2] / (h[j - 2] + h[j - 1]);
        const auto coeff2 = h[j + 1] / (h[j] + h[j + 1]);
        ffOnCoarseGrid[j/2 * N + i] = coeff1 * ff[(j-1) * N + i] + ff[j * N + i] + coeff2 * ff[(j+1)*N + i];

    }
}

template<class floating>
__global__ void getReducedGridKernel(const int coarseLen, const floating * const h, floating * const coarseH)
{
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < coarseLen)
        coarseH[idx] = h[2*idx] + h[2*idx+1];
}

template<class floating>
__global__ void smoothScaleKernel(const int N, const int i, const floating * const h, floating * const x)
{
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N)
    {
        const floating inv_scale_factor = 1/(h[i-1] + h[i]);
        const floating inv_coeff_middle = 1/(2*(h[i-1] + h[i])/h[i-1]/h[i]);
        if (idx == 0 || idx == 1 || idx == N-1)
            x[idx] *= inv_scale_factor;
        else
            x[idx] *= inv_coeff_middle;
    }
}

template<class floating>
__global__ void smoothFullScaleKernel(const int N, const int M, const floating * const h, floating * const X)
{
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < N*M)
    {
        const size_t j = idx / N;

        if (1 <= j && j < M-1)
        {
            const size_t i = idx - j*N;

            const floating inv_scale_factor = 1/(h[j-1] + h[j]);
            const floating inv_coeff_middle = 1/(2*(h[j-1] + h[j])/h[j-1]/h[j]);

            if (i == 0 || i == 1 || i == N-1)
                X[idx] *= inv_scale_factor;
            else
                X[idx] *= inv_coeff_middle;
        }
    }
}

template<class floating>
void deviceProlongation(const int N, const int M, const floating *const h, const floating *const ff,
                        floating *const solution) {
    const int block_size = 1024;
    const int number_of_blocks = N*M / block_size + 1;
    dim3 gridDim(number_of_blocks, 1);
    dim3 blockDim(block_size, 1);
    prolongationKernel<floating> <<<gridDim, blockDim>>>(N, M, h, ff, solution);

    cudaDeviceSynchronize();
}

template<class floating>
void deviceRescaleRhs(const int N, const int M, const floating * const h, floating * const rhs)
{
    const int block_size = 1024;
    const int number_of_blocks = N*M / block_size + 1;
    dim3 gridDim(number_of_blocks, 1);
    dim3 blockDim(block_size, 1);
    rescaleRhsKernel<floating> <<<gridDim, blockDim>>>(N, M, h, rhs);
    cudaDeviceSynchronize();
}

template<class floating>
void deviceRestriction(const int N, const int M, const floating * const h, const floating * const ff, floating * const ffOnCoarseGrid)
{
    const int block_size = 1024;
    const int number_of_blocks = N*M / block_size + 1;
    dim3 gridDim(number_of_blocks, 1);
    dim3 blockDim(block_size, 1);
    restrictionKernel<floating> <<<gridDim, blockDim>>>(N, M, h, ff, ffOnCoarseGrid);
    cudaDeviceSynchronize();
}

template<class floating>
void deviceGetReducedGrid(const int coarseLen, const floating * const h, floating * const coarseH)
{
    const int block_size = 1024;
    const int number_of_blocks = coarseLen / block_size + 1;
    dim3 gridDim(number_of_blocks, 1);
    dim3 blockDim(block_size, 1);
    getReducedGridKernel<floating> <<<gridDim, blockDim>>>(coarseLen, h, coarseH);
    cudaDeviceSynchronize();
}

template<class floating>
void deviceSmoothScale(const int N, const int i, const floating * const h, floating * const x)
{
    const int block_size = 1024;
    const int number_of_blocks = N / block_size + 1;
    dim3 gridDim(number_of_blocks, 1);
    dim3 blockDim(block_size, 1);
    smoothScaleKernel<floating> <<<gridDim, blockDim>>>(N, i, h, x);
    cudaDeviceSynchronize();
}

template<class floating>
void deviceSmoothFullScale(const int N, const int M, const floating * const h, floating * const X)
{
    const int block_size = 1024;
    const int number_of_blocks = N*M / block_size + 1;
    dim3 gridDim(number_of_blocks, 1);
    dim3 blockDim(block_size, 1);
    smoothFullScaleKernel<floating> <<<gridDim, blockDim>>>(N, M, h, X);
    cudaDeviceSynchronize();
}

template void deviceProlongation<float> (const int N, const int M, const float * const h, float const * const ff, float * const solution);
template void deviceProlongation<double>(const int N, const int M, const double * const h, double const * const ff, double * const solution);
template void deviceRescaleRhs<float> (const int N, const int M, const float * const h, float * const rhs);
template void deviceRescaleRhs<double>(const int N, const int M, const double * const h, double * const rhs);
template void deviceRestriction<float> (const int N, const int M, const float * const h, const float * const ffOnCoarseGrid, float * const ff);
template void deviceRestriction<double>(const int N, const int M, const double * const h, const double * const ffOnCoarseGrid, double * const ff);
template void deviceGetReducedGrid<float> (const int coarseLen, const float * const h, float * const coarseH);
template void deviceGetReducedGrid<double>(const int coarseLen, const double * const h, double * const coarseH);
template void deviceSmoothScale<float> (const int N, const int i, const float * const h, float * const x);
template void deviceSmoothScale<double>(const int N, const int i, const double * const h, double * const x);
template void deviceSmoothFullScale<float> (const int N, const int M, const float * const h, float * const X);
template void deviceSmoothFullScale<double>(const int N, const int M, const double * const h, double * const X);