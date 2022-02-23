#include "gpukernels.cuh"

template<class floating>
__global__ void prolongation_kernel(const int block_dim, const int num_blocks, const floating *const grid, const floating *const coarse_rhs,
                                    floating *const fine_rhs) {
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < block_dim * num_blocks)
    {
        if (idx < block_dim) // 0th column
        {
            fine_rhs[idx] += coarse_rhs[idx];
        }
        else // jth column
        {
            const size_t j = idx / block_dim;
            const size_t i = idx - j * block_dim;

            if (j % 2) // odd number of columns, i.e. 1st, 3rd, 5th etc.
            {
                const auto coeff1 = grid[j] / (grid[j - 1] + grid[j]);
                const auto coeff2 = grid[j - 1] / (grid[j - 1] + grid[j]);
                fine_rhs[idx] += coeff1 * coarse_rhs[(j - 1) / 2 * block_dim + i] + coeff2 * coarse_rhs[(j + 1) / 2 * block_dim + i];
            }
            else // even number of columns, i.e. 2nd, 4th, 6th etc.
            {
                fine_rhs[idx] += coarse_rhs[(j + 1) / 2 * block_dim + i];
            }
        }
    }
}

template<class floating>
__global__ void rescale_rhs_kernel(const int block_dim, const int num_blocks, const floating * const grid, floating * const rhs)
{
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < block_dim * num_blocks)
    {
        const size_t j = idx / block_dim;

        if (1 <= j && j <= num_blocks - 2)
        {
            rhs[idx] *= grid[j] + grid[j - 1];
        }
    }
}

template<class floating>
__global__ void restriction_kernel(const int block_dim, const int num_blocks, const floating * const grid, const floating * const fine_rhs, floating * const coarse_rhs)
{
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;

    const size_t j = idx / block_dim;
    const size_t i = idx - j * block_dim;

    if (idx < block_dim * num_blocks && (j % 2 == 0) && 2 <= j && j < num_blocks - 1)
    {
        const auto coeff1 = grid[j - 2] / (grid[j - 2] + grid[j - 1]);
        const auto coeff2 = grid[j + 1] / (grid[j] + grid[j + 1]);
        coarse_rhs[j / 2 * block_dim + i] = coeff1 * fine_rhs[(j - 1) * block_dim + i] + fine_rhs[j * block_dim + i] + coeff2 * fine_rhs[(j + 1) * block_dim + i];

    }
}

template<class floating>
__global__ void get_reduced_grid_kernel(const int coarse_len, const floating * const grid, floating * const coarse_grid)
{
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < coarse_len)
        coarse_grid[idx] = grid[2 * idx] + grid[2 * idx + 1];
}

template<class floating>
__global__ void smooth_scale_kernel(const int block_dim, const int i, const floating * const grid, floating * const residual_i)
{
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < block_dim)
    {
        const floating inv_scale_factor = 1/(grid[i - 1] + grid[i]);
        const floating inv_coeff_middle = 1/(2 * (grid[i - 1] + grid[i]) / grid[i - 1] / grid[i]);
        if (idx == 0 || idx == 1 || idx == block_dim - 1)
            residual_i[idx] *= inv_scale_factor;
        else
            residual_i[idx] *= inv_coeff_middle;
    }
}

template<class floating>
__global__ void smooth_full_scale_kernel(const int block_dim, const int num_blocks, const floating * const grid, floating * const residual)
{
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < block_dim * num_blocks)
    {
        const size_t j = idx / block_dim;

        if (1 <= j && j < num_blocks - 1)
        {
            const size_t i = idx - j * block_dim;

            const floating inv_scale_factor = 1/(grid[j - 1] + grid[j]);
            const floating inv_coeff_middle = 1/(2 * (grid[j - 1] + grid[j]) / grid[j - 1] / grid[j]);

            if (i == 0 || i == 1 || i == block_dim - 1)
                residual[idx] *= inv_scale_factor;
            else
                residual[idx] *= inv_coeff_middle;
        }
    }
}

template<class floating>
void device_prolongation(const int block_dim, const int num_blocks, const floating *const grid, const floating *const coarse_rhs,
                         floating *const fine_rhs) {
    const int block_size = 1024;
    const int number_of_blocks = block_dim * num_blocks / block_size + 1;
    dim3 gridDim(number_of_blocks, 1);
    dim3 blockDim(block_size, 1);
    prolongation_kernel<floating> <<<gridDim, blockDim>>>(block_dim, num_blocks, grid, coarse_rhs, fine_rhs);

    cudaDeviceSynchronize();
}

template<class floating>
void device_rescale_rhs(const int block_dim, const int num_blocks, const floating * const grid, floating * const rhs)
{
    const int block_size = 1024;
    const int number_of_blocks = block_dim * num_blocks / block_size + 1;
    dim3 gridDim(number_of_blocks, 1);
    dim3 blockDim(block_size, 1);
    rescale_rhs_kernel<floating> <<<gridDim, blockDim>>>(block_dim, num_blocks, grid, rhs);
    cudaDeviceSynchronize();
}

template<class floating>
void device_restriction(const int block_dim, const int num_blocks, const floating * const grid, const floating * const fine_rhs, floating * const coarse_rhs)
{
    const int block_size = 1024;
    const int number_of_blocks = block_dim * num_blocks / block_size + 1;
    dim3 gridDim(number_of_blocks, 1);
    dim3 blockDim(block_size, 1);
    restriction_kernel<floating> <<<gridDim, blockDim>>>(block_dim, num_blocks, grid, fine_rhs, coarse_rhs);
    cudaDeviceSynchronize();
}

template<class floating>
void device_get_reduced_grid(const int coarse_len, const floating * const grid, floating * const coarse_grid)
{
    const int block_size = 1024;
    const int number_of_blocks = coarse_len / block_size + 1;
    dim3 gridDim(number_of_blocks, 1);
    dim3 blockDim(block_size, 1);
    get_reduced_grid_kernel<floating> <<<gridDim, blockDim>>>(coarse_len, grid, coarse_grid);
    cudaDeviceSynchronize();
}

template<class floating>
void device_smooth_scale(const int block_dim, const int i, const floating * const grid, floating * const residual_i)
{
    const int block_size = 1024;
    const int number_of_blocks = block_dim / block_size + 1;
    dim3 gridDim(number_of_blocks, 1);
    dim3 blockDim(block_size, 1);
    smooth_scale_kernel<floating> <<<gridDim, blockDim>>>(block_dim, i, grid, residual_i);
    cudaDeviceSynchronize();
}

template<class floating>
void device_smooth_full_scale(const int block_dim, const int num_blocks, const floating * const grid, floating * const residual)
{
    const int block_size = 1024;
    const int number_of_blocks = block_dim * num_blocks / block_size + 1;
    dim3 gridDim(number_of_blocks, 1);
    dim3 blockDim(block_size, 1);
    smooth_full_scale_kernel<floating> <<<gridDim, blockDim>>>(block_dim, num_blocks, grid, residual);
    cudaDeviceSynchronize();
}

template void device_prolongation<float> (const int, const int, const float * const, float const * const, float * const);
template void device_prolongation<double>(const int, const int, const double * const, double const * const, double * const);
template void device_rescale_rhs<float> (const int, const int, const float * const, float * const);
template void device_rescale_rhs<double>(const int, const int, const double * const, double * const);
template void device_restriction<float> (const int, const int, const float * const, const float * const, float * const);
template void device_restriction<double>(const int, const int, const double * const, const double * const, double * const);
template void device_get_reduced_grid<float> (const int, const float * const, float * const);
template void device_get_reduced_grid<double>(const int, const double * const, double * const);
template void device_smooth_scale<float> (const int, const int , const float * const, float * const);
template void device_smooth_scale<double>(const int, const int , const double * const, double * const);
template void device_smooth_full_scale<float> (const int, const int , const float * const, float * const);
template void device_smooth_full_scale<double>(const int, const int , const double * const, double * const);