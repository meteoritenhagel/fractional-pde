#ifndef TR_HACK_GPUKERNELS_CUH
#define TR_HACK_GPUKERNELS_CUH

/**
 * @warning This is a Gpu kernel, so probably you would like to call the wrapper device_prolongation instead.
 *
 * Given classical C-style arrays @p grid, @p coarse_rhs and @p fine_rhs in the Gpu memory, this function
 * applies the (linear interpolation) prolongation to the coarse-grid right-hand side @p coarse_rhs.
 *
 * @tparam floating floating point type
 * @param[in] block_dim the order of a single square block in the block matrix system
 * @param[in] num_blocks the number of blocks per row resp. columns in the block matrix system
 * @param[in] grid vector of space grid interval lengths, must have @p num_blocks elements
 * @param[in] coarse_rhs coarse-grid right-hand side, must have @p block_dim x (@p num_blocks+1)/2 elements
 * @param[out] fine_rhs prolongation of @p coarse_rhs onto the fine grid, must be pre-allocated with @p block_dim x @p num_blocks elements
 */
template<class floating>
__global__ void prolongation_kernel(const int block_dim, const int num_blocks, const floating * const grid,
                                    floating const * const coarse_rhs, floating * const fine_rhs);

/**
 * @warning This is a Gpu kernel, so probably you would like to call the wrapper device_rescale_rhs instead.
 *
 * Given classical C-style arrays @p grid and @p rhs in the Gpu memory, this function
 * rescales the right-hand side @p rhs according to the grid @p grid to match the symmetrized linear system.
 *
 * @tparam floating floating point type
 * @param[in] block_dim the order of a single square block in the block matrix system
 * @param[in] num_blocks the number of blocks per row resp. columns in the block matrix system
 * @param[in] grid vector of space grid interval lengths, must have @p num_blocks-1 elements
 * @param[in,out] rhs rhs being rescaled, must have @p block_dim x @p num_blocks elements
 */
template<class floating>
__global__ void rescale_rhs_kernel(const int block_dim, const int num_blocks,
                                   const floating * const grid, floating * const rhs);

/**
 * @warning This is a Gpu kernel, so probably you would like to call the wrapper device_restriction instead.
 *
 * Given classical C-style arrays @p grid, @p fine_rhs and @p coarse_rhs in the Gpu memory, this function
 * restricts the fine-grid right-hand side vector @p fine_rhs onto the coarse grid.
 *
 * @tparam floating floating point type
 * @param[in] block_dim the order of a single square block in the block matrix system
 * @param[in] num_blocks the number of blocks per row resp. columns in the block matrix system
 * @param[in] grid vector of space grid interval lengths, must have @p num_blocks-1 elements
 * @param[in] fine_rhs fine-grid right-hand side vector
 * @param[out] coarse_rhs restriction of @p fine_rhs onto the fine grid, must be allocated with @p block_dim x (@p num_blocks+1)/2 elements
 */
template<class floating>
__global__ void restriction_kernel(const int block_dim, const int num_blocks, const floating * const grid,
                                   const floating * const fine_rhs, floating * const coarse_rhs);

/**
 * @warning This is a Gpu kernel, so probably you would like to call the wrapper device_get_reduced_grid instead.
 *
 * Given classical C-style arrays @p grid and @p coarse_grid,
 * this function calculates the grid (in form of the space grid interval lengths)
 * when leaving each second point out.
 *
 * @tparam floating floating point type
 * @param[in] coarse_len length of coarse grid (equals length of @p grid / 2)
 * @param[in] grid vector of fine space grid interval lengths
 * @param[out] coarse_grid vector of coarse space grid interval length, must be allocated with @p coarse_len elements
 */
template<class floating>
__global__ void get_reduced_grid_kernel(const int coarse_len, const floating * const grid, floating * const coarse_grid);

/**
 * @warning This is a Gpu kernel, so probably you would like to call the wrapper device_smooth_scale instead.
 *
 * Given classical C-style arrays @p grid and @p residual_i,
 * this function scales the @p residual_i as needed for Gauss-Seidel iterations.
 *
 * @tparam floating floating point type
 * @param[in] block_dim the order of a single square block in the block matrix system
 * @param[in] i block position of residual (needed for correct scaling)
 * @param[in] grid vector of fine space grid interval lengths
 * @param[in,out] residual_i @p i_th block of the residual being scaled and smoothed
 */
template<class floating>
__global__ void smooth_scale_kernel(const int block_dim, const int i, const floating * const grid, floating * const residual_i);

/**
 * @warning This is a Gpu kernel, so probably you would like to call the wrapper device_smooth_full_scale instead.
 *
 * Given classical C-style arrays @p grid and @p residual,
 * this function scales the full @p residual as needed for Jacobi iterations.
 *
 * @tparam floating floating point type
 * @param[in] block_dim the order of a single square block in the block matrix system
 * @param[in] num_blocks the number of blocks per row resp. columns in the block matrix system
 * @param[in] grid grid vector of space grid interval lengths, must have @p num_blocks-1 elements
 * @param[in,out] residual residual being rescaled, must have @p block_dim x @p num_blocks elements
 */
template<class floating>
__global__ void smooth_full_scale_kernel(const int block_dim, const int num_blocks, const floating * const grid, floating * const residual);

/**
 * Given classical C-style arrays @p grid, @p coarse_rhs and @p fine_rhs in the Gpu memory, this function
 * applies the (linear interpolation) prolongation to the coarse-grid right-hand side @p coarse_rhs.
 *
 * @tparam floating floating point type
 * @param[in] block_dim the order of a single square block in the block matrix system
 * @param[in] num_blocks the number of blocks per row resp. columns in the block matrix system
 * @param[in] grid vector of space grid interval lengths, must have @p num_blocks elements
 * @param[in] coarse_rhs coarse-grid right-hand side, must have @p block_dim x (@p num_blocks+1)/2 elements
 * @param[out] fine_rhs prolongation of @p coarse_rhs onto the fine grid, must be pre-allocated with @p block_dim x @p num_blocks elements
 */
template<class floating>
void device_prolongation(const int block_dim, const int num_blocks, const floating * const grid, floating const * const coarse_rhs, floating * const fine_rhs);

/**
 * Given classical C-style arrays @p grid and @p rhs in the Gpu memory, this function
 * rescales the right-hand side @p rhs according to the grid @p grid to match the symmetrized linear system.
 *
 * @tparam floating floating point type
 * @param[in] block_dim the order of a single square block in the block matrix system
 * @param[in] num_blocks the number of blocks per row resp. columns in the block matrix system
 * @param[in] grid vector of space grid interval lengths, must have @p num_blocks-1 elements
 * @param[in,out] rhs rhs being rescaled, must have @p block_dim x @p num_blocks elements
 */
template<class floating>
void device_rescale_rhs(const int block_dim, const int num_blocks, const floating * const grid, floating * const rhs);

/**
 * Given classical C-style arrays @p grid, @p fine_rhs and @p coarse_rhs in the Gpu memory, this function
 * restricts the fine-grid right-hand side vector @p fine_rhs onto the coarse grid.
 *
 * @tparam floating floating point type
 * @param[in] block_dim the order of a single square block in the block matrix system
 * @param[in] num_blocks the number of blocks per row resp. columns in the block matrix system
 * @param[in] grid vector of space grid interval lengths, must have @p num_blocks-1 elements
 * @param[in] fine_rhs fine-grid right-hand side vector
 * @param[out] coarse_rhs restriction of @p fine_rhs onto the fine grid, must be allocated with @p block_dim x (@p num_blocks+1)/2 elements
 */
template<class floating>
void device_restriction(const int block_dim, const int num_blocks, const floating * const grid, const floating * const fine_rhs, floating * const coarse_rhs);

/**
 * Given classical C-style arrays @p grid and @p coarse_grid,
 * this function calculates the grid (in form of the space grid interval lengths)
 * when leaving each second point out.
 *
 * @tparam floating floating point type
 * @param[in] coarse_len length of coarse grid (equals length of @p grid / 2)
 * @param[in] grid vector of fine space grid interval lengths
 * @param[out] coarse_grid vector of coarse space grid interval length, must be allocated with @p coarse_len elements
 */
template<class floating>
void device_get_reduced_grid(const int coarse_len, const floating * const grid, floating * const coarse_grid);

/**
 * Given classical C-style arrays @p grid and @p residual_i,
 * this function scales the @p residual_i as needed for Gauss-Seidel iterations.
 *
 * @tparam floating floating point type
 * @param[in] block_dim the order of a single square block in the block matrix system
 * @param[in] i block position of residual (needed for correct scaling)
 * @param[in] grid vector of fine space grid interval lengths
 * @param[in,out] residual_i @p i_th block of the residual being scaled and smoothed
 */
template<class floating>
void device_smooth_scale(const int block_dim, const int i, const floating * const grid, floating * const residual_i);

/**
 * Given classical C-style arrays @p grid and @p residual,
 * this function scales the full @p residual as needed for Jacobi iterations.
 *
 * @tparam floating floating point type
 * @param[in] block_dim the order of a single square block in the block matrix system
 * @param[in] num_blocks the number of blocks per row resp. columns in the block matrix system
 * @param[in] grid grid vector of space grid interval lengths, must have @p num_blocks-1 elements
 * @param[in,out] residual residual being rescaled, must have @p block_dim x @p num_blocks elements
 */
template<class floating>
void device_smooth_full_scale(const int block_dim, const int num_blocks, const floating * const grid, floating * const residual);

#endif //TR_HACK_GPUKERNELS_CUH
