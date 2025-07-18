#include "bindings.h"
#include "helpers.cuh"
#include "types.cuh"
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace gsplat {

namespace cg = cooperative_groups;

#define BLOCK_X 16
#define BLOCK_Y 16

/****************************************************************************
 * Utility tools
 ****************************************************************************/

__forceinline__ __host__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

__global__ void getTouchedIdsBool(
	int P,
	int height,
	int width,
	int world_size,
	const float2* means2D,
	const int* radii,// NOTE: radii is not const in getRect()
	const int* dist_global_strategy,
	bool* touchedIdsBool,
	bool avoid_pixel_all2all)
{
	auto i = cg::this_grid().thread_rank();
	if (i < P)
	{
		uint2 rect_min, rect_max;
		dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);

		getRect(means2D[i], radii[i], rect_min, rect_max, tile_grid);

		// printf("mean2D: %f %f, radii:%d\n", means2D[i].x, means2D[i].y, radii[i]);
		
		// method 1:
		int touched_min_tile_idx = rect_min.y * tile_grid.x + rect_min.x;
		int touched_max_tile_idx = (rect_max.y - 1 ) * tile_grid.x + rect_max.x - 1;

		if ( touched_max_tile_idx < touched_min_tile_idx )
			return;
		
		// 记录每个GS应该在哪个rank上
		for (int rk = 0; rk < world_size; rk++)
		{
			int tile_l = *(dist_global_strategy+rk);
			int tile_r = *(dist_global_strategy+rk+1);
			if (avoid_pixel_all2all) {
				// we could avoid the pixel all2all by rendering the pixels that are near border and out of border. 
				tile_l -= tile_grid.x+1;
				tile_r += tile_grid.x+1;
			}

			if (touched_max_tile_idx < tile_l || touched_min_tile_idx >= tile_r)
				continue;
			
			// TODO: If one worker's tiles are fewer than one row, then it is buggy. 
			// If we have other workload_division dimension, then we need to change this. 
			touchedIdsBool[i * world_size + rk] = true;
		}
	}
}

torch::Tensor get_local2j_ids_bool(
    int image_height,
	int image_width,
	int mp_rank,
	int mp_world_size,
	const torch::Tensor& means2D,
	const torch::Tensor& radii,
	const torch::Tensor& dist_global_strategy,
	bool avoid_pixel_all2all)
{
    const int P = means2D.size(0);
	const int H = image_height;
	const int W = image_width;

	torch::Tensor local2jIdsBool = torch::full({P, mp_world_size}, false, means2D.options().dtype(torch::kBool));

	getTouchedIdsBool << <(P + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS, GSPLAT_N_THREADS >> >(
		P,
		H,
		W,
		mp_world_size,
		reinterpret_cast<float2*>(means2D.contiguous().data<float>()),
		radii.contiguous().data<int>(),
		dist_global_strategy.contiguous().data<int>(),
		local2jIdsBool.contiguous().data<bool>(),
		avoid_pixel_all2all
	);

	return local2jIdsBool;
}

} // namespace gsplat