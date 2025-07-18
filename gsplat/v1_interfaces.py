import math
import torch
from typing import Tuple, Literal, Optional
from jaxtyping import Float, Int
from torch import Tensor
import gsplat.cuda._wrapper as wrapper
from .rasterize_to_visibilities import rasterize_to_visibilities


def get_intrinsics_matrix(fx, fy, cx, cy, device):
    K = torch.eye(3, device=device)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    return K

class _RasterizeToPixels2DGS(torch.autograd.Function):
    """Rasterize gaussians"""

    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,  # [N, 2]
        ray_transforms: Tensor,  # [C, N, 3, 3]
        colors: Tensor,  # [C, N, D]
        opacities: Tensor,  # [C, N]
        normals: Tensor, # [C, N, 3]
        densify: Tensor, # [C, N, 2]
        backgrounds: Tensor,  # [C, D], Optional
        masks: Tensor,  # [C, tile_height, tile_width], Optional
        width: int,
        height: int,
        tile_size: int,
        isect_offsets: Tensor,  # [C, tile_height, tile_width]
        flatten_ids: Tensor,  # [n_isects]
        absgrad: bool,
        distloss: bool,
    ) -> Tuple[Tensor, Tensor]:
        (
            render_colors,
            render_alphas,
            render_normals,
            render_distort,
            render_median,
            last_ids,
            median_ids,
        ) = wrapper._make_lazy_cuda_func("rasterize_to_pixels_fwd_2dgs")(
            means2d.unsqueeze(0),
            ray_transforms,
            colors,
            opacities,
            normals,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
        )

        ctx.save_for_backward(
            means2d,
            ray_transforms,
            colors,
            opacities,
            normals,
            densify,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
        )
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size
        ctx.absgrad = absgrad
        ctx.distloss = distloss

        # doubel to float
        render_alphas = render_alphas.float()
        return (
            render_colors,
            render_alphas,
            render_normals,
            render_distort,
            render_median,
        )

    @staticmethod
    def backward(
        ctx,
        v_render_colors: Tensor,  # [C, H, W, 3]
        v_render_alphas: Tensor,  # [C, H, W, 1]
        v_render_normals: Tensor,
        v_render_distort: Tensor,
        v_render_median: Tensor,
    ):
        
        (
            means2d,
            ray_transforms,
            colors,
            opacities,
            normals,
            densify,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        tile_size = ctx.tile_size
        absgrad = ctx.absgrad

        (
            v_means2d_abs,
            v_means2d,
            v_ray_transforms,
            v_colors,
            v_opacities,
            v_normals,
            v_densify,
        ) = wrapper._make_lazy_cuda_func("rasterize_to_pixels_bwd_2dgs")(
            means2d,
            ray_transforms,
            colors,
            opacities,
            normals,
            densify,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
            v_render_normals.contiguous(),
            v_render_distort.contiguous(),
            v_render_median.contiguous(),
            absgrad,
        )
        torch.cuda.synchronize()
        if absgrad:
            means2d.absgrad = v_means2d_abs.squeeze(0)

        if ctx.needs_input_grad[6]:
            v_backgrounds = (v_render_colors * (1.0 - render_alphas).float()).sum(
                dim=(1, 2)
            )
        else:
            v_backgrounds = None

        return (
            v_means2d,
            v_ray_transforms,
            v_colors,
            v_opacities,
            v_normals,
            v_densify,
            v_backgrounds,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def rasterize_to_pixels_2dgs(
    means2d: Tensor,  # [N, 2]
    ray_transforms: Tensor,  # [C, N, 3, 3] or [nnz, channels]
    colors: Tensor,  # [C, N, channels] or [nnz, channels]
    opacities: Tensor,  # [C, N] or [nnz]
    normals: Tensor, # [C, N, 3] or [nnz, 3]
    densify: Tensor, # [C, N, 2] or [nnz, 3]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [C, tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    backgrounds: Optional[Tensor] = None,  # [C, channels]
    masks: Optional[Tensor] = None,  # [C, tile_height, tile_width]
    packed: bool = False,
    absgrad: bool = False,
    distloss: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Rasterize Gaussians to pixels.

    Args:
        means2d: Projected Gaussian means. [C, N, 2] if packed is False, [nnz, 2] if packed is True.
        ray_transforms: transformation matrices that transforms xy-planes in pixel spaces into splat coordinates. [C, N, 3, 3] if packed is False, [nnz, channels] if packed is True.
        colors: Gaussian colors or ND features. [C, N, channels] if packed is False, [nnz, channels] if packed is True.
        opacities: Gaussian opacities that support per-view values. [C, N] if packed is False, [nnz] if packed is True.
        normals: The normals in camera space. [C, N, 3] if packed is False, [nnz, 3] if packed is True.
        densify: Dummy variable to keep track of gradient for densification. [C, N, 2] if packed, [nnz, 3] if packed is True.
        tile_size: Tile size.
        isect_offsets: Intersection offsets outputs from `isect_offset_encode()`. [C, tile_height, tile_width]
        flatten_ids: The global flatten indices in [C * N] or [nnz] from  `isect_tiles()`. [n_isects]
        backgrounds: Background colors. [C, channels]. Default: None.
        masks: Optional tile mask to skip rendering GS to masked tiles. [C, tile_height, tile_width]. Default: None.
        packed: If True, the input tensors are expected to be packed with shape [nnz, ...]. Default: False.
        absgrad: If True, the backward pass will compute a `.absgrad` attribute for `means2d`. Default: False.

    Returns:
        A tuple:

        - **Rendered colors**.      [C, image_height, image_width, channels]
        - **Rendered alphas**.      [C, image_height, image_width, 1]
        - **Rendered normals**.     [C, image_height, image_width, 3]
        - **Rendered distortion**.  [C, image_height, image_width, 1]
        - **Rendered median depth**.[C, image_height, image_width, 1]


    """
    C = isect_offsets.size(0)
    device = means2d.device
    assert packed is False
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.shape
        assert ray_transforms.shape == (nnz, 3, 3), ray_transforms.shape
        assert colors.shape[0] == nnz, colors.shape
        assert opacities.shape == (nnz,), opacities.shape
    else:
        N = means2d.size(0)
        assert means2d.shape == (N, 2), means2d.shape
        assert ray_transforms.shape == (C, N, 3, 3), ray_transforms.shape
        assert colors.shape[:2] == (C, N), colors.shape
        assert opacities.shape == (C, N), opacities.shape
    if backgrounds is not None:
        assert backgrounds.shape == (C, colors.shape[-1]), backgrounds.shape
        backgrounds = backgrounds.contiguous()
    if masks is not None:
        assert masks.shape == isect_offsets.shape, masks.shape
        masks = masks.contiguous()

    # Pad the channels to the nearest supported number if necessary
    channels = colors.shape[-1]
    if channels > 513 or channels == 0:
        # TODO: maybe worth to support zero channels?
        raise ValueError(f"Unsupported number of color channels: {channels}")
    if channels not in (1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        colors = torch.cat(
            [colors, torch.zeros(*colors.shape[:-1], padded_channels, device=device)],
            dim=-1,
        )
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(
                        *backgrounds.shape[:-1], padded_channels, device=device
                    ),
                ],
                dim=-1,
            )
    else:
        padded_channels = 0
    tile_height, tile_width = isect_offsets.shape[1:3]
    assert (
        tile_height * tile_size >= image_height
    ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert (
        tile_width * tile_size >= image_width
    ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    (
        render_colors,
        render_alphas,
        render_normals,
        render_distort,
        render_median,
    ) = _RasterizeToPixels2DGS.apply(
        means2d.contiguous(),
        ray_transforms.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        normals.contiguous(),
        densify.contiguous(),
        backgrounds,
        masks,
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
        absgrad,
        distloss,
    )

    if padded_channels > 0:
        render_colors = render_colors[..., :-padded_channels]

    return render_colors, render_alphas, render_normals, render_distort, render_median

def spherical_harmonics(
    degrees_to_use: int,
    viewdirs: Float[Tensor, "*batch 3"],
    coeffs: Float[Tensor, "*batch D C"],
    method: Literal["poly", "fast"] = "fast",
) -> Float[Tensor, "*batch C"]:
    return wrapper.spherical_harmonics(
        degrees_to_use=degrees_to_use,
        dirs=viewdirs,
        coeffs=coeffs,
    )

def get_local2j_ids_bool(
    image_height: int,
    image_width: int,
    mp_rank: int,
    mp_world_size: int,
    means2D: Float[Tensor, "P 2"],
    radii,
    dist_global_strategy_tensor,
    avoid_pixel_all2all
) -> Float[Tensor, "P mp_world_size"]:
    """Get local to global ids for the given image size and means2D.

    Args:
        image_height: Image height.
        image_width: Image width.
        mp_rank: Current rank.
        mp_world_size: Total number of ranks.
        means2D: The 2D means of the Gaussians. [P, 2]

    Returns:
        local2j_ids_bool: Local to global ids boolean tensor. [P, mp_world_size]
    """
    P = means2D.size(0)

    assert P > 0, "No Gaussians to rasterize"

    args = (
        image_height,
        image_width,
        mp_rank,
        mp_world_size,
        means2D,
        radii,
        dist_global_strategy_tensor,
        avoid_pixel_all2all,
    )

    local2j_ids_bool = wrapper._make_lazy_cuda_func("get_local2j_ids_bool")(*args)

    return local2j_ids_bool