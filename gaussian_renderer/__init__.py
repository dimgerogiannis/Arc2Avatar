#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
import random



def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, black_video=False,
           override_color=None, sh_deg_aug_ratio=0.1, bg_aug_ratio=0.3, shs_aug_ratio=1.0, scale_aug_ratio=1.0,
           test=False, end=False, mask=False):

    if mask:
        screenspace_points = torch.zeros_like(pc.get_xyz[pc.mask], dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda")
    else:
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda")
    try:
        screenspace_points.retain_grad()
    except:
        pass

    if black_video:
        bg_color = torch.zeros_like(bg_color)

    if random.random() < sh_deg_aug_ratio and not test:
        act_SH = 0
    else:
        act_SH = pc.active_sh_degree

    if random.random() < bg_aug_ratio and not test:
        if random.random() < 0.5:
            bg_color = torch.rand_like(bg_color)
        else:
            bg_color = torch.zeros_like(bg_color)

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    image_height = int(viewpoint_camera.image_height * (2 if end else 1))
    image_width = int(viewpoint_camera.image_width * (2 if end else 1))

    try:
        raster_settings = GaussianRasterizationSettings(
            image_height=image_height,
            image_width=image_width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=act_SH,
            campos=viewpoint_camera.camera_center,
            prefiltered=False
        )
    except TypeError as e:
        raster_settings = GaussianRasterizationSettings(
            image_height=image_height,
            image_width=image_width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=act_SH,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False
        )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if mask:
        means3D = pc.get_xyz[pc.mask]
        opacity = pc.get_opacity[pc.mask]
        scales = pc.get_scaling[pc.mask]
        rotations = pc.get_rotation[pc.mask]
        shs = pc.get_features[pc.mask]
    else:
        means3D = pc.get_xyz
        opacity = pc.get_opacity
        scales = pc.get_scaling
        rotations = pc.get_rotation
        shs = pc.get_features
    

    if random.random() < shs_aug_ratio and not test:
        variance = (0.2 ** 0.5) * shs
        shs = shs + (torch.randn_like(shs) * variance)

    if random.random() < scale_aug_ratio and not test:
        variance = (0.2 ** 0.5) * scales / 4
        scales = torch.clamp(scales + (torch.randn_like(scales) * variance), 0.0)

    rendered_image, radii, depth_alpha = rasterizer(
        means3D=means3D,
        means2D=screenspace_points,
        shs=shs,
        colors_precomp=override_color,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None
    )

    depth, alpha = torch.chunk(depth_alpha, 2)

    focal = 1 / (2 * math.tan(viewpoint_camera.FoVx / 2))
    disp = focal / (depth + (alpha * 10) + 1e-5)

    try:
        min_d = disp[alpha <= 0.1].min()
    except Exception:
        min_d = disp.min()

    disp = torch.clamp((disp - min_d) / (disp.max() - min_d), 0.0, 1.0)

    return {
        "render": rendered_image,
        "depth": disp,
        "alpha": alpha,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "scales": scales
    }

