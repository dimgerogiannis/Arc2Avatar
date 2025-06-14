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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH,SH2RGB
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation


# from .resnet import *

from scipy.spatial import KDTree
import torch

def ndistCUDA2(points):
    points_np = points.detach().cpu().float().numpy()
    dists, inds = KDTree(points_np).query(points_np, k=4)
    meanDists = (dists[:, 1:] ** 2).mean(1)

    return torch.tensor(meanDists, dtype=points.dtype, device=points.device)

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._background = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.mask = None
        self.masked_faces = None
        self.faces = None
        self.vertices = None
        self.longer_mask = None
        self.setup_functions()
        self.load_mask()

    def clip_grad_norm(self, val):
        torch.nn.utils.clip_grad_norm_(self._xyz, val)
        torch.nn.utils.clip_grad_norm_(self._features_dc, val)
        torch.nn.utils.clip_grad_norm_(self._features_rest, val)
        torch.nn.utils.clip_grad_norm_(self._opacity, val)
        torch.nn.utils.clip_grad_norm_(self._scaling, val)
        torch.nn.utils.clip_grad_norm_(self._rotation, val)

    def load_mask(self, mask_file="scene/template/masked_template.ply"):
        ply_data = PlyData.read(mask_file)
        
        face_vertices = np.stack((ply_data['vertex']['x'], 
                                ply_data['vertex']['y'], 
                                ply_data['vertex']['z']), axis=1)
        colors = np.stack((ply_data['vertex']['red'], 
                        ply_data['vertex']['green'], 
                        ply_data['vertex']['blue']), axis=1)

        red_vertices_mask = (colors[:, 0] > 200) & (colors[:, 1] < 50) & (colors[:, 2] < 50)

        self.mask = torch.tensor(red_vertices_mask, dtype=torch.bool, device="cuda") 

        masked_vertex_indices = np.where(red_vertices_mask)[0]

        index_mapping = {orig_idx: new_idx for new_idx, orig_idx in enumerate(masked_vertex_indices)}

        faces = np.array([face[0] for face in ply_data['face'].data])

        faces_mask = red_vertices_mask[faces].all(axis=1)

        masked_faces = faces[faces_mask]

        remapped_faces = np.array([[index_mapping[idx] for idx in face] for face in masked_faces])

        self.masked_faces = torch.tensor(remapped_faces, dtype=torch.long, device="cuda")

        faces_converted = faces.astype(np.int64)
        self.faces = torch.tensor(faces_converted, dtype=torch.long, device="cuda")


        ply_data = PlyData.read("scene/template/long_masked_template.ply")
        
        face_vertices = np.stack((ply_data['vertex']['x'], 
                                ply_data['vertex']['y'], 
                                ply_data['vertex']['z']), axis=1)
        colors = np.stack((ply_data['vertex']['red'], 
                        ply_data['vertex']['green'], 
                        ply_data['vertex']['blue']), axis=1)

        red_vertices_mask = (colors[:, 0] > 200) & (colors[:, 1] < 50) & (colors[:, 2] < 50)

        self.longer_mask = torch.tensor(red_vertices_mask, dtype=torch.bool, device="cuda") 

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.mask,
            self.masked_faces,
            self.longer_mask,
        )

    def initialize(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale, self.mask, _, self.masked_faces, _,
            self.longer_mask) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale, self.mask, self.masked_faces,
            self.longer_mask) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_background(self):
        return torch.sigmoid(self._background)

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
    
    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(pcd.colors)).float().cuda()
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = torch.clamp_min(ndistCUDA2(fused_point_cloud), 1e-7)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._background = nn.Parameter(torch.zeros((3, 1, 1), device="cuda").requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.normals = torch.from_numpy(pcd.normals).to(self._xyz.device).float()
        self.normals.requires_grad = False

    def training_setup(self, training_args, *, expression_correction: bool = False):
        """
        Builds the optimiser & LR schedulers.
        Set use_init=True to activate the *init_* hyper-parameter set.
        """
        def P(name: str):
            return getattr(training_args,
                        f"{'expr_' if expression_correction else ''}{name}")

        self.percent_dense      = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom              = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz],
            'lr'   : P('position_lr_init') * self.spatial_lr_scale,
            "name" : "xyz"},

            {'params': [self._features_dc],
            'lr'   : P('feature_lr'),
            "name" : "f_dc"},
            {'params': [self._features_rest],
            'lr'   : P('feature_lr'),
            "name" : "f_rest"},

            {'params': [self._opacity],
            'lr'   : P('opacity_lr'),
            "name" : "opacity"},
            {'params': [self._scaling],
            'lr'   : P('scaling_lr'),
            "name" : "scaling"},

            {'params': [self._rotation],
            'lr'   : P('rotation_lr'),
            "name" : "rotation"},
            {'params': [self._background],
            'lr'   : P('feature_lr'),
            "name" : "background"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        def _masked_hook(factor: float):
            if factor == 1.0:
                return None
            def _hook(grad):
                g = grad.clone()
                g[self.mask] *= factor
                return g
            return _hook

        scale_fac = getattr(training_args, 'scaling_lr_masked',
                            training_args.scaling_lr) / P('scaling_lr')
        feat_fac  = getattr(training_args, 'feature_lr_masked',
                            training_args.feature_lr) / P('feature_lr')
        
        opacity_fac = getattr(training_args, 'opacity_lr_masked',
                            training_args.opacity_lr) / P('opacity_lr')

        if (h := _masked_hook(scale_fac)): self._scaling.register_hook(h)
        if (h := _masked_hook(feat_fac)):
            self._features_dc.register_hook(h)
            self._features_rest.register_hook(h)

        if (h := _masked_hook(opacity_fac)):
            self._opacity.register_hook(h)

        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init       = P('position_lr_init')  * self.spatial_lr_scale,
            lr_final      = P('position_lr_final') * self.spatial_lr_scale,
            lr_delay_mult = training_args.position_lr_delay_mult,
            max_steps     = training_args.iterations)

        self.scaling_scheduler_args = get_expon_lr_func(
            lr_init       = P('scaling_lr'),
            lr_final      = P('scaling_lr_final'),
            lr_delay_mult = training_args.position_lr_delay_mult,
            max_steps     = training_args.iterations)

        self.feature_scheduler_args = get_expon_lr_func(
            lr_init       = P('feature_lr'),
            lr_final      = P('feature_lr_final'),
            lr_delay_mult = training_args.position_lr_delay_mult,
            max_steps     = training_args.iterations)

        self.rotation_scheduler_args = get_expon_lr_func(
            lr_init       = P('rotation_lr'),
            lr_final      = P('rotation_lr_final'),
            lr_delay_mult = training_args.position_lr_delay_mult,
            max_steps     = training_args.iterations)
        
        self.update_learning_rate(1)
        self.update_feature_learning_rate(1)
        self.update_scaling_learning_rate(1)
        self.update_rotation_learning_rate(1)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def update_feature_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "f_dc":
                lr = self.feature_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def update_rotation_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "rotation":
                lr = self.rotation_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def update_scaling_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "scaling":
                lr = self.scaling_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

                
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        np.savetxt(os.path.join(os.path.split(path)[0],"point_cloud_rgb.txt"),np.concatenate((xyz, SH2RGB(f_dc)), axis=1))

    def reset_opacity(self):
        if not self.mask.any():
            return

        reset_mask = ~self.mask

        opacities_new = inverse_sigmoid(
            torch.min(
                self.get_opacity[reset_mask],
                torch.ones_like(self.get_opacity[reset_mask]) * 0.2
            )
        )

        new_opacity = self.get_opacity.clone()  
        new_opacity[reset_mask] = opacities_new 

        optimizable_tensors = self.replace_tensor_to_optimizer(new_opacity, "opacity")

        self._opacity = optimizable_tensors["opacity"]

        assert self._opacity.shape[0] == self.mask.shape[0], (
            f"Mismatch between _opacity and mask size! _opacity: {self._opacity.shape[0]}, mask: {self.mask.shape[0]}"
        )


    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] not in ['background']:
                if group["name"] == name:
                    assert group["params"][0].shape == tensor.shape, (
                        f"Size mismatch when replacing tensor '{name}'. "
                        f"Existing size: {group['params'][0].shape}, New size: {tensor.shape}"
                    )
                    
                    stored_state = self.optimizer.state.get(group['params'][0], None)
                    if stored_state is not None:
                        stored_state["exp_avg"] = torch.zeros_like(tensor)
                        stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    
                    del self.optimizer.state[group['params'][0]]  
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state 
                    
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if group["name"] not in ['background']:
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, prune_mask):
        valid_prune_mask = ~prune_mask

        if valid_prune_mask.sum() == 0:
            return

        optimizable_tensors = self._prune_optimizer(valid_prune_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.mask = self.mask[valid_prune_mask]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_prune_mask]
        self.denom = self.denom[valid_prune_mask]
        self.max_radii2D = self.max_radii2D[valid_prune_mask]

        assert self._xyz.shape[0] == self.mask.shape[0], f"Mismatch between _xyz and mask size after pruning! _xyz: {self._xyz.shape[0]}, mask: {self.mask.shape[0]}"
        assert self._opacity.shape[0] == self.mask.shape[0], f"Mismatch between _opacity and mask size after pruning! _opacity: {self._opacity.shape[0]}, mask: {self.mask.shape[0]}"

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] not in ['background']:
                assert len(group["params"]) == 1, f"Group {group['name']} has more than one parameter."
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group['params'][0], None)
                                
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        
        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, mask=False):
        num_new_points = new_xyz.shape[0]

        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        new_mask_values = torch.zeros((num_new_points,), dtype=torch.bool, device="cuda")
        self.mask = torch.cat((self.mask, new_mask_values), dim=0)


        assert self._xyz.shape[0] == self.mask.shape[0], f"Mismatch between _xyz and mask size! _xyz: {self._xyz.shape[0]}, mask: {self.mask.shape[0]}"
        assert self._opacity.shape[0] == self.mask.shape[0], f"Mismatch between _opacity and mask size! _opacity: {self._opacity.shape[0]}, mask: {self.mask.shape[0]}"

        return num_new_points

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        )
        
        valid_selection_mask = selected_pts_mask & ~self.mask[:n_init_points] 

        stds = self.get_scaling[valid_selection_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[valid_selection_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[valid_selection_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[valid_selection_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[valid_selection_mask].repeat(N, 1)
        new_features_dc = self._features_dc[valid_selection_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[valid_selection_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[valid_selection_mask].view(-1, 1).repeat(N, 1) 

        num_points = self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((valid_selection_mask, torch.zeros(num_points, device="cuda", dtype=bool)))

        self.prune_points(prune_filter)
        return num_points

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent
        )
        

        valid_selection_mask = selected_pts_mask & ~self.mask[:self.get_xyz.shape[0]] #

        new_xyz = self._xyz[valid_selection_mask]
        new_features_dc = self._features_dc[valid_selection_mask]
        new_features_rest = self._features_rest[valid_selection_mask]
        new_opacities = self._opacity[valid_selection_mask]
        new_scaling = self._scaling[valid_selection_mask]
        new_rotation = self._rotation[valid_selection_mask]

        num_points = self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
        return num_points

    def prune_below_masked_points(self, epsilon=0):
        if not self.mask.any():
            return

        y_coords_masked = self._xyz[self.mask][:, 2]

        y_min_masked = y_coords_masked.min()

        y_threshold = y_min_masked + epsilon

        prune_mask = self._xyz[:, 2] < y_threshold

        prune_mask = prune_mask & (~self.mask)

        self.prune_points(prune_mask)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0  

        num_points_1 = self.densify_and_clone(grads, max_grad, extent)
        num_points_2 = self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(prune_mask, big_points_vs)
            prune_mask = torch.logical_or(prune_mask, big_points_ws)

        prune_mask = prune_mask & ~self.mask 

        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1