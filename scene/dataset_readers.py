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
import random
import torch.nn.functional as F
from typing import NamedTuple
from utils.graphics_utils import focal2fov, fov2focal
import numpy as np
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import trimesh

class RandCameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    width: int
    height: int 
    delta_polar : np.array
    delta_azimuth : np.array
    delta_radius : np.array


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


class RSceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    test_cameras: list
    ply_path: str

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readCircleCamInfo(path, opt):
    print("Reading Test Transforms")
    test_cam_infos = GenerateCircleCameras(opt, render45=opt.render_45)

    ply_path = "scene/template/masked_template.ply"

    def load_ply(ply_file):
        """
        Loads a PLY file to extract vertices and vertex normals.
        
        Parameters:
            ply_file (str): Path to the PLY file.
        
        Returns:
            tuple:
                vertices (np.ndarray): Array of vertex positions with shape (n_vertices, 3).
                vertex_normals (np.ndarray or None): Array of vertex normals with shape (n_vertices, 3) if available, else None.
                _, _ : Placeholder for vertex colors and UV coordinates.
        """
        # Load the PLY file using trimesh
        mesh = trimesh.load(ply_file, process=False)
        
        vertices = mesh.vertices
        
        vertex_normals = mesh.vertex_normals if hasattr(mesh, 'vertex_normals') else None
        
        return vertices, vertex_normals

    def rotate_point_cloud(point_cloud, angle, axis='x'):
        """
        Rotate the point cloud around the specified axis by the given angle in degrees.
        Args:
            point_cloud (np.array): N x 3 array of xyz positions.
            angle (float): Rotation angle in degrees.
            axis (str): Axis to rotate about ('x', 'y', 'z').
        Returns:
            np.array: Rotated point cloud.
        """
        angle_rad = np.radians(angle)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        if axis == 'x':
            rot_matrix = np.array([[1, 0, 0],
                                   [0, cos_angle, -sin_angle],
                                   [0, sin_angle, cos_angle]])
        elif axis == 'y':
            rot_matrix = np.array([[cos_angle, 0, sin_angle],
                                   [0, 1, 0],
                                   [-sin_angle, 0, cos_angle]])
        elif axis == 'z':
            rot_matrix = np.array([[cos_angle, -sin_angle, 0],
                                   [sin_angle, cos_angle, 0],
                                   [0, 0, 1]])
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'")

        return np.dot(point_cloud, rot_matrix.T)

    target_vertices = None
    aligned_mesh = None
    aligned_colors = None

    aligned_vertices, vertex_normals = load_ply(ply_path)

    rotated_vertices = rotate_point_cloud(aligned_vertices, 90, axis='x')
    rotated_vertices = rotate_point_cloud(rotated_vertices, 180, axis='z')

    shs = np.random.random((rotated_vertices.shape[0], 3)) / 255.0
    colors = SH2RGB(shs)

    pcd = BasicPointCloud(points=rotated_vertices, colors=colors, normals=vertex_normals)

    scene_info = RSceneInfo(point_cloud=pcd, test_cameras=test_cam_infos, ply_path=ply_path)

    return scene_info

#borrow from https://github.com/ashawkey/stable-dreamfusion

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

def circle_poses(radius=torch.tensor([3.2]), theta=torch.tensor([60]), phi=torch.tensor([0]), angle_overhead=30, angle_front=60):

    theta = theta / 180 * np.pi
    phi = phi / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi

    centers = torch.stack([
        radius * torch.sin(theta) * torch.sin(phi),
        radius * torch.sin(theta) * torch.cos(phi),
        radius * torch.cos(theta),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = safe_normalize(centers)
    up_vector = torch.FloatTensor([0, 0, 1]).unsqueeze(0).repeat(len(centers), 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float).unsqueeze(0).repeat(len(centers), 1, 1)
    poses[:, :3, :3] = torch.stack((-right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses.numpy()

def gen_random_pos(size, param_range, gamma=1):
    lower, higher = param_range[0], param_range[1]
    
    mid = lower + (higher - lower) * 0.5
    radius = (higher - lower) * 0.5

    rand_ = torch.rand(size) # 0, 1
    sign = torch.where(torch.rand(size) > 0.5, torch.ones(size) * -1., torch.ones(size))
    rand_ = sign * (rand_ ** gamma)          

    return (rand_ * radius) + mid

def sym_rand_poses(size, opt, radius_range=[1, 1.5], theta_range=[0, 120], phi_range=[-180, 180],
                   angle_overhead=30, angle_front=60, uniform_sphere_rate=0.0, rand_cam_gamma=1):
    ''' Generate symmetric random poses from an orbit camera.
    Args:
        size: batch size of generated pose pairs.
        opt: options containing necessary parameters.
        radius_range: range of camera radii.
        theta_range: [min, max] in degrees, should be in [0, 180].
        phi_range: [min, max] in degrees, should be in [-180, 180].
    Return:
        poses_concat: [2*size, 4, 4], concatenated poses for both cameras in each pair.
        thetas_concat: [2*size], concatenated theta angles in degrees.
        phis_concat: [2*size], concatenated phi angles in degrees.
        radius_concat: [2*size], concatenated camera radii.
    '''

    # Convert angles from degrees to radians
    theta_range_rad = np.array(theta_range) * np.pi / 180
    phi_range_rad = np.array(phi_range) * np.pi / 180

    # Generate random radii
    radius = gen_random_pos(size, radius_range)

    # Generate random thetas and phis
    thetas = gen_random_pos(size, theta_range_rad, rand_cam_gamma)
    phis = gen_random_pos(size, phi_range_rad, rand_cam_gamma)

    # Ensure phis are in [-π, π]
    phis = ((phis + np.pi) % (2 * np.pi)) - np.pi

    # Mirror phis around default_azimuth (assumed to be 0 degrees)
    phis2 = -phis

    # Generate centers for both sets
    centers1 = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.sin(thetas) * torch.cos(phis),
        radius * torch.cos(thetas),
    ], dim=-1)  # [size, 3]

    centers2 = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis2),
        radius * torch.sin(thetas) * torch.cos(phis2),
        radius * torch.cos(thetas),
    ], dim=-1)  # [size, 3]

    # The target point is the origin
    targets = 0

    # Apply jitters if enabled
    if opt.jitter_pose:
        jit_center = opt.jitter_center
        jit_target = opt.jitter_target
        centers1 += torch.rand_like(centers1) * jit_center - jit_center / 2.0
        centers2 += torch.rand_like(centers2) * jit_center - jit_center / 2.0
        targets += torch.randn_like(centers1) * jit_target

    # Compute forward vectors
    forward_vector1 = safe_normalize(centers1 - targets)
    forward_vector2 = safe_normalize(centers2 - targets)

    # Use the same up vector for both cameras
    up_vector = torch.FloatTensor([0, 0, 1]).unsqueeze(0).repeat(size, 1)

    # Compute right and up vectors for both sets
    right_vector1 = safe_normalize(torch.cross(forward_vector1, up_vector, dim=-1))
    if opt.jitter_pose:
        up_noise = torch.randn_like(up_vector) * opt.jitter_up
    else:
        up_noise = 0
    up_vector1 = safe_normalize(torch.cross(right_vector1, forward_vector1, dim=-1) + up_noise)

    right_vector2 = safe_normalize(torch.cross(forward_vector2, up_vector, dim=-1))
    if opt.jitter_pose:
        up_noise = torch.randn_like(up_vector) * opt.jitter_up
    else:
        up_noise = 0
    up_vector2 = safe_normalize(torch.cross(right_vector2, forward_vector2, dim=-1) + up_noise)

    # Assemble pose matrices
    poses1 = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(size, 1, 1)
    poses1[:, :3, :3] = torch.stack((-right_vector1, up_vector1, forward_vector1), dim=-1)
    poses1[:, :3, 3] = centers1

    poses2 = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(size, 1, 1)
    poses2[:, :3, :3] = torch.stack((-right_vector2, up_vector2, forward_vector2), dim=-1)
    poses2[:, :3, 3] = centers2

    # Convert thetas and phis back to degrees
    thetas_deg = thetas * 180 / np.pi
    phis_deg = phis * 180 / np.pi
    phis2_deg = phis2 * 180 / np.pi

    # Concatenate the outputs
    poses_concat = torch.cat([poses1, poses2], dim=0)
    thetas_concat = torch.cat([thetas_deg, thetas_deg], dim=0)
    phis_concat = torch.cat([phis_deg, phis2_deg], dim=0)
    radius_concat = torch.cat([radius, radius], dim=0)

    return poses_concat.numpy(), thetas_concat.numpy(), phis_concat.numpy(), radius_concat.numpy()

def rand_poses(size, opt, radius_range=[1, 1.5], theta_range=[0, 120], phi_range=[0, 360], angle_overhead=30, angle_front=60, uniform_sphere_rate=0.5, rand_cam_gamma=1):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, pi]
        phi_range: [min, max], should be in [0, 2 * pi]
    Return:
        poses: [size, 4, 4]
    '''

    theta_range = np.array(theta_range) / 180 * np.pi
    phi_range = np.array(phi_range) / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi

    # radius = torch.rand(size) * (radius_range[1] - radius_range[0]) + radius_range[0]
    radius = gen_random_pos(size, radius_range)

    if random.random() < uniform_sphere_rate:
        unit_centers = F.normalize(
            torch.stack([
                torch.randn(size),
                torch.abs(torch.randn(size)),
                torch.randn(size),
            ], dim=-1), p=2, dim=1
        )
        thetas = torch.acos(unit_centers[:,1])
        phis = torch.atan2(unit_centers[:,0], unit_centers[:,2])
        phis[phis < 0] += 2 * np.pi
        centers = unit_centers * radius.unsqueeze(-1)
    else:
        # thetas = torch.rand(size) * (theta_range[1] - theta_range[0]) + theta_range[0]
        # phis = torch.rand(size) * (phi_range[1] - phi_range[0]) + phi_range[0]
        # phis[phis < 0] += 2 * np.pi

        # centers = torch.stack([
        #     radius * torch.sin(thetas) * torch.sin(phis),
        #     radius * torch.cos(thetas),
        #     radius * torch.sin(thetas) * torch.cos(phis),
        # ], dim=-1) # [B, 3]
        # thetas = torch.rand(size) * (theta_range[1] - theta_range[0]) + theta_range[0]
        # phis = torch.rand(size) * (phi_range[1] - phi_range[0]) + phi_range[0]
        thetas = gen_random_pos(size, theta_range, rand_cam_gamma)
        phis = gen_random_pos(size, phi_range, rand_cam_gamma)
        phis[phis < 0] += 2 * np.pi

        centers = torch.stack([
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.sin(thetas) * torch.cos(phis),
            radius * torch.cos(thetas),
        ], dim=-1) # [B, 3]

    targets = 0

    # jitters
    if opt.jitter_pose:
        jit_center = opt.jitter_center # 0.015  # was 0.2
        jit_target = opt.jitter_target
        centers += torch.rand_like(centers) * jit_center - jit_center/2.0
        targets += torch.randn_like(centers) * jit_target

    # lookat
    forward_vector = safe_normalize(centers - targets)
    up_vector = torch.FloatTensor([0, 0, 1]).unsqueeze(0).repeat(size, 1)
    #up_vector = torch.FloatTensor([0, 0, 1]).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))

    if opt.jitter_pose:
        up_noise = torch.randn_like(up_vector) * opt.jitter_up
    else:
        up_noise = 0

    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise) #forward_vector

    poses = torch.eye(4, dtype=torch.float).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((-right_vector, up_vector, forward_vector), dim=-1) #up_vector
    poses[:, :3, 3] = centers


    # back to degree
    thetas = thetas / np.pi * 180
    phis = phis / np.pi * 180

    return poses.numpy(), thetas.numpy(), phis.numpy(), radius.numpy()

def GenerateFrontalCameras(opt, size=8, render45 = False):
    # random focal
    fov = opt.default_fovy
    cam_infos = []
    #generate specific data structure
    for idx in range(size):
        thetas = torch.FloatTensor([opt.default_polar])
        phis = torch.FloatTensor([(0 / size) * 360])
        radius = torch.FloatTensor([opt.default_radius])
        # random pose on the fly
        poses = circle_poses(radius=radius, theta=thetas, phi=phis, angle_overhead=opt.angle_overhead, angle_front=opt.angle_front)
        matrix = np.linalg.inv(poses[0])
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]
        fovy = focal2fov(fov2focal(fov, opt.image_h), opt.image_w)
        FovY = fovy
        FovX = fov

        # delta polar/azimuth/radius to default view
        delta_polar = thetas - opt.default_polar
        delta_azimuth = phis - opt.default_azimuth
        delta_azimuth[delta_azimuth > 180] -= 360 # range in [-180, 180]
        delta_radius = radius - opt.default_radius
        cam_infos.append(RandCameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,width=opt.image_w, 
                        height = opt.image_h, delta_polar = delta_polar,delta_azimuth = delta_azimuth, delta_radius = delta_radius))  
    if render45:
        for idx in range(size):
            thetas = torch.FloatTensor([opt.default_polar*2//3])
            phis = torch.FloatTensor([(idx / size) * 360])
            radius = torch.FloatTensor([opt.default_radius])
            # random pose on the fly
            poses = circle_poses(radius=radius, theta=thetas, phi=phis, angle_overhead=opt.angle_overhead, angle_front=opt.angle_front)
            matrix = np.linalg.inv(poses[0])
            R = -np.transpose(matrix[:3,:3])
            R[:,0] = -R[:,0]
            T = -matrix[:3, 3]
            fovy = focal2fov(fov2focal(fov, opt.image_h), opt.image_w)
            FovY = fovy
            FovX = fov

            # delta polar/azimuth/radius to default view
            delta_polar = thetas - opt.default_polar
            delta_azimuth = phis - opt.default_azimuth
            delta_azimuth[delta_azimuth > 180] -= 360 # range in [-180, 180]
            delta_radius = radius - opt.default_radius
            cam_infos.append(RandCameraInfo(uid=idx+size, R=R, T=T, FovY=FovY, FovX=FovX,width=opt.image_w, 
                            height = opt.image_h, delta_polar = delta_polar,delta_azimuth = delta_azimuth, delta_radius = delta_radius))         
    return cam_infos

def GenerateCircleCameras(opt, size=8, render45 = False):
    # random focal
    fov = opt.default_fovy
    cam_infos = []
    #generate specific data structure
    for idx in range(size):
        thetas = torch.FloatTensor([opt.default_polar])
        phis = torch.FloatTensor([(idx / size) * 360])
        radius = torch.FloatTensor([opt.default_radius])
        # random pose on the fly
        poses = circle_poses(radius=radius, theta=thetas, phi=phis, angle_overhead=opt.angle_overhead, angle_front=opt.angle_front)
        matrix = np.linalg.inv(poses[0])
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]
        fovy = focal2fov(fov2focal(fov, opt.image_h), opt.image_w)
        FovY = fovy
        FovX = fov

        # delta polar/azimuth/radius to default view
        delta_polar = thetas - opt.default_polar
        delta_azimuth = phis - opt.default_azimuth
        delta_azimuth[delta_azimuth > 180] -= 360 # range in [-180, 180]
        delta_radius = radius - opt.default_radius
        cam_infos.append(RandCameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,width=opt.image_w, 
                        height = opt.image_h, delta_polar = delta_polar,delta_azimuth = delta_azimuth, delta_radius = delta_radius))  
    if render45:
        for idx in range(size):
            thetas = torch.FloatTensor([opt.default_polar*2//3])
            phis = torch.FloatTensor([(idx / size) * 360])
            radius = torch.FloatTensor([opt.default_radius])
            # random pose on the fly
            poses = circle_poses(radius=radius, theta=thetas, phi=phis, angle_overhead=opt.angle_overhead, angle_front=opt.angle_front)
            matrix = np.linalg.inv(poses[0])
            R = -np.transpose(matrix[:3,:3])
            R[:,0] = -R[:,0]
            T = -matrix[:3, 3]
            fovy = focal2fov(fov2focal(fov, opt.image_h), opt.image_w)
            FovY = fovy
            FovX = fov

            # delta polar/azimuth/radius to default view
            delta_polar = thetas - opt.default_polar
            delta_azimuth = phis - opt.default_azimuth
            delta_azimuth[delta_azimuth > 180] -= 360 # range in [-180, 180]
            delta_radius = radius - opt.default_radius
            cam_infos.append(RandCameraInfo(uid=idx+size, R=R, T=T, FovY=FovY, FovX=FovX,width=opt.image_w, 
                            height = opt.image_h, delta_polar = delta_polar,delta_azimuth = delta_azimuth, delta_radius = delta_radius))         
    return cam_infos

def GenerateRandomCameras(opt, size=2000, SSAA=True, fov=1):
    # random pose on the fly
    poses, thetas, phis, radius = rand_poses(size, opt, radius_range=opt.radius_range, theta_range=opt.theta_range, phi_range=opt.phi_range, 
                                             angle_overhead=opt.angle_overhead, angle_front=opt.angle_front, uniform_sphere_rate=opt.uniform_sphere_rate,
                                             rand_cam_gamma=opt.rand_cam_gamma)
                                             
    # delta polar/azimuth/radius to default view
    delta_polar = thetas - opt.default_polar
    delta_azimuth = phis - opt.default_azimuth
    delta_azimuth[delta_azimuth > 180] -= 360 # range in [-180, 180]
    delta_radius = radius - opt.default_radius
    # random focal
    fov = fov
    
    cam_infos = []

    if SSAA:
        ssaa = opt.SSAA
    else:
        ssaa = 1

    image_h = opt.image_h * ssaa
    image_w = opt.image_w * ssaa

    #generate specific data structure
    for idx in range(size):
        matrix = np.linalg.inv(poses[idx])
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]
        # matrix = poses[idx]
        # R = matrix[:3,:3]
        # T = matrix[:3, 3]
        fovy = focal2fov(fov2focal(fov, image_h), image_w)
        FovY = fovy
        FovX = fov

        cam_infos.append(RandCameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,width=image_w, 
                                        height=image_h, delta_polar = delta_polar[idx],
                                        delta_azimuth = delta_azimuth[idx], delta_radius = delta_radius[idx]))           
    return cam_infos

def GeneratePurnCameras(opt, size=300):
    # random pose on the fly
    poses, thetas, phis, radius = rand_poses(size, opt, radius_range=[opt.default_radius,opt.default_radius+0.1], theta_range=opt.theta_range, phi_range=opt.phi_range, angle_overhead=opt.angle_overhead, angle_front=opt.angle_front, uniform_sphere_rate=opt.uniform_sphere_rate)
    # delta polar/azimuth/radius to default view
    delta_polar = thetas - opt.default_polar
    delta_azimuth = phis - opt.default_azimuth
    delta_azimuth[delta_azimuth > 180] -= 360 # range in [-180, 180]
    delta_radius = radius - opt.default_radius
    # random focal
    #fov = random.random() * (opt.fovy_range[1] - opt.fovy_range[0]) + opt.fovy_range[0]
    fov = opt.default_fovy
    cam_infos = []
    #generate specific data structure
    for idx in range(size):
        matrix = np.linalg.inv(poses[idx])     
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]
        # matrix = poses[idx]
        # R = matrix[:3,:3]
        # T = matrix[:3, 3]
        fovy = focal2fov(fov2focal(fov, opt.image_h), opt.image_w)
        FovY = fovy
        FovX = fov

        cam_infos.append(RandCameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,width=opt.image_w, 
                        height = opt.image_h, delta_polar = delta_polar[idx],delta_azimuth = delta_azimuth[idx], delta_radius = delta_radius[idx]))           
    return cam_infos

sceneLoadTypeCallbacks = {
    # "Colmap": readColmapSceneInfo,
    # "Blender" : readNerfSyntheticInfo,
    "RandomCam" : readCircleCamInfo
}
