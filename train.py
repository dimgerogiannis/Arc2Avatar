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

import random
import imageio
import os
from utils.loss_utils import tv_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, GenerateCamParams, GuidanceParams
import math
from torchvision.utils import save_image
import torchvision.transforms as T
from arc2face import project_face_embs
import shutil 
import torch.nn as nn
import torch
from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np
import torch.nn.functional as F
import logging
from utils.general_utils import inverse_sigmoid
import glob

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False



def adjust_text_embeddings(embeddings, azimuth, guidance_opt):
    text_z_list = []
    weights_list = []
    K = 0

    text_z_, weights_ = get_pos_neg_text_embeddings(embeddings, azimuth, guidance_opt)
    K = max(K, weights_.shape[0])
    text_z_list.append(text_z_)
    weights_list.append(weights_)

    text_embeddings = []
    for i in range(K):
        for text_z in text_z_list:
            text_embeddings.append(text_z[i] if i < len(text_z) else text_z[0])
    text_embeddings = torch.stack(text_embeddings, dim=0)

    weights = []
    for i in range(K):
        for weights_ in weights_list:
            weights.append(weights_[i] if i < len(weights_) else torch.zeros_like(weights_[0]))
    weights = torch.stack(weights, dim=0)
    return text_embeddings, weights

def get_pos_neg_text_embeddings(embeddings, azimuth_val, opt):
    if azimuth_val >= -90 and azimuth_val < 90:
        if azimuth_val >= 0:
            r = 1 - azimuth_val / 90
        else:
            r = 1 + azimuth_val / 90
        start_z = embeddings['front']
        end_z = embeddings['side']

        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['front'], embeddings['side']], dim=0)
        if r > 0.8:
            front_neg_w = 0.0
        else:
            front_neg_w = math.exp(-r * opt.front_decay_factor) * opt.negative_w
        if r < 0.2:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-(1-r) * opt.side_decay_factor) * opt.negative_w

        weights = torch.tensor([1.0, front_neg_w, side_neg_w])
    else:
        if azimuth_val >= 0:
            r = 1 - (azimuth_val - 90) / 90
        else:
            r = 1 + (azimuth_val + 90) / 90
        start_z = embeddings['side']
        end_z = embeddings['back']

        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['side'], embeddings['front']], dim=0)
        front_neg_w = opt.negative_w 
        if r > 0.8:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-r * opt.side_decay_factor) * opt.negative_w / 2

        weights = torch.tensor([1.0, side_neg_w, front_neg_w])
    return text_z, weights.to(text_z.device)

def prepare_embeddings(guidance_opt, guidance, id_emb, pipeline, factor=0.9):
    arcface_token_id = pipeline.tokenizer.encode("id", add_special_tokens=False)[0]

    def get_contextual_embedding(text):
        input_ids = pipeline.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=pipeline.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(pipeline.device)

        face_embs_padded = F.pad(id_emb, (0, pipeline.text_encoder.config.hidden_size - 512), "constant", 0)
        token_embs = pipeline.arc_text_encoder(input_ids=input_ids.repeat(len(id_emb), 1), return_token_embs=True)
        token_embs[input_ids == arcface_token_id] = face_embs_padded

        prompt_embeds = pipeline.arc_text_encoder(
            input_ids=input_ids,
            input_token_embs=token_embs
        )[0]

        return prompt_embeds

    def get_text_embedding(text):
        return guidance.get_text_embeds([text])

    embeddings = {}
    embeddings['default'] = get_contextual_embedding("photo of a id person")
    for d in ['front', 'side', 'back']:
        view_text = f"{d} view"
        if view_text == "back view":
            embeddings[d] = get_text_embedding(view_text)
            embeddings[d] = factor * embeddings['default'] + (1 - factor) * embeddings[d]
        elif view_text == "front view":
            embeddings[d] = get_text_embedding(view_text)
            embeddings[d] = factor * embeddings['default'] + (1 - factor) * embeddings[d]
        else:
            embeddings[d] = get_text_embedding(view_text)
            embeddings[d] = factor * embeddings['default'] + (1 - factor) * embeddings[d]

    embeddings['uncond'] = get_text_embedding(guidance_opt.negative)
    embeddings['inverse_text'] = get_text_embedding(guidance_opt.inverse_text)

    return embeddings

def guidance_setup(guidance_opt, id_emb, pipeline):
    if guidance_opt.guidance=="SD":
        from guidance.sd_utils import StableDiffusion
        guidance = StableDiffusion(guidance_opt.g_device, guidance_opt.fp16, guidance_opt.vram_O, 
                                   guidance_opt.t_range, guidance_opt.max_t_range, 
                                   num_train_timesteps=guidance_opt.num_train_timesteps, 
                                   ddim_inv=guidance_opt.ddim_inv,
                                   textual_inversion_path = guidance_opt.textual_inversion_path,
                                   LoRA_path = guidance_opt.LoRA_path,
                                   guidance_opt=guidance_opt)
    else:
        raise ValueError(f'{guidance_opt.guidance} not supported.')
    if guidance is not None:
        for p in guidance.parameters():
            p.requires_grad = False
    embeddings = prepare_embeddings(guidance_opt, guidance, id_emb, pipeline)
    return guidance, embeddings

class LaplacianReg(nn.Module):
    def __init__(self, vertex_num, faces):
        super(LaplacianReg, self).__init__()
        self.vertex_num = vertex_num
        self.neighbor_idxs, self.neighbor_weights = self.get_neighbor(vertex_num, faces)
    
    def get_neighbor(self, vertex_num, faces, neighbor_max_num=10):
        adj = {i: set() for i in range(vertex_num)}
        for face in faces.cpu().numpy():
            for idx in face:
                adj[idx].update(set(face) - {idx})
        
        neighbor_idxs = np.zeros((vertex_num, neighbor_max_num), dtype=int)
        neighbor_weights = np.zeros((vertex_num, neighbor_max_num), dtype=np.float32)
        for idx in range(vertex_num):
            neighbors = list(adj[idx])
            neighbor_num = min(len(neighbors), neighbor_max_num)
            if neighbor_num > 0:
                neighbor_idxs[idx, :neighbor_num] = neighbors[:neighbor_num]
                neighbor_weights[idx, :neighbor_num] = -1.0 / neighbor_num
            else:
                neighbor_idxs[idx, :] = idx
                neighbor_weights[idx, :] = 0.0
        
        device = faces.device
        neighbor_idxs = torch.from_numpy(neighbor_idxs).to(device)
        neighbor_weights = torch.from_numpy(neighbor_weights).to(device)
        return neighbor_idxs, neighbor_weights

    def compute_laplacian(self, x):
        neighbor_x = x[:, self.neighbor_idxs, :]
        weights = self.neighbor_weights.unsqueeze(0).unsqueeze(-1) 
        laplacian = x + (neighbor_x * weights).sum(dim=2)
        return laplacian

    def forward(self, out, target=None):
        lap_out = self.compute_laplacian(out)
        if target is None:
            loss = (lap_out) ** 2
        else:
            lap_target = self.compute_laplacian(target)
            loss = (lap_out - lap_target) ** 2
        return loss.mean()

def parse_image_path(full_image_path):
    directory = os.path.dirname(full_image_path)
    filename = os.path.basename(full_image_path)
    base_name, _ = os.path.splitext(filename)
    return base_name, directory

def compute_arcface_embedding(celeb_dir, extensions=['.jpg', '.jpeg', '.png']):
    #Use a single image in the directory of the subject
    #or use multiple images from which the average embedding is used
    subdir_name = os.path.basename(os.path.normpath(celeb_dir))
    
    celeb_name = os.path.basename(os.path.dirname(os.path.normpath(celeb_dir)))
    
    splat_dir = os.path.join(celeb_dir, 'splat')
    
    if not os.path.exists(splat_dir):
        os.makedirs(splat_dir)
        print(f"[INFO] created splat directory: {splat_dir}")
    else:
        print(f"[INFO] splat directory already exists: {splat_dir}")
    
    subject_id = f"{celeb_name}_{subdir_name}_splat"
    print(f"[INFO] processing directory: {celeb_dir} with subject_id: {subject_id}")
    
    try:
        print(f"[INFO] initializing ArcFace")
        app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(256, 256))
        print(f"[INFO] ArcFace initialized successfully")
    except Exception as e:
        print(f"[INFO] failed to initialize ArcFace")
        return None, None

    embeddings = []

    search_patterns = [os.path.join(celeb_dir, f"*{ext}") for ext in extensions]

    image_files = []
    for pattern in search_patterns:
        matched_files = glob.glob(pattern)
        image_files.extend(matched_files)

    if not image_files:
        print(f"[INFO] no images found in directory: {splat_dir}")
        return splat_dir, None

    print(f"[INFO] found {len(image_files)} images in '{celeb_dir}'")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for img_path in image_files:
        try:
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img)[:, :, ::-1] 
        except Exception as e:
            print(f"[INFO] error opening image '{img_path}': {e}")
            continue

        try:
            faces = app.get(img_np)
        except Exception as e:
            print(f"[INFO] error detecting faces in image '{img_path}': {e}")
            continue

        if not faces:
            print(f"[INFO] no faces detected in image '{img_path}', skipping")
            continue

        try:
            faces = sorted(faces, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]), reverse=True)
            largest_face = faces[0]
        except Exception as e:
            print(f"[INFO] error processing faces in image '{img_path}': {e}")
            continue

        try:
            id_emb = torch.tensor(largest_face['embedding'], dtype=torch.float16).unsqueeze(0).to(device)
            id_emb = id_emb / torch.norm(id_emb, dim=1, keepdim=True)
            embeddings.append(id_emb)
            print(f"[INFO] processed image '{img_path}' and extracted embedding")
        except Exception as e:
            print(f"[INFO] error processing embedding for image '{img_path}': {e}")
            continue

    if not embeddings:
        print(f"[INFO] no embeddings were extracted from images in '{celeb_dir}'")
        return splat_dir, None

    try:
        mean_id_emb = torch.cat(embeddings, dim=0).mean(dim=0, keepdim=True)
        mean_id_emb = mean_id_emb / torch.norm(mean_id_emb, dim=1, keepdim=True)
        print(f"[INFO] computed ArcFace embedding for subject: {subject_id}.")
    except Exception as e:
        print(f"[INFO] error computing ArcFace embedding: {e}")
        return splat_dir, None

    return splat_dir, mean_id_emb

def training(dataset, opt, pipe, gcams, guidance_opt, debug_from, save_video):

    full_image_path = opt.subject_id 

    splat_dir, id_emb = compute_arcface_embedding(full_image_path, extensions=['.jpg', '.jpeg', '.png'])

    dataset._model_path = splat_dir

    from guidance.sd_utils import StableDiffusion
    pipe = StableDiffusion(guidance_opt.g_device, guidance_opt.fp16, guidance_opt.vram_O, 
                                guidance_opt.t_range, guidance_opt.max_t_range, 
                                num_train_timesteps=guidance_opt.num_train_timesteps, 
                                ddim_inv=guidance_opt.ddim_inv,
                                textual_inversion_path = guidance_opt.textual_inversion_path,
                                LoRA_path = guidance_opt.LoRA_path,
                                guidance_opt=guidance_opt)

    embeddings = project_face_embs(pipe, id_emb)

    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gcams, gaussians)
    
    checkpoint = "init/mean_face_2.pth"
    (model_params, first_iter) = torch.load(checkpoint)
    gaussians.initialize(model_params, opt)

    gaussians.load_mask()
    
    initt_txyz = torch.from_numpy(gaussians._xyz[gaussians.mask].detach().cpu().numpy()).to(gaussians._xyz.device)

    initt_txyz.requires_grad = False

    with torch.no_grad():
            gaussians._opacity[~gaussians.longer_mask] = inverse_sigmoid(
            0.25 * torch.ones_like(gaussians._opacity[~gaussians.longer_mask])
            )

    gaussians.training_setup(opt)

    vertex_num = gaussians.masked_faces.max().item() + 1
    masked_lap = LaplacianReg(vertex_num, gaussians.masked_faces)

    bg_color = [1, 1, 1] if dataset._white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=dataset.data_device)

    save_folder = os.path.join(dataset._model_path,"train_process/")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder) 
        print('[INFO] train_process is in :', save_folder)

    use_control_net = False
    guidance, embeddings = guidance_setup(guidance_opt, id_emb, pipe)   

    ema_loss_for_log = 0.0

    if opt.save_process:
        save_folder_proc = os.path.join(scene.args._model_path,"process_videos/")
        if not os.path.exists(save_folder_proc):
            os.makedirs(save_folder_proc)
        process_view_points = scene.getCircleVideoCameras(batch_size=opt.pro_frames_num,render45=opt.pro_render_45).copy()    
        save_process_iter = opt.iterations // len(process_view_points)
        pro_img_frames = []
    
    def sample_camera(fov):
        cameras = scene.getRandTrainCameras(fov=fov)
        if not cameras:
            print("[INFO] no cameras generated")
            return None
        cam = random.choice(cameras)
        return cam

    process_view_points_copy = scene.getCircleVideoCameras(batch_size=opt.pro_frames_num,render45=opt.pro_render_45).copy()  
    frontal_viewpoint = process_view_points_copy.pop(0)
    
    embeddings = prepare_embeddings(guidance_opt, guidance, id_emb, pipe, factor=0.9) 

    first_iter = 1

    print("[INFO] optimization process has started")
    progress_bar = tqdm(range(first_iter, opt.iterations + 1), desc="Training progress")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    for iteration in progress_bar:     
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, guidance_opt.text)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        gaussians.update_feature_learning_rate(iteration)
        gaussians.update_rotation_learning_rate(iteration)
        gaussians.update_scaling_learning_rate(iteration)

        if not opt.use_progressive:                
            if iteration > 0 and iteration % opt.scale_up_cameras_iter == 0:
                scene.pose_args.fovy_range[0] = max(scene.pose_args.max_fovy_range[0], scene.pose_args.fovy_range[0] * opt.fovy_scale_up_factor[0])
                scene.pose_args.fovy_range[1] = min(scene.pose_args.max_fovy_range[1], scene.pose_args.fovy_range[1] * opt.fovy_scale_up_factor[1])


                scene.pose_args.radius_range[1] = max(scene.pose_args.max_radius_range[1], scene.pose_args.radius_range[1] * opt.scale_up_factor)
                scene.pose_args.radius_range[0] = max(scene.pose_args.max_radius_range[0], scene.pose_args.radius_range[0] * opt.scale_up_factor)

                scene.pose_args.theta_range[1] = min(scene.pose_args.max_theta_range[1], scene.pose_args.theta_range[1] * opt.phi_scale_up_factor)
                scene.pose_args.theta_range[0] = max(scene.pose_args.max_theta_range[0], scene.pose_args.theta_range[0] * 1/opt.phi_scale_up_factor)

                scene.pose_args.phi_range[0] = max(scene.pose_args.max_phi_range[0] , scene.pose_args.phi_range[0] * opt.phi_scale_up_factor)
                scene.pose_args.phi_range[1] = min(scene.pose_args.max_phi_range[1], scene.pose_args.phi_range[1] * opt.phi_scale_up_factor)
                
                print('[INFO] scale up theta_range to:', scene.pose_args.theta_range)
                print('[INFO] scale up radius_range to:', scene.pose_args.radius_range)
                print('[INFO] scale up phi_range to:', scene.pose_args.phi_range)
                print('[INFO] scale up fovy_range to:', scene.pose_args.fovy_range)

        logging.basicConfig(level=logging.INFO)

        viewpoint_cams = []
        images = []
        text_z_ = []
        weights_ = []
        depths = []
        alphas = []
        scales = []

        text_z_inverse = torch.cat([embeddings['uncond'], embeddings['inverse_text']], dim=0)

        total_samples = 0

        if 'viewpoint_cams' not in locals():
            viewpoint_cams = []

        while total_samples < guidance_opt.C_batch_size:
            fov = random.random() * (scene.pose_args.fovy_range[1] - scene.pose_args.fovy_range[0]) + scene.pose_args.fovy_range[0]

            viewpoint_cam = sample_camera(fov=fov)

            if viewpoint_cam is None:
                logging.error("[INFO] failed to sample a valid camera")
                continue 

            viewpoint_cams.append(viewpoint_cam)

            total_samples += 1

            text_z = [embeddings['uncond']]

            text_z_comp, weights = adjust_text_embeddings(embeddings, viewpoint_cam.delta_azimuth, guidance_opt)
            text_z.append(text_z_comp)
            weights_.append(weights)

            text_z = torch.cat(text_z, dim=0)
            text_z_.append(text_z)

            if (iteration - 1) == debug_from:
                pipe.debug = True
            render_pkg = render(
                viewpoint_cam, gaussians, pipe, background, 
                sh_deg_aug_ratio=dataset.sh_deg_aug_ratio, 
                bg_aug_ratio=dataset.bg_aug_ratio, 
                shs_aug_ratio=dataset.shs_aug_ratio, 
                scale_aug_ratio=dataset.scale_aug_ratio,
            )
            image = render_pkg["render"]
            viewspace_point_tensor = render_pkg["viewspace_points"]
            visibility_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]
            depth = render_pkg["depth"]
            alpha = render_pkg["alpha"]

            scales.append(render_pkg["scales"])
            images.append(image)
            depths.append(depth)
            alphas.append(alpha)
            viewpoint_cams.append(viewpoint_cam)  #

            if (iteration - 1) == debug_from:
                pipe.debug = False

        images = torch.stack(images, dim=0)
        depths = torch.stack(depths, dim=0)
        alphas = torch.stack(alphas, dim=0)

        warm_up_rate = 1. - min(iteration/opt.warmup_iter,1.)
        _aslatent = False
        if iteration < opt.geo_iter or random.random()< opt.as_latent_ratio:
            _aslatent=True
        if iteration > opt.use_control_net_iter and (random.random() < guidance_opt.controlnet_ratio):
                use_control_net = True
        ism_loss = guidance.train_step_perpneg(torch.stack(text_z_, dim=1), images, 
                                            pred_depth=depths, pred_alpha=alphas,
                                            grad_scale=guidance_opt.lambda_guidance,
                                            use_control_net = use_control_net ,save_folder = save_folder,  iteration = iteration, warm_up_rate=warm_up_rate, 
                                            weights = torch.stack(weights_, dim=1), resolution=(gcams.image_h, gcams.image_w),
                                            guidance_opt=guidance_opt,as_latent=_aslatent, embedding_inverse = text_z_inverse)
        
        scales = torch.stack(scales, dim=0)
        loss_scale = torch.mean(scales,dim=-1).mean()
        loss_tv = tv_loss(images) + tv_loss(depths) 

        ism_loss = ism_loss + opt.lambda_tv * loss_tv + loss_scale

        pos_weight = 100000000
        lap_weight = 100000000
  

        laploss = masked_lap(gaussians._xyz[gaussians.mask].unsqueeze(0), initt_txyz.unsqueeze(0))

        xyzloss = torch.nanmean((gaussians._xyz[gaussians.mask] - initt_txyz)**2)
    

        if iteration % 100 == 0:
            print(f"[TRAIN] ISM Loss: {loss.item()} -- Positional Loss: {(xyzloss).item() * pos_weight} -- Laplacian Loss: {(laploss).item() * lap_weight}")
        loss = ism_loss + xyzloss * pos_weight + laploss * lap_weight

        loss.backward()

        iter_end.record()

        if iteration % 100 == 0:
            print("[TRAIN] current number of gaussians:", gaussians._xyz.shape)

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if opt.save_process:
                if iteration % save_process_iter == 0 and len(process_view_points) > 0:
                    viewpoint_cam_p = process_view_points.pop(0)
                    render_p = render(viewpoint_cam_p, gaussians, pipe, background, test=True)
                    img_p = torch.clamp(render_p["render"], 0.0, 1.0) 
                    img_p = img_p.detach().cpu().permute(1,2,0).numpy()
                    img_p = (img_p * 255).round().astype('uint8')
                    pro_img_frames.append(img_p)  

            viewpoint_cam_p = frontal_viewpoint
            render_p = render(viewpoint_cam_p, gaussians, pipe, background, test=True)
            img_p = torch.clamp(render_p["render"], 0.0, 1.0) 
            img_p = img_p.detach().cpu().permute(1,2,0).numpy()
            img_p = (img_p * 255).round().astype('uint8')

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()


            if (iteration % 1000 == 0 and iteration > 0):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), dataset._model_path + "/chkpnt" + str(iteration) + ".pth")
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                if save_video:
                    video_inference(iteration, scene, render, (pipe, background))
                training_report(tb_writer, iteration, iter_start.elapsed_time(iter_end), [iteration], scene, render, (pipe, background))
                  
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration >= opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.2, scene.cameras_extent, size_threshold)
            
                if iteration >= opt.densify_from_iter and iteration >= 400 and (iteration % opt.opacity_reset_interval == 0):
                    gaussians.reset_opacity()

                if iteration % 100 == 0:
                    gaussians.prune_below_masked_points()
                    
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

    if opt.save_process:
        imageio.mimwrite(os.path.join(save_folder_proc, "video_rgb.mp4"), pro_img_frames, fps=40, quality=10)



import shutil 
def prepare_output_and_logger(args):    
    if not args._model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args._model_path = os.path.join("./celeb_output/", args.workspace)
        
    print("[INFO] output folder: {}".format(args._model_path))
    if os.path.exists(args._model_path):
        shutil.rmtree(args._model_path)

    os.makedirs(args._model_path, exist_ok=True)

    if args.opt_path is not None:
        os.system(' '.join(['cp', args.opt_path, os.path.join(args._model_path, 'config.yaml')]))

    with open(os.path.join(args._model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args._model_path)
    else:
        print("[INFO] tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    if iteration in testing_iterations:
        save_folder = os.path.join(scene.args._model_path,"test_six_views/{}_iteration".format(iteration))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder) 
            print('test views is in :', save_folder)
        torch.cuda.empty_cache()
        config = ({'name': 'test', 'cameras' : scene.getTestCameras()})
        if config['cameras'] and len(config['cameras']) > 0:
            for idx, viewpoint in enumerate(config['cameras']):
                render_out = renderFunc(viewpoint, scene.gaussians, *renderArgs, test=True)
                rgb, depth = render_out["render"],render_out["depth"]
                if depth is not None:
                    depth_norm = depth/depth.max()
                    save_image(depth_norm,os.path.join(save_folder,"render_depth_{}.png".format(viewpoint.uid)))

                image = torch.clamp(rgb, 0.0, 1.0)
                save_image(image,os.path.join(save_folder,"render_view_{}.png".format(viewpoint.uid)))
                if tb_writer:
                    tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.uid), image[None], global_step=iteration)     
            print("\n[ITER {}] Eval Done!".format(iteration))
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def video_inference(iteration, scene : Scene, renderFunc, renderArgs):
    sharp = T.RandomAdjustSharpness(3, p=1.0)

    save_folder = os.path.join(scene.args._model_path,"videos/{}_iteration".format(iteration))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)  # makedirs 
        print('videos is in :', save_folder)
    torch.cuda.empty_cache()
    config = ({'name': 'test', 'cameras' : scene.getCircleVideoCameras()})
    if config['cameras'] and len(config['cameras']) > 0:
        img_frames = []
        depth_frames = []
        print("[INFO] generating Video using", len(config['cameras']), "different view points")
        for idx, viewpoint in enumerate(config['cameras']):
            render_out = renderFunc(viewpoint, scene.gaussians, *renderArgs, test=True)
            rgb,depth = render_out["render"],render_out["depth"]
            if depth is not None:
                depth_norm = depth/depth.max()
                depths = torch.clamp(depth_norm, 0.0, 1.0) 
                depths = depths.detach().cpu().permute(1,2,0).numpy()
                depths = (depths * 255).round().astype('uint8')          
                depth_frames.append(depths)    
  
            image = torch.clamp(rgb, 0.0, 1.0) 
            image = image.detach().cpu().permute(1,2,0).numpy()
            image = (image * 255).round().astype('uint8')
            img_frames.append(image)    
        imageio.mimwrite(os.path.join(save_folder, "video_rgb_{}.mp4".format(iteration)), img_frames, fps=40, quality=10)
        if len(depth_frames) > 0:
            imageio.mimwrite(os.path.join(save_folder, "video_depth_{}.mp4".format(iteration)), depth_frames, fps=40, quality=10)
        print("\n[ITER {}] Video Save Done!".format(iteration))
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import yaml

    def update_yaml(yaml_file_path, subject_id, batch_size):
        with open(yaml_file_path, 'r') as file:
            lines = file.readlines()

        workspace_name = subject_id.rsplit('.', 1)[0]

        with open(yaml_file_path, 'w') as file:
            for line in lines:
                if line.strip().startswith('subject_id:'):
                    file.write(f'  subject_id: "{subject_id}"\n')
                elif line.strip().startswith('workspace:'):
                    file.write(f'  workspace: "{workspace_name}"\n')
                elif line.strip().startswith('C_batch_size:'):
                    file.write(f'  C_batch_size: {batch_size}\n')
                else:
                    file.write(line)

    parser = ArgumentParser(description="Training script parameters")

    parser.add_argument('--opt', type=str, default=None)
    parser.add_argument('--subject', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_video", type=bool, default=False)
    parser.add_argument("--quiet", action="store_true")

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    gcp = GenerateCamParams(parser)
    gp = GuidanceParams(parser)

    args = parser.parse_args(sys.argv[1:])

    update_yaml(args.opt, args.subject, args.batch_size)

    if args.opt is not None:
        with open(args.opt) as f:
            opts = yaml.load(f, Loader=yaml.FullLoader)
        lp.load_yaml(opts.get('ModelParams', None))
        op.load_yaml(opts.get('OptimizationParams', None))
        pp.load_yaml(opts.get('PipelineParams', None))
        gcp.load_yaml(opts.get('GenerateCamParams', None))
        gp.load_yaml(opts.get('GuidanceParams', None))
        
        lp.opt_path = args.opt
        args.port = opts['port']
        args.save_video = opts.get('save_video', True)
        args.seed = opts.get('seed', 0)
        args.device = opts.get('device', 'cuda')

        gp.g_device = args.device
        lp.data_device = args.device
        gcp.device = args.device

    safe_state(args.quiet, seed=args.seed)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp, op, pp, gcp, gp, args.debug_from, args.save_video)

    print("\n[TRAINING] training complete")