import argparse
import math
import glob
import numpy as np
import sys
import os

import torch
from torchvision.utils import save_image
from tqdm import tqdm

import curriculums as curriculums

import pandas as pd

from torch_ema import ExponentialMovingAverage

import torchvision.transforms as transforms
import skvideo.io
import copy
import PIL
from PIL import Image
from moviepy.editor import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def show(tensor_img):
    if len(tensor_img.shape) > 3:
        tensor_img = tensor_img.squeeze(0)
    tensor_img = tensor_img.permute(1, 2, 0).squeeze().cpu().numpy()
    plt.imshow(tensor_img)
    plt.show()

def generate_img(gen, z, **kwargs):

    with torch.no_grad():
        img = generator.staged_forward(z, None, None, max_batch_size=opt.max_batch_size, **kwargs)[0].to(device)
        tensor_img = img.detach()

        img_min = img.min()
        img_max = img.max()
        img = (img - img_min)/(img_max-img_min)
        img = img.permute(0, 2, 3, 1).squeeze().cpu().numpy()
    return img, tensor_img

def generate_img_recon(gen, z, pos_z, **kwargs):

    with torch.no_grad():
        img = generator.staged_forward(z, pos_z[:, 0], pos_z[:, 1], mode='recon', **kwargs)[0].to(device)
        tensor_img = img.detach()

        img_min = img.min()
        img_max = img.max()
        img = (img - img_min)/(img_max-img_min)
        img = img.permute(0, 2, 3, 1).squeeze().cpu().numpy()
    return img, tensor_img

def tensor_to_PIL(img):
    img = img.squeeze() * 0.5 + 0.5
    return Image.fromarray(img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())

inv_normalize = transforms.Normalize(
   mean=[-0.5/0.5],
   std=[1/0.5]
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--max_batch_size', type=int, default=600000)
    parser.add_argument('--lock_view_dependence', action='store_true')
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--ray_steps', type=int, default=96)
    parser.add_argument('--curriculum', type=str, default='celeba')
    parser.add_argument('--trajectory', type=str, default='front')
    parser.add_argument('--num_frames', type=int, default=64)
    parser.add_argument('--img_path', type=str, default='')
    opt = parser.parse_args()

    curriculum = getattr(curriculums, opt.curriculum)
    curriculum['num_steps'] = opt.ray_steps
    curriculum['img_size'] = opt.image_size
    curriculum['v_stddev'] = 0
    curriculum['h_stddev'] = 0
    curriculum['lock_view_dependence'] = opt.lock_view_dependence
    curriculum['last_back'] = True
    curriculum['nerf_noise'] = 0
    curriculum = {key: value for key, value in curriculum.items() if type(key) is str}

    os.makedirs(opt.output_dir, exist_ok=True)

    checkpoint = torch.load(opt.path, map_location=torch.device(device))
    generator = checkpoint['generator.pth'].to(device)
    encoder = checkpoint['encoder.pth'].to(device)
    ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
    ema_encoder = ExponentialMovingAverage(encoder.parameters(), decay=0.999)
    ema.load_state_dict(checkpoint['ema.pth'])
    ema_encoder.load_state_dict(checkpoint['ema_encoder.pth'])
    ema.copy_to(generator.parameters())
    ema_encoder.copy_to(encoder.parameters())
    generator.set_device(device)
    generator.eval()
    encoder.eval()

    if opt.trajectory == 'front':
        trajectory = []
        for t in np.linspace(0, 1, opt.num_frames):
            pitch = 0.2 * np.cos(t * 2 * math.pi) + math.pi/2
            yaw = 0.4 * np.sin(t * 2 * math.pi) + math.pi/2
            fov = 12
            # fov = 12 + 5 + np.sin(t * 2 * math.pi) * 5

            trajectory.append((pitch, yaw, fov))
    elif opt.trajectory == 'orbit':
        trajectory = []
        for t in np.linspace(0, 1, opt.num_frames):
            pitch = math.pi/4
            yaw = t * 2 * math.pi
            fov = curriculum['fov']

            trajectory.append((pitch, yaw, fov))

    if opt.curriculum == 'celeba':
        this_transform = transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            transforms.Resize((64, 64), interpolation=0)
        ])
    if opt.curriculum == 'carla':
        this_transform = transforms.Compose([
            transforms.Resize((64, 64), interpolation=0),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    if opt.curriculum == 'srnchairs':
        this_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64, 64), interpolation=0),
            transforms.Normalize([0.5], [0.5])
        ])

    this_img = PIL.Image.open(opt.img_path)
    this_img = this_transform(this_img)[:3]
    frames = []
    z, pos = encoder(this_img.unsqueeze(0).to(device), alpha=1.0)
    output_name = 'video.mp4'
    writer = skvideo.io.FFmpegWriter(os.path.join(opt.output_dir, output_name), outputdict={'-pix_fmt': 'yuv420p', '-crf': '21'})
    for pitch, yaw, fov in tqdm(trajectory):
        curriculum_copy = copy.deepcopy(curriculum)
        curriculum_copy['h_mean'] = yaw
        curriculum_copy['v_mean'] = pitch
        curriculum_copy['fov'] = fov
        curriculum_copy['h_stddev'] = 0
        curriculum_copy['v_stddev'] = 0

        frame = generator.staged_forward(z, None, None, max_batch_size=opt.max_batch_size, **curriculum_copy)[0].to(device)
        frames.append(tensor_to_PIL(frame))
    for frame in frames:
        writer.writeFrame(np.array(frame))

    writer.close()
    real_imgs = inv_normalize(this_img)
    save_image(real_imgs, os.path.join(opt.output_dir, f'reconstruction.png'), normalize=False)

    clip = (VideoFileClip(os.path.join(opt.output_dir, output_name)))
    gif_name = 'gif.gif'
    clip.write_gif(os.path.join(opt.output_dir, gif_name))