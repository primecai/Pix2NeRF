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
import datasets as datasets

import torchvision.transforms as transforms

from PIL import Image
import PIL

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

inv_normalize = transforms.Normalize(
   mean=[-0.5/0.5],
   std=[1/0.5]
)

render_options = {
    'img_size': 128,
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'num_steps': 48,
    'h_stddev': 0,
    'v_stddev': 0,
    'v_mean': math.pi/2,
    'hierarchical_sample': True,
    'sample_dist': None,
    'clamp_mode': 'relu',
    'nerf_noise': 0,
    'last_back': True,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--max_batch_size', type=int, default=2400000)
    parser.add_argument('--lock_view_dependence', action='store_true')
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--ray_steps', type=int, default=12)
    parser.add_argument('--curriculum', type=str, default='celeba')
    parser.add_argument('--img_path', type=str, default='')
    opt = parser.parse_args()

    curriculum = getattr(curriculums, opt.curriculum)
    curriculum['num_steps'] = opt.ray_steps
    curriculum['img_size'] = opt.image_size
    curriculum['v_stddev'] = 0
    curriculum['h_stddev'] = 0
    curriculum['h_mean']: torch.tensor(math.pi/2).to(device)
    curriculum['v_mean']: torch.tensor(math.pi/2).to(device)
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

    car_angles_yaw = [-0.5, 0., 0.5]

    car_angles_yaw = [a + curriculum['h_mean'] for a in car_angles_yaw]

    car_angles_pitch = [-0.25, 0., 0.25]

    car_angles_pitch = [a + curriculum['v_mean'] for a in car_angles_pitch]
    list_objects = []
    tar_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            transforms.Resize((128, 128), interpolation=0)
        ])
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            transforms.Resize((64, 64), interpolation=0)
        ])
    img = PIL.Image.open(opt.img_path)
    tar_img = tar_transform(img)[:3]
    img = transform(img)[:3]
    with torch.no_grad():
        z, pos = encoder(img.unsqueeze(0).to(device), alpha=1.0)
        frequencies, phase_shifts = generator.siren.mapping_network(z)
    images = []

    w_frequencies = frequencies.mean(0, keepdim=True)
    w_phase_shifts = phase_shifts.mean(0, keepdim=True)

    w_frequency_offsets = torch.zeros_like(w_frequencies)
    w_phase_shift_offsets = torch.zeros_like(w_phase_shifts)
    w_frequency_offsets.requires_grad_()
    w_phase_shift_offsets.requires_grad_()

    optimizer = torch.optim.Adam([w_frequency_offsets, w_phase_shift_offsets], lr=1e-2, weight_decay = 1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.75)

    n_iterations = 200

    for i in tqdm(range(n_iterations)):
        noise_w_frequencies = 0.03 * torch.randn_like(w_frequencies) * (n_iterations - i)/n_iterations
        noise_w_phase_shifts = 0.03 * torch.randn_like(w_phase_shifts) * (n_iterations - i)/n_iterations
        frame, _ = generator.forward_with_frequencies(w_frequencies + noise_w_frequencies + w_frequency_offsets, w_phase_shifts + noise_w_phase_shifts + w_phase_shift_offsets, pos[:, 0], pos[:, 1], mode='recon', **curriculum)
        loss = torch.nn.MSELoss()(frame, img.unsqueeze(0).to(device))
        loss = loss.mean()
        print(loss)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    for i, y in enumerate(car_angles_yaw):
        tensor_img, _ = generator.staged_forward_with_frequencies(w_frequencies + w_frequency_offsets, w_phase_shifts + w_phase_shift_offsets, pos[:, 0], pos[:, 1], max_batch_size=opt.max_batch_size, mode='recon', h_mean=(y), lock_view_dependence=True, **render_options)
        print(torch.nn.MSELoss()(tensor_img.to(device), tar_img.unsqueeze(0).to(device)))
        this_output_dir = opt.output_dir
        os.makedirs(this_output_dir, exist_ok=True)
        tensor_img = inv_normalize(tensor_img)
        images.append(tensor_img)
    for i, y in enumerate(car_angles_yaw):
        tensor_img, _ = generator.staged_forward_with_frequencies(w_frequencies + w_frequency_offsets, w_phase_shifts + w_phase_shift_offsets, None, None, h_mean=(y), max_batch_size=opt.max_batch_size, lock_view_dependence=True, **render_options)
        print(torch.nn.MSELoss()(tensor_img.to(device), tar_img.unsqueeze(0).to(device)))
        this_output_dir = opt.output_dir
        os.makedirs(this_output_dir, exist_ok=True)
        tensor_img = inv_normalize(tensor_img)
        images.append(tensor_img)
    img_recon, tensor_img_recon = generate_img_recon(generator, z, pos, **curriculum)
    images = torch.cat(images)
    save_image(images, os.path.join(this_output_dir, f'grid.png'), nrow=1, normalize=False)
    tensor_img_recon = inv_normalize(tensor_img_recon)
    save_image(tensor_img_recon, os.path.join(this_output_dir, f'recon.png'), normalize=False)