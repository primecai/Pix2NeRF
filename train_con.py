"""Train pi-GAN. Supports distributed training."""

import argparse
import copy
import math
import os
from datetime import datetime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_ema import ExponentialMovingAverage
from torchvision.utils import save_image
from tqdm import tqdm

import curriculums as curriculums
import datasets as datasets
import pytorch_ssim
from discriminators import discriminators_con as discriminators
from generators import generators_con as generators
from losses.losses import VGGPerceptualLoss
from PIL import Image
from siren import siren_con as siren

import torchvision.transforms as transforms

DDP_FIND_UNUSED_PARAM = True
WANDB_UPLOAD_IMAGES = True


def rmlock(log_dir):
    file_lock = os.path.join(log_dir, 'process_group_sync.lock')
    if os.path.isfile(file_lock):
        print('Removed lock')
        os.remove(file_lock)
    else:
        print('Lock not found')


def setup(rank, world_size, port, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    file_lock = f'file://{log_dir}/process_group_sync.lock'
    dist.init_process_group('gloo', init_method=file_lock, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()
    wandb.finish()


def load_images(images, curriculum, device):
    return_images = []
    head = 0
    for stage in curriculum['stages']:
        stage_images = images[head:head + stage['batch_size']]
        stage_images = F.interpolate(stage_images, size=stage['img_size'],  mode='bilinear', align_corners=True)
        return_images.append(stage_images)
        head += stage['batch_size']
    return return_images


def z_sampler(shape, device, dist):
    if dist == 'gaussian':
        z = torch.randn(shape, device=device)
    elif dist == 'uniform':
        z = torch.rand(shape, device=device) * 2 - 1
    return z


def torch_save_atomic(what, path):
    path_tmp = path + '.tmp'
    torch.save(what, path_tmp)
    os.rename(path_tmp, path)


def train(rank, world_size, opt):
    torch.cuda.empty_cache()

    torch.manual_seed(0)

    setup(rank, world_size, opt.port, opt.output_dir)
    device = torch.device(rank)

    curriculum = getattr(curriculums, opt.curriculum)
    metadata = curriculums.extract_metadata(curriculum, 0)

    fixed_z = z_sampler((25, metadata['latent_dim']), device='cpu', dist=metadata['z_dist'])

    fixed_img = None
    fixed_pitch = None
    fixed_yaw = None

    SIREN = getattr(siren, metadata['model'])

    scaler = torch.cuda.amp.GradScaler()

    ssim = pytorch_ssim.SSIM()
    vgg_perceptual = VGGPerceptualLoss().cuda(device=device)

    inv_normalize = transforms.Normalize(
    mean=[-0.5/0.5],
    std=[1/0.5]
    )

    if opt.load_dir != '':
        print('load previous model')
        checkpoint = torch.load(os.path.join(opt.load_dir, 'checkpoint_train.pth'), map_location=device)
        generator = checkpoint['generator.pth'].to(device)
        discriminator = checkpoint['discriminator.pth'].to(device)
        if opt.load_encoder == 1:
            encoder = checkpoint['encoder.pth'].to(device)
        else:
            print('start new encoder')
            if opt.encoder_type == 'CCS':
                print('use CCS encoder')
                encoder = discriminators.CCSEncoder(metadata['latent_dim']).to(device)
            elif opt.encoder_type == 'progressive':
                print('use progressive encoder')
                encoder = discriminators.ProgressiveEncoder(metadata['latent_dim']).to(device)
            else:
                print('encoder type wrongly defined')
        if opt.ema == 1:
            ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
            ema.load_state_dict(checkpoint['ema.pth'])
            if opt.load_encoder == 1:
                ema_encoder = ExponentialMovingAverage(encoder.parameters(), decay=0.999)
                ema_encoder.load_state_dict(checkpoint['ema_encoder.pth'])
            else:
                ema_encoder = ExponentialMovingAverage(encoder.parameters(), decay=0.999)
    else:
        print('create new model')
        generator = getattr(generators, metadata['generator'])(SIREN, metadata['latent_dim']).to(device)
        if opt.pretrained_dir != '':
            print('load pretrained model')
            sd_generator = torch.load(os.path.join(opt.pretrained_dir, 'generator.pth'), map_location=device).state_dict()
            generator.load_state_dict(sd_generator)
        discriminator = getattr(discriminators, metadata['discriminator'])(sn=(opt.sn > 0)).to(device)
        if opt.encoder_type == 'CCS':
            print('use CCS encoder')
            encoder = discriminators.CCSEncoder(metadata['latent_dim']).to(device)
        elif opt.encoder_type == 'progressive':
            print('use progressive encoder')
            encoder = discriminators.ProgressiveEncoder(metadata['latent_dim']).to(device)
        else:
            print('encoder type wrongly defined')
        if opt.ema == 1:
            ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
            ema_encoder = ExponentialMovingAverage(encoder.parameters(), decay=0.999)

    generator_ddp = DDP(
        generator,
        device_ids=[rank],
        find_unused_parameters=DDP_FIND_UNUSED_PARAM
    )
    discriminator_ddp = DDP(
        discriminator,
        device_ids=[rank],
        find_unused_parameters=DDP_FIND_UNUSED_PARAM,
        broadcast_buffers=False
    )
    encoder_ddp = DDP(
        encoder,
        device_ids=[rank],
        find_unused_parameters=DDP_FIND_UNUSED_PARAM
    )
    generator = generator_ddp.module
    discriminator = discriminator_ddp.module
    encoder = encoder_ddp.module

    if metadata.get('unique_lr', False):
        mapping_network_param_names = [name for name, _ in generator_ddp.module.siren.mapping_network.named_parameters()]
        mapping_network_parameters = [p for n, p in generator_ddp.named_parameters() if n in mapping_network_param_names]
        generator_parameters = [p for n, p in generator_ddp.named_parameters() if n not in mapping_network_param_names]
        optimizer_G = torch.optim.Adam([{'params': generator_parameters, 'name': 'generator'},
                                        {'params': mapping_network_parameters, 'name': 'mapping_network', 'lr':metadata['gen_lr']*5e-2}],
                                       lr=metadata['gen_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])
    else:
        optimizer_G = torch.optim.Adam(generator_ddp.parameters(), lr=metadata['gen_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])

    optimizer_D = torch.optim.Adam(discriminator_ddp.parameters(), lr=metadata['disc_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])
    optimizer_E = torch.optim.Adam(encoder_ddp.parameters(), lr=metadata['enc_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])

    if opt.load_dir != '':
        optimizer_G.load_state_dict(checkpoint['optimizer_G.pth'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D.pth'])
        if opt.load_encoder == 1:
            optimizer_E.load_state_dict(checkpoint['optimizer_E.pth'])
        if not metadata.get('disable_scaler', False):
            scaler.load_state_dict(checkpoint['scaler.pth'])

    generator_losses = []
    discriminator_losses = []
    encoder_losses = []

    if opt.set_step != None:
        generator.step = opt.set_step
        discriminator.step = opt.set_step
        encoder.step = opt.set_step

    if metadata.get('disable_scaler', False):
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    generator.set_device(device)

    # ----------
    #  Training
    # ----------
    with open(os.path.join(opt.output_dir, 'options.txt'), 'w') as f:
        f.write(str(opt))
        f.write('\n\n')
        f.write(str(generator))
        f.write('\n\n')
        f.write(str(discriminator))
        f.write('\n\n')
        f.write(str(curriculum))

    torch.manual_seed(rank)
    dataloader = None
    step_last_upsample = None
    total_progress_bar = tqdm(total = opt.n_epochs, desc = "Total progress", dynamic_ncols=True)
    total_progress_bar.update(discriminator.epoch)
    interior_step_bar = tqdm(dynamic_ncols=True)

    if rank == 0 and opt.wandb_name != '':
        os.environ['WANDB_START_METHOD'] = 'thread'  # hack: https://github.com/wandb/client/issues/1771#issuecomment-859670559
        wandb.init(
            project=opt.wandb_project,
            resume=True,
            entity=opt.wandb_entity if opt.wandb_entity != '' else None,
            name=opt.wandb_name,
            id=wandb.util.generate_id(),
            # id=opt.wandb_name,
            dir=opt.output_dir,
            save_code=False,
        )
        print(opt)

    for _ in range(opt.n_epochs):
        total_progress_bar.update(1)

        metadata = curriculums.extract_metadata(curriculum, discriminator.step)

        # Set learning rates
        for param_group in optimizer_G.param_groups:
            if param_group.get('name', None) == 'mapping_network':
                param_group['lr'] = metadata['gen_lr'] * 5e-2
            else:
                param_group['lr'] = metadata['gen_lr']
            param_group['betas'] = metadata['betas']
            param_group['weight_decay'] = metadata['weight_decay']
        for param_group in optimizer_D.param_groups:
            param_group['lr'] = metadata['disc_lr']
            param_group['betas'] = metadata['betas']
            param_group['weight_decay'] = metadata['weight_decay']
        for param_group in optimizer_E.param_groups:
            param_group['lr'] = metadata['enc_lr']
            param_group['betas'] = metadata['betas']
            param_group['weight_decay'] = metadata['weight_decay']

        if not dataloader or dataloader.batch_size != metadata['batch_size']:
            dataloader, CHANNELS = datasets.get_dataset_distributed(metadata['dataset'],
                                        world_size,
                                        rank,
                                        dataset_dir=opt.dataset_dir,
                                        split='train',
                                        **metadata)

            step_next_upsample = curriculums.next_upsample_step(curriculum, discriminator.step)
            step_last_upsample = curriculums.last_upsample_step(curriculum, discriminator.step)

            interior_step_bar.reset(total=(step_next_upsample - step_last_upsample))
            interior_step_bar.set_description(f"Progress to next stage")
            interior_step_bar.update((discriminator.step - step_last_upsample))

        if rank == 0:
            dataloader_mini, _ = datasets.get_dataset(metadata['dataset'],
                                            dataset_dir=opt.dataset_dir,
                                            batch_size=25,
                                            img_size=metadata['img_size'],
                                            split='test'
                                            )
            for i, (imgs_mini, _) in enumerate(dataloader_mini):
                with torch.no_grad():
                    fixed_img = imgs_mini.clone()
                    print (fixed_img.shape)
                break

        for i, (imgs, _) in enumerate(dataloader):
            metadata = curriculums.extract_metadata(curriculum, discriminator.step)

            if dataloader.batch_size != metadata['batch_size']: break

            if scaler.get_scale() < 1:
                scaler.update(1.)

            generator_ddp.train()
            discriminator_ddp.train()
            encoder_ddp.train()

            alpha = min(1, (discriminator.step - step_last_upsample) / (metadata['fade_steps']))

            real_imgs = imgs.to(device, non_blocking=True)

            metadata['nerf_noise'] = max(0, 1. - discriminator.step/5000.)

            # TRAIN DISCRIMINATOR
            with torch.cuda.amp.autocast():
                # Generate images for discriminator training
                with torch.no_grad():
                    z = z_sampler((real_imgs.shape[0], metadata['latent_dim']), device=device, dist=metadata['z_dist'])
                    split_batch_size = z.shape[0] // metadata['batch_split']
                    gen_imgs = []
                    gen_positions = []
                    for split in range(metadata['batch_split']):
                        subset_z = z[split * split_batch_size:(split+1) * split_batch_size]
                        g_imgs, g_pos = generator_ddp(subset_z, None, None, **metadata)

                        gen_imgs.append(g_imgs)
                        gen_positions.append(g_pos)

                    gen_imgs = torch.cat(gen_imgs, axis=0)
                    gen_positions = torch.cat(gen_positions, axis=0)

                real_imgs.requires_grad = True
                r_preds, r_pred_position = discriminator_ddp(real_imgs, alpha, **metadata)

            if metadata['r1_lambda'] > 0:
                # Gradient penalty
                grad_real = torch.autograd.grad(outputs=scaler.scale(r_preds.sum()), inputs=real_imgs, create_graph=True)
                inv_scale = 1./scaler.get_scale()
                grad_real = [p * inv_scale for p in grad_real][0]
            with torch.cuda.amp.autocast():
                if metadata['r1_lambda'] > 0:
                    grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                    grad_penalty = 0.5 * metadata['r1_lambda'] * grad_penalty
                else:
                    grad_penalty = 0

                g_preds, g_pred_position = discriminator_ddp(gen_imgs, alpha, **metadata)
                if opt.pos_lambda_gen > 0:
                    g_position_penalty = torch.nn.MSELoss()(g_pred_position, gen_positions) * opt.pos_lambda_gen
                    position_penalty = g_position_penalty
                    identity_penalty = position_penalty
                    if rank == 0 and opt.wandb_name != '':
                        wandb.log({
                            "d_g_position_penalty": g_position_penalty,
                            "d_position_penalty": position_penalty
                        })
                else:
                    g_position_penalty=0
                    identity_penalty=0

                e_latent, e_pos = encoder_ddp(gen_imgs, alpha)
                e_latent_loss = torch.nn.MSELoss()(e_latent, z) * opt.lambda_e_latent
                e_pos_loss = torch.nn.MSELoss()(e_pos, gen_positions) * opt.lambda_e_pos
                e_loss = e_latent_loss + e_pos_loss
                encoder_losses.append(e_loss.item())

                d_loss = torch.nn.functional.softplus(g_preds).mean() + torch.nn.functional.softplus(-r_preds).mean() + grad_penalty + identity_penalty
                discriminator_losses.append(d_loss.item())
                if rank == 0 and opt.wandb_name != '':
                    wandb.log({
                        "d_gan": d_loss,
                        "e_total": e_loss,
                        "e_latent": e_latent_loss,
                        "e_pos": e_pos_loss,
                    })

            optimizer_D.zero_grad()
            scaler.scale(d_loss).backward()
            scaler.unscale_(optimizer_D)
            torch.nn.utils.clip_grad_norm_(discriminator_ddp.parameters(), metadata['grad_clip'])
            scaler.step(optimizer_D)

            optimizer_E.zero_grad()
            scaler.scale(e_loss).backward()
            scaler.unscale_(optimizer_E)
            torch.nn.utils.clip_grad_norm_(encoder_ddp.parameters(), metadata['grad_clip'])
            scaler.step(optimizer_E)
            scaler.update()
            if opt.ema == 1:
                ema_encoder.update(encoder_ddp.parameters())
            optimizer_E.zero_grad()

            # TRAIN GENERATOR
            z = z_sampler((imgs.shape[0], metadata['latent_dim']), device=device, dist=metadata['z_dist'])

            split_batch_size = z.shape[0] // metadata['batch_split']
            if i % 2 == 1:
                for split in range(metadata['batch_split']):
                    with torch.cuda.amp.autocast():
                        subset_img = real_imgs[split * split_batch_size:(split+1) * split_batch_size]
                        img_z, pos_z = encoder_ddp(subset_img, alpha)
                        subset_img_pitches = pos_z[:, 0]
                        subset_img_yaws = pos_z[:, 1]

                        # conditional generative images
                        if opt.cond_lambda > 0:
                            gen_imgs, gen_positions = generator_ddp(img_z, None, None, **metadata)

                        if opt.cond_lambda > 0:
                            g_preds, g_pred_position = discriminator_ddp(gen_imgs, alpha, **metadata)

                        topk_percentage = max(0.99 ** (discriminator.step/metadata['topk_interval']), metadata['topk_v']) if 'topk_interval' in metadata and 'topk_v' in metadata else 1
                        if opt.cond_lambda > 0:
                            topk_num = math.ceil(topk_percentage * g_preds.shape[0])
                            g_preds = torch.topk(g_preds, topk_num, dim=0).values

                        # positional penalty
                        if metadata['z_lambda'] > 0 or opt.pos_lambda_gen > 0 and opt.cond_lambda > 0:
                            position_penalty = torch.nn.MSELoss()(g_pred_position, gen_positions) * opt.pos_lambda_gen
                            identity_penalty = position_penalty
                            if rank == 0 and opt.wandb_name != '':
                                wandb.log({
                                    "g_g_position_penalty": position_penalty.item()
                                })
                        else:
                            identity_penalty = 0

                        # GAN adversarial loss
                        if opt.cond_lambda > 0:
                            gan_loss = opt.cond_lambda * torch.nn.functional.softplus(-g_preds).mean()
                        else:
                            gan_loss = 0

                        # conditional reconstruction images
                        gen_imgs, _ = generator_ddp.module.forward(img_z, subset_img_pitches, subset_img_yaws, mode='recon', **metadata)
                        # Reconstruction loss
                        recon_l2_loss = torch.nn.MSELoss()(gen_imgs, subset_img)
                        recon_ssim_loss = opt.ssim_lambda * (1 - ssim(gen_imgs, subset_img))
                        recon_perceptual_loss = opt.vgg_lambda * vgg_perceptual(gen_imgs, subset_img)
                        recon_loss = opt.recon_lambda * (recon_l2_loss + recon_ssim_loss + recon_perceptual_loss)

                        g_loss = gan_loss + identity_penalty + recon_loss
                        generator_losses.append(g_loss.item())

                    scaler.scale(g_loss).backward()

                if metadata['warm_up'] == 0:
                    scaler.unscale_(optimizer_E)
                    torch.nn.utils.clip_grad_norm_(encoder_ddp.parameters(), metadata['grad_clip'])
                    scaler.step(optimizer_E)
                    scaler.update()
                    optimizer_E.zero_grad()
                    if opt.ema == 1:
                        ema_encoder.update(encoder_ddp.parameters())
                optimizer_E.zero_grad()

                scaler.unscale_(optimizer_G)
                torch.nn.utils.clip_grad_norm_(generator_ddp.parameters(), metadata.get('grad_clip', 0.3))
                scaler.step(optimizer_G)
                scaler.update()
                optimizer_G.zero_grad()
                if opt.ema == 1:
                    ema.update(generator_ddp.parameters())

                if rank == 0:
                    interior_step_bar.update(1)
                    if i%25 == 0:
                        tqdm.write(f"[Experiment: {opt.output_dir}] [GPU: {os.environ['CUDA_VISIBLE_DEVICES']}] [Epoch: {discriminator.epoch}/{opt.n_epochs}] [Step: {discriminator.step}] [Img Size: {metadata['img_size']}] [Batch Size: {metadata['batch_size']}] [Scale: {scaler.get_scale()}]")
                        if opt.wandb_name != '':
                            wandb.log({"g_total": g_loss,
                                    "g_con_gan": gan_loss,
                                    "recon_l2": recon_l2_loss,
                                    "recon_ssim": recon_ssim_loss,
                                    "recon_perceptual": recon_perceptual_loss,
                                    "recon_total": recon_loss,
                                    })
            elif i % 2 == 0:
                for split in range(metadata['batch_split']):
                    with torch.cuda.amp.autocast():
                        subset_z = z[split * split_batch_size:(split+1) * split_batch_size]
                        gen_imgs, gen_positions = generator_ddp(subset_z, None, None, **metadata)

                        g_preds, g_pred_position = discriminator_ddp(gen_imgs, alpha, **metadata)

                        topk_percentage = max(0.99 ** (discriminator.step/metadata['topk_interval']), metadata['topk_v']) if 'topk_interval' in metadata and 'topk_v' in metadata else 1
                        topk_num = math.ceil(topk_percentage * g_preds.shape[0])

                        g_preds = torch.topk(g_preds, topk_num, dim=0).values

                        if metadata['z_lambda'] > 0 or opt.pos_lambda_gen > 0:
                            position_penalty = torch.nn.MSELoss()(g_pred_position, gen_positions) * opt.pos_lambda_gen
                            identity_penalty = position_penalty
                            if rank == 0 and opt.wandb_name != '':
                                wandb.log({
                                    "g_g_position_penalty": position_penalty.item()
                                })
                        else:
                            identity_penalty = 0

                        # GAN adversarial loss
                        gan_loss = torch.nn.functional.softplus(-g_preds).mean()

                        g_loss = gan_loss + identity_penalty
                        generator_losses.append(g_loss.item())

                    scaler.scale(g_loss).backward()

                scaler.unscale_(optimizer_G)
                torch.nn.utils.clip_grad_norm_(generator_ddp.parameters(), metadata.get('grad_clip', 0.3))
                scaler.step(optimizer_G)
                scaler.update()
                optimizer_G.zero_grad()
                if opt.ema == 1:
                    ema.update(generator_ddp.parameters())

                if rank == 0:
                    interior_step_bar.update(1)
                    if i%25 == 1:
                        tqdm.write(f"[Experiment: {opt.output_dir}] [GPU: {os.environ['CUDA_VISIBLE_DEVICES']}] [Epoch: {discriminator.epoch}/{opt.n_epochs}] [Step: {discriminator.step}] [Img Size: {metadata['img_size']}] [Batch Size: {metadata['batch_size']}] [Scale: {scaler.get_scale()}]")
                        if opt.wandb_name != '':
                            wandb.log({"g_total": g_loss,
                                    "g_gan": gan_loss,
                                    })
            else:
                raise RuntimeError('alternative iteration collapse')

            if rank == 0:
                if discriminator.step % opt.sample_interval == 0:

                    dataloader_mini, _ = datasets.get_dataset(metadata['dataset'],
                                                    dataset_dir=opt.dataset_dir,
                                                    batch_size=25,
                                                    img_size=metadata['img_size'],
                                                    split='test'
                                                    )
                    for i, (imgs_mini, _) in enumerate(dataloader_mini):
                        with torch.no_grad():
                            fixed_img = imgs_mini.clone()
                            print (fixed_img.shape)
                        break

                    generator_ddp.eval()
                    encoder_ddp.eval()
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            # copied_metadata['img_size'] = 128
                            gen_imgs = generator_ddp.module.staged_forward(fixed_z.to(device), None, None, **copied_metadata)[0]
                            gen_imgs = inv_normalize(gen_imgs)
                    save_image(gen_imgs[:25], os.path.join(opt.output_dir, f"{discriminator.step}_fixed.png"), nrow=25, normalize=True)
                    save_image(inv_normalize(fixed_img), os.path.join(opt.output_dir, f"{discriminator.step}_input.png"), nrow=25, normalize=True)

                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            # copied_metadata['img_size'] = 128
                            img_z, pos_z = encoder_ddp(fixed_img.to(device), alpha)
                            gen_imgs = generator_ddp.module.staged_forward(img_z, pos_z[:, 0], pos_z[:, 1], mode='recon', **copied_metadata)[0]
                            gen_imgs = inv_normalize(gen_imgs)
                    save_image(gen_imgs[:25], os.path.join(opt.output_dir, f"{discriminator.step}_recon.png"), nrow=25, normalize=True)

                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            # copied_metadata['img_size'] = 128
                            img_z, pos_z = encoder_ddp(fixed_img.to(device), alpha)
                            gen_imgs = generator_ddp.module.staged_forward(img_z, None, None, **copied_metadata)[0]
                            gen_imgs = inv_normalize(gen_imgs)
                    save_image(gen_imgs[:25], os.path.join(opt.output_dir, f"{discriminator.step}_recon_fixed.png"), nrow=25, normalize=True)

                    if opt.wandb_name != '' and WANDB_UPLOAD_IMAGES:
                        wandb.log({"fixed": [wandb.Image(Image.open(os.path.join(opt.output_dir, f"{discriminator.step}_fixed.png")), caption=f"{discriminator.step}_fixed")],
                                    "recon": [wandb.Image(Image.open(os.path.join(opt.output_dir, f"{discriminator.step}_recon.png")), caption=f"{discriminator.step}_recon")],
                                    "recon_fixed": [wandb.Image(Image.open(os.path.join(opt.output_dir, f"{discriminator.step}_recon_fixed.png")), caption=f"{discriminator.step}_recon_fixed")],
                                    "input": [wandb.Image(Image.open(os.path.join(opt.output_dir, f"{discriminator.step}_input.png")), caption=f"{discriminator.step}_input")]})

                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['h_mean'] += 0.5
                            # copied_metadata['img_size'] = 128
                            gen_imgs = generator_ddp.module.staged_forward(fixed_z.to(device), None, None, **copied_metadata)[0]
                            gen_imgs = inv_normalize(gen_imgs)
                    save_image(gen_imgs[:25], os.path.join(opt.output_dir, f"{discriminator.step}_tilted.png"), nrow=25, normalize=True)

                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['h_mean'] += 0.5
                            # copied_metadata['img_size'] = 128
                            img_z, pos_z = encoder_ddp(fixed_img.to(device), alpha)
                            gen_imgs = generator_ddp.module.staged_forward(img_z, None, None, **copied_metadata)[0]
                            gen_imgs = inv_normalize(gen_imgs)
                    save_image(gen_imgs[:25], os.path.join(opt.output_dir, f"{discriminator.step}_recon_tilted.png"), nrow=25, normalize=True)

                    if opt.wandb_name != '' and WANDB_UPLOAD_IMAGES:
                        wandb.log({"tilted": [wandb.Image(Image.open(os.path.join(opt.output_dir, f"{discriminator.step}_tilted.png")), caption=f"{discriminator.step}_tilted")]})
                        wandb.log({"recon_tilted": [wandb.Image(Image.open(os.path.join(opt.output_dir, f"{discriminator.step}_recon_tilted.png")), caption=f"{discriminator.step}_recon_tilted")],})

                if discriminator.step % opt.sample_interval == 0:
                    if opt.ema == 1:
                        model_dict = {
                            'ema.pth': ema.state_dict(),
                            'ema_encoder.pth': ema_encoder.state_dict(),
                            'generator.pth': generator_ddp.module,
                            'discriminator.pth': discriminator_ddp.module,
                            'encoder.pth': encoder_ddp.module,
                            'optimizer_G.pth': optimizer_G.state_dict(),
                            'optimizer_D.pth': optimizer_D.state_dict(),
                            'optimizer_E.pth': optimizer_E.state_dict(),
                            'scaler.pth': scaler.state_dict(),
                            'generator.losses': generator_losses,
                            'discriminator.losses': discriminator_losses,
                            'encoder.losses': encoder_losses
                        }
                    else:
                        model_dict = {
                            'generator.pth': generator_ddp.module,
                            'discriminator.pth': discriminator_ddp.module,
                            'encoder.pth': encoder_ddp.module,
                            'optimizer_G.pth': optimizer_G.state_dict(),
                            'optimizer_D.pth': optimizer_D.state_dict(),
                            'optimizer_E.pth': optimizer_E.state_dict(),
                            'scaler.pth': scaler.state_dict(),
                            'generator.losses': generator_losses,
                            'discriminator.losses': discriminator_losses,
                            'encoder.losses': encoder_losses
                        }
                    torch_save_atomic(model_dict, os.path.join(opt.output_dir, 'checkpoint_train.pth'))

            if opt.eval_freq > 0 and (discriminator.step + 1) % opt.eval_freq == 0:

                torch.cuda.empty_cache()

            discriminator.step += 1
            generator.step += 1
        discriminator.epoch += 1
        generator.epoch += 1

    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=3000, help="number of epochs of training")
    parser.add_argument("--sample_interval", type=int, default=1000, help="interval between image sampling")
    parser.add_argument('--output_dir', type=str, default='debug')
    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--curriculum', type=str, required=True)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--port', type=str, default='12354')
    parser.add_argument('--set_step', type=int, default=None)
    parser.add_argument('--model_save_interval', type=int, default=200)
    parser.add_argument('--pretrained_dir', type=str, default='')
    parser.add_argument('--wandb_name', type=str, default='')
    parser.add_argument('--wandb_entity', type=str, default='')
    parser.add_argument('--wandb_project', type=str, default='')
    parser.add_argument('--recon_lambda', type=float, required=True)
    parser.add_argument('--ssim_lambda', type=float, required=True)
    parser.add_argument('--vgg_lambda', type=float, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--pos_lambda_gen', type=float, required=True)
    parser.add_argument('--sn', type=int, default=0, required=False)
    parser.add_argument('--lambda_e_latent', type=float, required=True)
    parser.add_argument('--lambda_e_pos', type=float, required=True)
    parser.add_argument('--encoder_type', type=str, required=True)
    parser.add_argument('--cond_lambda', type=float, required=True)
    parser.add_argument('--ema', type=int, default=1, required=False)
    parser.add_argument('--load_encoder', type=int, default=1, required=False)



    opt = parser.parse_args()
    # if os.path.exists(os.path.join(opt.output_dir, 'discriminator.losses')):
    if os.path.exists(os.path.join(opt.output_dir, 'checkpoint_train.pth')):
        opt.load_dir = opt.output_dir
    else:
        os.makedirs(opt.output_dir, exist_ok=True)
    print(opt)
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    assert num_gpus > 0, 'No GPUs found'
    rmlock(opt.output_dir)
    mp.spawn(train, args=(num_gpus, opt), nprocs=num_gpus, join=True)
    cleanup()