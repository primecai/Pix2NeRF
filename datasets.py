"""Datasets"""

import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision
import glob
import PIL
import random
import math
import pickle
import numpy as np
import imageio

import pandas as pd
import csv

from itertools import groupby


class celeba(Dataset):
    """CelebA Dataset"""

    def __init__(self, dataset_dir, img_size, split='train', **kwargs):
        super().__init__()

        self.dataset_dir = dataset_dir
        with open(os.path.join(dataset_dir, 'CelebA_pos.csv'), newline='') as f:
            reader = csv.reader(f)
            self.data = list(reader)
        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
        if split == 'all':
            self.data = self.data[8000:]
        elif split == 'test':
            self.data = self.data[:8000]
        self.transform = transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            transforms.Resize((img_size, img_size), interpolation=0)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        this_data = self.data[index]
        X = PIL.Image.open(os.path.join(self.dataset_dir, this_data[0]))
        X = self.transform(X)

        return X, 0


class carla(Dataset):
    """Carla Dataset"""

    def __init__(self, dataset_path, img_size, split, **kwargs):
        super().__init__()
        self.data = glob.glob(dataset_path)
        self.data.sort()
        random.shuffle(self.data)
        # CARLA is an extrmely small dataset with only 10k images.
        # We used these splits during the development stage,
        # but for final evaluation, we trained on all images and
        # test on 8000 images to match prior works for FID, KID and IS.
        # if split == 'train':
        #     self.data = glob.glob(dataset_path)[1000:]
        # elif split == 'test':
        #     self.data = glob.glob(dataset_path)[:1000]
        # else:
        #     raise RuntimeError("wrong split")
        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
        self.transform = transforms.Compose(
                    [transforms.Resize((img_size, img_size), interpolation=0), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)

        return X, 0


class srnchairs(Dataset):
    """srnchairs Dataset"""

    def __init__(self, dataset_path, img_size, split, **kwargs):
        super().__init__()
        self.data = []
        self.path = os.path.join(dataset_path)
        if split == 'train':
            with open(os.path.join(dataset_path, 'srn_chairs_train_filted.csv'), newline='') as f:
                reader = csv.reader(f)
                train_data = list(reader)
            print(train_data[0])
            train_data = [os.path.join('chairs_train', x[0]) for x in train_data]
            self.data = train_data
        elif split == 'val':
            with open(os.path.join(dataset_path, 'srn_chairs_val_filted.csv'), newline='') as f:
                reader = csv.reader(f)
                val_data = list(reader)
            val_data = [os.path.join('chairs_val', x[0]) for x in val_data]
            self.data = val_data
        elif split == 'test':
            with open(os.path.join(dataset_path, 'srn_chairs_test_filted.csv'), newline='') as f:
                reader = csv.reader(f)
                test_data = list(reader)
            test_data = [os.path.join('chairs_test', x[0]) for x in test_data]
            self.data = test_data
        elif split == 'train+val':
            with open(os.path.join(dataset_path, 'srn_chairs_train_filted.csv'), newline='') as f:
                reader = csv.reader(f)
                train_data = list(reader)
            train_data = [os.path.join('chairs_train', x[0]) for x in train_data]
            with open(os.path.join(dataset_path, 'srn_chairs_val_filted.csv'), newline='') as f:
                reader = csv.reader(f)
                val_data = list(reader)
            val_data = [os.path.join('chairs_val', x[0]) for x in val_data]
            self.data = train_data + val_data
        elif split == 'val+test':
            with open(os.path.join(dataset_path, 'srn_chairs_test_filted.csv'), newline='') as f:
                reader = csv.reader(f)
                test_data = list(reader)
            test_data = [os.path.join('chairs_test', x[0]) for x in test_data]
            with open(os.path.join(dataset_path, 'srn_chairs_val_filted.csv'), newline='') as f:
                reader = csv.reader(f)
                val_data = list(reader)
            val_data = [os.path.join('chairs_val', x[0]) for x in val_data]
            self.data = test_data + val_data
        else:
            raise RuntimeError("wrong split")
        random.shuffle(self.data)
        print(len(self.data))
        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
        self.transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Resize((img_size, img_size), interpolation=0), transforms.Normalize([0.5], [0.5])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = imageio.imread(os.path.join(self.path, self.data[index]))[..., :3]
        X = self.transform(X)

        return X, 0


def get_dataset(name, dataset_dir, img_size, split, subsample=None, batch_size=1, **kwargs):
    dataset = globals()[name](dataset_dir, img_size=img_size, split=split, **kwargs)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
        num_workers=4
    )
    return dataloader, 3


def get_dataset_distributed(name, world_size, rank, batch_size, dataset_dir, split, **kwargs):
    dataset = globals()[name](dataset_dir, split=split, **kwargs)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
    )

    return dataloader, 3
