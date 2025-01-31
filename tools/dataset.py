#!/usr/bin/env python
# -*- coding: utf-8 -*-


# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import json
from PIL import Image 
import numpy as np
from pathlib import Path
import torch 
from torch.utils.data import Dataset, DataLoader
# from functools import partial
import pytorch_lightning as pl
from ldm.util import instantiate_from_config
import pandas as pd
import os, random
from typing import Any, Callable, List, Optional, Tuple
import torch
from torchvision import transforms


def worker_init_fn(_):
    """
    This function is from RoSteALS
    """
    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id
    return np.random.seed(np.random.get_state()[1][0] + worker_id)


class DataModule(pl.LightningDataModule):
    def __init__(self, train, validation, batch_size= 8, num_workers=None , use_worker_init_fn=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if self.use_worker_init_fn:
            self.init_fn = worker_init_fn
        else:
            self.init_fn = None
        
        self.train_config = train
        self.validation_config = validation

    def setup(self, stage= None):
        self.dataset_train = instantiate_from_config(self.train_config)
        self.dataset_validation = instantiate_from_config(self.validation_config)
       

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True,
                          worker_init_fn=self.init_fn, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_validation, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True,
                          worker_init_fn=self.init_fn, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.dataset_validation, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True,
                          worker_init_fn=self.init_fn, drop_last=True)

    def predict_dataloader(self):
        return DataLoader(self.dataset_validation, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True,
                          worker_init_fn=self.init_fn, drop_last=True)

class dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_list, message_len=48, resize=256, transform=None, **kwargs):
        super().__init__()
        if resize != 'all':
            if transform is None:
                self.transform = [transforms.RandomResizedCrop((resize, resize), scale=(0.8, 1.0), ratio=(0.75, 1.33))]
            else:
                self.transform = transform 
        else:
            self.transform = [transforms.RandomResizedCrop(256, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
                               transforms.RandomResizedCrop(512, scale=(0.8, 1.0), ratio=(0.75, 1.33))]
        self.data_dir = data_dir
        self.data_list = pd.read_csv(data_list)['path'].tolist()
        self.N = len(self.data_list)
        self.kwargs = kwargs
        self.message_len = message_len
    
    def __getitem__(self, index):
        path = self.data_list[index]
        img = Image.open(os.path.join(self.data_dir, path))
        img = img.convert('RGB')
        transform = random.choice(self.transform)
        img = transform(img)
        img = np.array(img, dtype=np.float32)/127.5-1.  # [-1, 1]
        msgs = torch.rand(self.message_len) > 0.5 # b k
        msgs = 2 * msgs.type(torch.float32) - 1.
        return {'image': img, 'message': msgs}

    def __len__(self) -> int:
        return self.N 

# class WrappedDataset(Dataset):
#     """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

#     def __init__(self, dataset):
#         self.data = dataset

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]

# class DataModuleFromConfig(pl.LightningDataModule):
#     """
#     This code is from RoSteALS paper
#     """
#     def __init__(self, batch_size, train=None, validation=None, test=None, predict=None, wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
#                  shuffle_val_dataloader=False):
#         super().__init__()
#         self.batch_size = batch_size
#         self.dataset_configs = dict()
#         self.num_workers = num_workers if num_workers is not None else batch_size * 2
#         self.use_worker_init_fn = use_worker_init_fn
#         if train is not None:
#             self.dataset_configs["train"] = train
#             self.train_dataloader = self._train_dataloader
#         if validation is not None:
#             self.dataset_configs["validation"] = validation
#             self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
#         if test is not None:
#             self.dataset_configs["test"] = test
#             self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
#         if predict is not None:
#             self.dataset_configs["predict"] = predict
#             self.predict_dataloader = self._predict_dataloader
#         self.wrap = wrap

#     def prepare_data(self):
#         for data_cfg in self.dataset_configs.values():
#             print(data_cfg)
#             instantiate_from_config(data_cfg)

#     def setup(self, stage=None):
#         self.datasets = dict(
#             (k, instantiate_from_config(self.dataset_configs[k]))
#             for k in self.dataset_configs)
#         if self.wrap:
#             for k in self.datasets:
#                 self.datasets[k] = WrappedDataset(self.datasets[k])

#     def _train_dataloader(self):
#         if self.use_worker_init_fn:
#             init_fn = worker_init_fn
#         else:
#             init_fn = None
#         return DataLoader(self.datasets["train"], batch_size=self.batch_size,
#                           num_workers=self.num_workers, shuffle=True,
#                           worker_init_fn=init_fn, drop_last=True)

#     def _val_dataloader(self, shuffle=False):
#         if self.use_worker_init_fn:
#             init_fn = worker_init_fn
#         else:
#             init_fn = None
#         return DataLoader(self.datasets["validation"],
#                           batch_size=self.batch_size,
#                           num_workers=self.num_workers,
#                           worker_init_fn=init_fn,
#                           shuffle=shuffle, drop_last=True)

#     def _test_dataloader(self, shuffle=False):
#         if self.use_worker_init_fn:
#             init_fn = worker_init_fn
#         else:
#             init_fn = None

#         return DataLoader(self.datasets["test"], batch_size=self.batch_size,
#                           num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

#     def _predict_dataloader(self, shuffle=False):
#         if self.use_worker_init_fn:
#             init_fn = worker_init_fn
#         else:
#             init_fn = None
#         return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
#                           num_workers=self.num_workers, worker_init_fn=init_fn)