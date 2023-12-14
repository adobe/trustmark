# Copyright 2022 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import io
import time
import pandas as pd 
import numpy as np
import random
from PIL import Image
from typing import Any, Callable, List, Optional, Tuple
import torch
from torchvision import transforms
from .base_lmdb import PILlmdb, ArrayDatabase
# from . import debug


def worker_init_fn(worker_id):
    # to be passed to torch.utils.data.DataLoader to fix the 
    #  random seed issue with numpy in multi-worker settings
    torch_seed = torch.initial_seed()
    random.seed(torch_seed + worker_id)
    if torch_seed >= 2**30:  # make sure torch_seed + workder_id < 2**32
        torch_seed = torch_seed % 2**30
    np.random.seed(torch_seed + worker_id)


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def dataset_wrapper(data_dir, data_list, **kwargs):
    if os.path.exists(os.path.join(data_dir, 'data.mdb')):
        return ImageDataset(data_dir, data_list, **kwargs)
    else:
        return ImageFolder(data_dir, data_list, **kwargs)


class ImageFolder(torch.utils.data.Dataset):
    _repr_indent = 4
    def __init__(self, data_dir, data_list, secret_len=100, resize=256, transform=None, **kwargs):
        super().__init__()
        if transform is None:  # random resized crop, good for training
            self.transform = transforms.RandomResizedCrop((resize, resize), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333))
        elif hasattr(transform, '__call__'):  # a callable transform
            self.transform = transform
        else:
            self.transform = transforms.Resize((resize, resize))
        self.build_data(data_dir, data_list, **kwargs)
        self.kwargs = kwargs
        self.secret_len = secret_len
    
    def build_data(self, data_dir, data_list, **kwargs):
        self.data_dir = data_dir
        if isinstance(data_list, list):
            self.data_list = data_list
        elif isinstance(data_list, str):
            self.data_list = pd.read_csv(data_list)['path'].tolist()
        elif isinstance(data_list, pd.DataFrame):
            self.data_list = data_list['path'].tolist()
        else:
            raise ValueError('data_list must be a list, str or pd.DataFrame')
        self.N = len(self.data_list)
    
    def __getitem__(self, index):
        path = self.data_list[index]
        img = pil_loader(os.path.join(self.data_dir, path))
        img = self.transform(img)
        img = np.array(img, dtype=np.float32)/127.5-1.  # [-1, 1]
        secret = torch.zeros(self.secret_len, dtype=torch.float).random_(0, 2)
        return {'image': img, 'secret': secret}  # {'img': x, 'index': index}

    def __len__(self) -> int:
        # raise NotImplementedError
        return self.N 

class ImageDataset(torch.utils.data.Dataset):
    r"""
    Customised Image Folder class for pytorch.
    Accept lmdb and a csv list as the input.
    Usage:
        dataset = ImageDataset(img_dir, img_list)
        dataset.set_transform(some_pytorch_transforms)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True,
            num_workers=4, worker_init_fn=worker_init_fn)
        for x,y in loader:
            # x and y is input and target (dict), the keys can be customised.
    """
    _repr_indent = 4
    def __init__(self, data_dir, data_list, secret_len=100, resize=None,  transform=None, target_transform=None, **kwargs):
        super().__init__()
        if resize is not None:
            self.resize = transforms.Resize((resize, resize))
        self.set_transform(transform, target_transform)
        self.build_data(data_dir, data_list, **kwargs)
        self.secret_len = secret_len
        self.kwargs = kwargs

    def set_transform(self, transform, target_transform=None):
        self.transform, self.target_transform = transform, target_transform

    def build_data(self, data_dir, data_list, **kwargs):
        """
        Args:
            data_list    (text file) must have at least 3 fields: id, path and label

        This method must create an attribute self.samples containing ID, input and target samples; and another attribute N storing the dataset size

        Optional attributes: classes (list of unique classes), group (useful for 
        metric learning)
        """
        self.data_dir, self.list = data_dir, data_list
        if ('dtype' in kwargs) and (kwargs['dtype'].lower() == 'array'):
            data = ArrayDatabase(data_dir, data_list)
        else:
            data = PILlmdb(data_dir, data_list, **kwargs)
        self.N = len(data)
        self.classes = np.unique(data.labels)
        self.samples = {'x': data, 'y': data.labels}
        # assert isinstance(data_list, str) or isinstance(data_list, pd.DataFrame)
        # df = pd.read_csv(data_list) if isinstance(data_list, str) else data_list
        # assert 'id' in df and 'label' in df, f'[DATA] Error! {data_list} must contains "id" and "label".'
        # ids = df['id'].tolist()
        # labels = np.array(df['label'].tolist())
        # data = PILlmdb(data_dir)
        # assert set(ids).issubset(set(data.keys))  # ids should exist in lmdb
        # self.N = len(ids)
        # self.classes, inds = np.unique(labels, return_index=True)
        # self.samples = {'id': ids, 'x': data, 'y': labels}

    def set_ids(self, ids):
        self.samples['x'].set_ids(ids)
        self.samples['y'] = [self.samples['y'][i] for i in ids]
        self.N = len(self.samples['x'])

    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index
        Returns:
            dict: (x: sample, y: target, **kwargs)
        """
        x, y = self.samples['x'][index], self.samples['y'][index]
        if hasattr(self, 'resize'):
            x = self.resize(x)
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        x = np.array(x, dtype=np.float32)/127.5-1.
        secret = torch.zeros(self.secret_len, dtype=torch.float).random_(0, 2)
        return {'image': x, 'secret': secret}  # {'img': x, 'index': index}

    def __len__(self) -> int:
        # raise NotImplementedError
        return self.N 

    def __repr__(self) -> str:
        head = "\nDataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if hasattr(self, 'data_dir') and self.data_dir is not None:
            body.append("data_dir location: {}".format(self.data_dir))
        if hasattr(self, 'kwargs'):
            body.append(f'kwargs: {self.kwargs}')
        body += self.extra_repr().splitlines()
        if hasattr(self, "transform") and self.transform is not None:
            body += [repr(self.transform)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self) -> str:
        return ""

