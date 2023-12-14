# Copyright 2022 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import json
from PIL import Image 
import numpy as np
from pathlib import Path
import torch 
from torch.utils.data import Dataset, DataLoader
from functools import partial
try:
    import lightning as pl
except ImportError:
    import pytorch_lightning as pl
from .util import instantiate_from_config
import pandas as pd


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id
    return np.random.seed(np.random.get_state()[1][0] + worker_id)


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None, wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True,
                          worker_init_fn=init_fn)

    def _val_dataloader(self, shuffle=False):
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)


class ImageCaptionRaw(Dataset):
    def __init__(self, image_dir, caption_file, secret_len=100, transform=None):
        super().__init__()
        self.image_dir = Path(image_dir)
        self.data = []
        with open(caption_file, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.secret_len = secret_len
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(self.image_dir/item['image']).convert('RGB').resize((512,512))
        caption = item['captions']
        cid = torch.randint(0, len(caption), (1,)).item()
        caption = caption[cid]
        if self.transform is not None:
            image = self.transform(image)

        image = np.array(image, dtype=np.float32)/ 255.0  # normalize to [0, 1]
        target = image * 2.0 - 1.0  # normalize to [-1, 1]
        secret = torch.zeros(self.secret_len, dtype=torch.float).random_(0, 2)
        return dict(image=image, caption=caption, target=target, secret=secret)


class BAMFG(Dataset):
    def __init__(self, style_dir, gt_dir, data_list, transform=None):
        super().__init__()
        self.style_dir = Path(style_dir)
        self.gt_dir = Path(gt_dir)
        self.data = pd.read_csv(data_list)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        gt_img = Image.open(self.gt_dir/item['gt_img']).convert('RGB').resize((512,512))
        style_img = Image.open(self.style_dir/item['style_img']).convert('RGB').resize((512,512))
        txt = item['prompt']
        if self.transform is not None:
            gt_img = self.transform(gt_img)
            style_img = self.transform(style_img)

        gt_img = np.array(gt_img, dtype=np.float32)/ 255.0  # normalize to [0, 1]
        style_img = np.array(style_img, dtype=np.float32)/ 255.0  # normalize to [0, 1]
        target = gt_img * 2.0 - 1.0  # normalize to [-1, 1]

        return dict(image=gt_img, txt=txt, hint=style_img)
