# Copyright 2023 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import math
import torch
from torch import nn
from torch.nn import functional as thf
try:
    import lightning as pl
except ImportError:
    import pytorch_lightning as pl
import einops
import kornia
import numpy as np
import torchvision
import importlib
from torchmetrics.functional import peak_signal_noise_ratio
from contextlib import contextmanager


class Identity(nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__()
    def forward(self, x):
        return x
    

class TrustMark_Arch(pl.LightningModule):
    def __init__(self,
                 cover_key,
                 secret_key,
                 secret_len,
                 resolution,
                 secret_encoder_config,
                 secret_decoder_config,
                 discriminator_config,
                 loss_config,
                 bit_acc_thresholds=[0.9, 0.95, 0.98],
                 noise_config='__none__',
                 ckpt_path="__none__",
                 lr_scheduler='__none__',
                 use_ema=False
                 ):
        super().__init__()
        self.automatic_optimization = False
        self.cover_key = cover_key
        self.secret_key = secret_key
        secret_encoder_config.params.secret_len = secret_len
        secret_decoder_config.params.secret_len = secret_len
        secret_encoder_config.params.resolution = resolution
        secret_decoder_config.params.resolution = 224
        self.encoder = instantiate_from_config(secret_encoder_config)
        self.decoder = instantiate_from_config(secret_decoder_config)
        self.loss_layer = instantiate_from_config(loss_config)
        self.discriminator = instantiate_from_config(discriminator_config)

        if noise_config != '__none__':
            self.noise = instantiate_from_config(noise_config)
        
        self.lr_scheduler = None if lr_scheduler == '__none__' else lr_scheduler

        self.use_ema = use_ema
        if self.use_ema:
            print('Using EMA')
            self.encoder_ema = LitEma(self.encoder)
            self.decoder_ema = LitEma(self.decoder)
            self.discriminator_ema = LitEma(self.discriminator)
            print(f"Keeping EMAs of {len(list(self.encoder_ema.buffers()) + list(self.decoder_ema.buffers()) + list(self.discriminator_ema.buffers()))}.")

        if ckpt_path != "__none__":
            self.init_from_ckpt(ckpt_path, ignore_keys=[])
        
        # early training phase
        self.fixed_img = None
        self.fixed_secret = None
        self.register_buffer("fixed_input", torch.tensor(True))
        self.register_buffer("update_gen", torch.tensor(False))  # update generator to fool discriminator
        self.bit_acc_thresholds = bit_acc_thresholds
        if noise_config == '__none__' or noise_config.target == 'cldm.transformations.TransformNet':  # no noise or imagenetc
            print('Noise model from transformations.py (ImagenetC)')
            self.crop = Identity()
        else:
            self.crop = kornia.augmentation.CenterCrop((224, 224), cropping_mode="resample")  # early training phase
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
    

    
    @torch.no_grad()
    def get_input(self, batch, bs=None):
        image = batch[self.cover_key]
        secret = batch[self.secret_key]
        if bs is not None:
            image = image[:bs]
            secret = secret[:bs]
        else:
            bs = image.shape[0]
        # encode image 1st stage
        image = einops.rearrange(image, "b h w c -> b c h w").contiguous()
        
        # check if using fixed input (early training phase)
        # if self.training and self.fixed_input:
        if self.fixed_input:
            if self.fixed_img is None:  # first iteration
                print('[TRAINING] Warmup - using fixed input image for now!')
                self.fixed_img = image.detach().clone()[:bs]
                self.fixed_secret = secret.detach().clone()[:bs]  # use for log_images with fixed_input option only
            image = self.fixed_img
            new_bs = min(secret.shape[0], image.shape[0])
            image, secret = image[:new_bs], secret[:new_bs]
        
        out = [image, secret]
        return out
    
    def forward(self, cover, secret):
        # return a tuple (stego, residual)
        enc_out = self.encoder(cover, secret)
        if hasattr(self.encoder, 'return_residual') and self.encoder.return_residual:
            return cover + enc_out, enc_out
        else:
            return enc_out, enc_out - cover


    
    @torch.no_grad()
    def log_images(self, batch, fixed_input=False, **kwargs):
        log = dict()
        if fixed_input and self.fixed_img is not None:
            x, s = self.fixed_img, self.fixed_secret
        else:
            x, s = self.get_input(batch)
        stego, residual = self(x, s)
        if hasattr(self, 'noise') and self.noise.is_activated():
            img_noise = self.noise(stego, self.global_step, p=1.0)
            log['noised'] = img_noise
        log['input'] = x
        log['stego'] = stego
        log['residual'] = (residual - residual.min()) / (residual.max() - residual.min() + 1e-8)*2 - 1
        return log
    

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


        
def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))
        
   
