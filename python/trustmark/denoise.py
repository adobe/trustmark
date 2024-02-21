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
from omegaconf import OmegaConf
from torchmetrics.functional import peak_signal_noise_ratio
from contextlib import contextmanager



class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'): 
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        
        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)   
        
        # initialize activation 
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'silu':
            self.activation = nn.SiLU(inplace=True)
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)
            
        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)
            
    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)    
        if self.activation:
            x = self.activation(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)




class Identity(nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__()
    def forward(self, x):
        return x
    

class WMRemover(pl.LightningModule):
    def __init__(self,
                 cover_key,
                 secret_key,
                 secret_embedder_config_path,
                 secret_embedder_ckpt_path,
                 denoise_config,
                 discriminator_config,
                 loss_config,
                 ckpt_path="__none__",
                 lr_scheduler='__none__',
                 use_ema=False,
                 is_train=True,
    ):
        super().__init__()
        self.automatic_optimization = False  # for GAN training
        self.cover_key = cover_key
        self.secret_key = secret_key
        
        if is_train:
            secret_embedder_config = OmegaConf.load(secret_embedder_config_path).model
            secret_embedder_config.params.ckpt_path = secret_embedder_ckpt_path
            self.secret_len = secret_embedder_config.params.secret_len

            self.secret_embedder = instantiate_from_config(secret_embedder_config).eval()
            for p in self.secret_embedder.parameters():
                p.requires_grad = False

        self.denoise = instantiate_from_config(denoise_config)

        if is_train:
            self.discriminator = instantiate_from_config(discriminator_config)
            self.loss_layer = instantiate_from_config(loss_config)
        self.use_ema = use_ema
        if use_ema:
            print('Using EMA')
            self.denoise_ema = LitEma(self.denoise)
            self.discriminator_ema = LitEma(self.discriminator)
        if ckpt_path != "__none__":
            self.init_from_ckpt(ckpt_path, ignore_keys=[])
        self.lr_scheduler = None if lr_scheduler == '__none__' else lr_scheduler
    
    def init_from_ckpt(self, path, ignore_keys=[]):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
    
    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.denoise_ema.store(self.denoise.parameters())
            self.discriminator_ema.store(self.discriminator.parameters())
            self.denoise_ema.copy_to(self.denoise)
            self.discriminator_ema.copy_to(self.discriminator)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.denoise_ema.restore(self.denoise.parameters())
                self.discriminator_ema.restore(self.discriminator.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:  # update EMA
            self.denoise_ema(self.denoise)
            self.discriminator_ema(self.discriminator)
    
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
        n = torch.multinomial(torch.tensor([0.5,0.3,0.2]), 1).item() + 1
        stego = image
        for i in range(n):
            secret = torch.zeros_like(secret).random_(0, 2)
            stego = self.secret_embedder(stego, secret)[0]
        # stego = self.secret_embedder(image, secret)[0]
        out = [stego, image, secret]
        return out
    
    def forward(self, x):
        return torch.clamp(self.denoise(x), -1, 1)
    

    @torch.no_grad()
    def log_images(self, batch, fixed_input=False, **kwargs):
        log = dict()
        x, y, s = self.get_input(batch)
        x_denoised = self(x)
        log['clean'] = y
        log['stego'] = x
        log['denoised'] = x_denoised
        residual = x_denoised - y
        log['residual'] = (residual - residual.min()) / (residual.max() - residual.min() + 1e-8)*2 - 1
        return log
    
    

class SimpleUnet(nn.Module):
    def __init__(self, dim=32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim, dim, 3, 2, 1)
        self.conv3 = nn.Conv2d(dim, dim*2, 3, 2, 1)
        self.conv4 = nn.Conv2d(dim*2, dim*4, 3, 2, 1)
        self.conv5 = nn.Conv2d(dim*4, dim*8, 3, 2, 1)
        self.pad6 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up6 = nn.Conv2d(dim*8, dim*4, 2, 1)
        self.upsample6 = nn.Upsample(scale_factor=(2, 2))
        self.conv6 = nn.Conv2d(dim*4 + dim*4, dim*4, 3, 1, 1)
        self.pad7 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up7 = nn.Conv2d(dim*4, dim*2, 2, 1)
        self.upsample7 = nn.Upsample(scale_factor=(2, 2))
        self.conv7 = nn.Conv2d(dim*2 + dim*2, dim*2, 3, 1, 1)
        self.pad8 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up8 = nn.Conv2d(dim*2, dim, 2, 1)
        self.upsample8 = nn.Upsample(scale_factor=(2, 2))
        self.conv8 = nn.Conv2d(dim+dim, dim, 3, 1, 1)
        self.pad9 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up9 = nn.Conv2d(dim, dim, 2, 1)
        self.upsample9 = nn.Upsample(scale_factor=(2, 2))
        self.conv9 = nn.Conv2d(dim + dim + 3, dim, 3, 1, 1)
        self.conv10 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.post = nn.Conv2d(dim, dim//2, 1)
        self.silu = nn.SiLU()
        self.out = nn.Conv2d(dim//2, 3, 1)
    
    def forward(self, image):
        inputs = image

        conv1 = thf.relu(self.conv1(inputs))
        conv2 = thf.relu(self.conv2(conv1))
        conv3 = thf.relu(self.conv3(conv2))
        conv4 = thf.relu(self.conv4(conv3))
        conv5 = thf.relu(self.conv5(conv4))
        up6 = thf.relu(self.up6(self.pad6(self.upsample6(conv5))))
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = thf.relu(self.conv6(merge6))
        up7 = thf.relu(self.up7(self.pad7(self.upsample7(conv6))))
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = thf.relu(self.conv7(merge7))
        up8 = thf.relu(self.up8(self.pad8(self.upsample8(conv7))))
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = thf.relu(self.conv8(merge8))
        up9 = thf.relu(self.up9(self.pad9(self.upsample9(conv8))))
        merge9 = torch.cat([conv1, up9, inputs], dim=1)
        conv9 = thf.relu(self.conv9(merge9))
        conv10 = thf.relu(self.conv10(conv9))
        post = self.silu(self.post(conv10))
        out = thf.tanh(self.out(post))
        return out
    



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
        
   
