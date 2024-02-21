# Copyright 2022 Adobe
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
from omegaconf import OmegaConf


class WMRemoverKBNet(pl.LightningModule):
    def __init__(self,
                 cover_key,
                 secret_key,
                 secret_embedder_config_path,
                 secret_embedder_ckpt_path,
                 denoise_config,
                 grad_clip,
                 ckpt_path="__none__",
                 ):
        super().__init__()
        self.automatic_optimization = False  # for GAN training
        self.cover_key = cover_key
        self.secret_key = secret_key
        self.grad_clip = grad_clip
        
        secret_embedder_config = OmegaConf.load(secret_embedder_config_path).model
        secret_embedder_config.params.ckpt_path = secret_embedder_ckpt_path
        self.secret_len = secret_embedder_config.params.secret_len

        self.secret_embedder = instantiate_from_config(secret_embedder_config).eval()
        for p in self.secret_embedder.parameters():
            p.requires_grad = False

        self.denoise = instantiate_from_config(denoise_config)
        if ckpt_path != "__none__":
            self.init_from_ckpt(ckpt_path, ignore_keys=[])
    
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
        stego = self.secret_embedder(image, secret)[0]
        out = [stego, image, secret]
        return out
    
    def forward(self, x):
        return torch.clamp(self.denoise(x), -1, 1)

    def shared_step(self, batch, batch_idx):
        is_training = self.training
        x, y, s = self.get_input(batch)
        if is_training:
            opt_g = self.optimizers()
        x_denoised = self(x)
        loss = torch.abs(x_denoised - y).mean()
        s_pred = self.secret_embedder.decoder(x_denoised)
        loss_dict = {}
        loss_dict['total_loss'] = loss
        loss_dict['bit_acc'] = ((torch.sigmoid(s_pred.detach()) > 0.5).float() == s).float().mean()
        if is_training:
            self.manual_backward(loss)
            if self.grad_clip:
                # torch.nn.utils.clip_grad_norm_(self.denoise.parameters(), 0.01)
                self.clip_gradients(opt_g, gradient_clip_val=0.01, gradient_clip_algorithm="norm")
            opt_g.step()
            opt_g.zero_grad()

        loss_dict['psnr_denoise'] = peak_signal_noise_ratio(x_denoised.detach(), y.detach(), data_range=2.0)
        loss_dict['psnr_stego'] = peak_signal_noise_ratio(x.detach(), y.detach(), data_range=2.0)
    
        return loss, loss_dict
    
    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, batch_idx)
        # logging
        loss_dict = {f"train/{key}": val for key, val in loss_dict.items()}
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)
        
        self.log("global_step", float(self.global_step),
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)
        # if self.use_scheduler:
        #     lr = self.optimizers().param_groups[0]['lr']
        #     self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        sch = self.lr_schedulers()
        self.log('lr_abs', sch.get_lr()[0], prog_bar=True, logger=True, on_step=True, on_epoch=False)
        sch.step()

        # return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch, batch_idx)
        loss_dict_no_ema = {f"val/{key}": val for key, val in loss_dict_no_ema.items() if key != 'img_lw'}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

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
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params_g = list(self.denoise.parameters())
        opt_g = torch.optim.AdamW(params_g, lr=lr, weight_decay=1e-4, betas=(0.9, 0.999))
        lr_sch = lr_scheduler.CosineAnnealingRestartCyclicLR(
                        opt_g, periods=[92000, 208000], restart_weights= [1,1], eta_mins=[0.0003,0.000001])
        return [opt_g], [lr_sch] 



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
        
   
