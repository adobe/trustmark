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
# import lightning as PL
from .utils.util import instantiate_from_config
import einops
import kornia
import numpy as np
import torchvision
from omegaconf import OmegaConf
from torchmetrics.functional import peak_signal_noise_ratio
from contextlib import contextmanager
from .ema import LitEma
from .munit import Conv2dBlock, ResBlocks, LinearBlock, MLP, MsImageDis
from . import lr_scheduler


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
    
    def shared_step(self, batch, batch_idx):
        is_training = self.training
        x, y, s = self.get_input(batch)
        if is_training:
            opt_g, opt_d = self.optimizers()
            # train denoiser
            self.toggle_optimizer(opt_g)
        x_denoised = self(x)
        s_pred = self.secret_embedder.decoder(x_denoised)
        s_clean = self.secret_embedder.decoder(y).detach()
        loss, loss_dict = self.loss_layer(x_denoised, y, None, s_clean, s_pred, s)
        g_loss = self.discriminator.calc_gen_loss(x_denoised)

        if (batch_idx % self.loss_layer.generator_update_freq == 0):
            loss += g_loss*self.loss_layer.generator_weight
        else:
            loss += g_loss.detach()*self.loss_layer.generator_weight  # disable g_loss
        loss_dict["g_loss"] = g_loss
        loss_dict['total_loss'] = loss

        if is_training:
            self.manual_backward(loss)
            opt_g.step()
            opt_g.zero_grad()
            self.untoggle_optimizer(opt_g)
            # train discriminator
            self.toggle_optimizer(opt_d)

        d_loss = self.discriminator.calc_dis_loss(x_denoised.detach(), y, is_training)
        loss_dict["d_loss"] = d_loss*self.loss_layer.discriminator_weight

        if is_training:
            self.manual_backward(loss_dict["d_loss"])
            opt_d.step()
            opt_d.zero_grad()
            self.untoggle_optimizer(opt_d)

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
        if self.lr_scheduler:
            sch_g, sch_d = self.lr_schedulers()
            self.log('lr_abs', sch_g.get_lr()[0], prog_bar=True, logger=True, on_step=True, on_epoch=False)
            sch_g.step()
            sch_d.step()

        # return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch, batch_idx)
        loss_dict_no_ema = {f"val/{key}": val for key, val in loss_dict_no_ema.items() if key != 'img_lw'}
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch, batch_idx)
            loss_dict_ema = {'val/' + key + '_ema': loss_dict_ema[key] for key in loss_dict_ema if key != 'img_lw'}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

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
        params_d = list(self.discriminator.parameters())
        opt_g = torch.optim.AdamW(params_g, lr=lr)
        opt_d = torch.optim.AdamW(params_d, lr=lr)
        if self.lr_scheduler == 'CosineAnnealingRestartCyclicLR':
            print(f'Setup LR scheduler {self.lr_scheduler}')
            # fixed lr in 1st and last period, cosine annealing in between
            sch_g = lr_scheduler.CosineAnnealingRestartCyclicLR(opt_g, periods=[100000, 100000, 1000000], eta_mins=[lr, 1e-5, 1e-5], restart_weights=[1, 1, 0.])
            sch_d = lr_scheduler.CosineAnnealingRestartCyclicLR(opt_d, periods=[100000, 100000, 1000000], eta_mins=[lr, 1e-5, 1e-5], restart_weights=[1, 1, 0.])
            return [opt_g, opt_d], [sch_g, sch_d]
        return [opt_g, opt_d], []  


class WMRemover0(pl.LightningModule):
    def __init__(self,
                 cover_key,
                 secret_key,
                 secret_embedder_config_path,
                 secret_embedder_ckpt_path,
                 denoise_config,
                 loss_config,
                 ckpt_path="__none__",
                 lr_scheduler='__none__',
                 use_ema=False,
    ):
        super().__init__()
        self.cover_key = cover_key
        self.secret_key = secret_key
        
        secret_embedder_config = OmegaConf.load(secret_embedder_config_path).model
        secret_embedder_config.params.ckpt_path = secret_embedder_ckpt_path
        self.secret_len = secret_embedder_config.params.secret_len

        self.secret_embedder = instantiate_from_config(secret_embedder_config).eval()
        for p in self.secret_embedder.parameters():
            p.requires_grad = False

        self.denoise = instantiate_from_config(denoise_config)
        self.loss_layer = instantiate_from_config(loss_config)
        self.use_ema = use_ema
        if use_ema:
            print('Using EMA')
            self.denoise_ema = LitEma(self.denoise)
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
            self.denoise_ema.copy_to(self.denoise)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.denoise_ema.restore(self.denoise.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:  # update EMA
            self.denoise_ema(self.denoise)
    
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
    
    def shared_step(self, batch, batch_idx):
        x, y, s = self.get_input(batch)
        x_denoised = self(x)
        s_pred = self.secret_embedder.decoder(x_denoised)
        s_clean = self.secret_embedder.decoder(y).detach()
        loss, loss_dict = self.loss_layer(x_denoised, y, None, s_clean, s_pred, s)

        loss_dict['total_loss'] = loss

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
        if self.lr_scheduler:
            sch_g, sch_d = self.lr_schedulers()
            self.log('lr_abs', sch_g.get_lr()[0], prog_bar=True, logger=True, on_step=True, on_epoch=False)
            sch_g.step()
            sch_d.step()

        # return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch, batch_idx)
        loss_dict_no_ema = {f"val/{key}": val for key, val in loss_dict_no_ema.items() if key != 'img_lw'}
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch, batch_idx)
            loss_dict_ema = {'val/' + key + '_ema': loss_dict_ema[key] for key in loss_dict_ema if key != 'img_lw'}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

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
        opt_g = torch.optim.AdamW(params_g, lr=lr)
        if self.lr_scheduler == 'CosineAnnealingRestartCyclicLR':
            print(f'Setup LR scheduler {self.lr_scheduler}')
            # fixed lr in 1st and last period, cosine annealing in between
            sch_g = lr_scheduler.CosineAnnealingRestartCyclicLR(opt_g, periods=[100000, 100000, 1000000], eta_mins=[lr, 1e-5, 1e-5], restart_weights=[1, 1, 0.])
            return [opt_g], [sch_g]
        return [opt_g], []  
    

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
    
