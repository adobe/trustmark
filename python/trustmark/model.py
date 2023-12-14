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
from torchmetrics.functional import peak_signal_noise_ratio
from contextlib import contextmanager
from .ema import LitEma
from . import lr_scheduler
from .munit import Conv2dBlock, ResBlocks, LinearBlock, MLP, MsImageDis


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
    
    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.encoder_ema.store(self.encoder.parameters())
            self.decoder_ema.store(self.decoder.parameters())
            self.discriminator_ema.store(self.discriminator.parameters())
            self.encoder_ema.copy_to(self.encoder)
            self.decoder_ema.copy_to(self.decoder)
            self.discriminator_ema.copy_to(self.discriminator)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.encoder_ema.restore(self.encoder.parameters())
                self.decoder_ema.restore(self.decoder.parameters())
                self.discriminator_ema.restore(self.discriminator.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.encoder_ema(self.encoder)
            self.decoder_ema(self.decoder)
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

    def shared_step(self, batch, batch_idx):
        is_train = self.training
        x, s = self.get_input(batch)
        if is_train:
            opt_g, opt_d = self.optimizers()
            # train generator
            self.toggle_optimizer(opt_g)
        stego, residual = self(x, s)
        if hasattr(self, "noise") and self.noise.is_activated():
            stego_noised = self.noise(stego, self.global_step, p=0.9)
        else:
            stego_noised = self.crop(stego)
        stego_noised = torch.clamp(stego_noised, -1, 1)
        spred = self.decoder(stego_noised)

        loss, loss_dict = self.loss_layer(x, stego, None, s, spred, self.global_step)
        g_loss = self.discriminator.calc_gen_loss(stego)
        
        if self.update_gen and (batch_idx % self.loss_layer.generator_update_freq == 0):
            loss += g_loss*loss_dict['img_lw']*self.loss_layer.generator_weight
        else:
            loss += g_loss.detach()*loss_dict['img_lw']*self.loss_layer.generator_weight  # disable g_loss
        loss_dict["g_loss"] = g_loss

        if is_train:
            self.manual_backward(loss)
            opt_g.step()
            opt_g.zero_grad()
            self.untoggle_optimizer(opt_g)
            # train discriminator
            self.toggle_optimizer(opt_d)

        d_loss = self.discriminator.calc_dis_loss(stego.detach(), x, is_train)
        loss_dict["d_loss"] = d_loss*loss_dict['img_lw']*self.loss_layer.discriminator_weight

        if is_train:
            self.manual_backward(loss_dict["d_loss"])
            opt_d.step()
            opt_d.zero_grad()
            self.untoggle_optimizer(opt_d)

        loss_dict['psnr'] = peak_signal_noise_ratio(stego.detach(), x.detach(), data_range=2.0)
        bit_acc = loss_dict["g_loss/bit_acc"]
        bit_acc_ = bit_acc.item()

        if is_train:
            if (bit_acc_ > self.bit_acc_thresholds[2]) and (not self.update_gen) and (not self.fixed_input):
                if (not hasattr(self, 'noise')) or (hasattr(self, 'noise') and self.noise.is_activated()):
                    self.loss_layer.activate_ramp(self.global_step)
                    self.update_gen = ~self.update_gen

            if (bit_acc_ > self.bit_acc_thresholds[1]) and (not self.fixed_input):  # ramp up image loss at late training stage
                if hasattr(self, 'noise') and (not self.noise.is_activated()):
                    self.noise.activate(self.global_step) 

            if (bit_acc_ > self.bit_acc_thresholds[0]) and self.fixed_input:  # execute only once
                print(f'[TRAINING] High bit acc ({bit_acc_}) achieved, switch to full image dataset training.')
                self.fixed_input = ~self.fixed_input
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
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

        if self.use_ema:
            with self.ema_scope():
                _, loss_dict_ema = self.shared_step(batch, batch_idx)
                loss_dict_ema = {'val/' + key + '_ema': loss_dict_ema[key] for key in loss_dict_ema if key != 'img_lw'}
            self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch, batch_idx)
        loss_dict_no_ema = {f"test/{key}": val for key, val in loss_dict_no_ema.items() if key in ['gloss/bit_acc', 'gloss/psnr']}
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch, batch_idx)
            loss_dict_ema = {'test/' + key + '_ema': loss_dict_ema[key] for key in loss_dict_ema if key in ['gloss/bit_acc', 'gloss/psnr']}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
    
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
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params_g = list(self.encoder.parameters()) + list(self.decoder.parameters())
        params_d = list(self.discriminator.parameters())
        opt_g = torch.optim.AdamW(params_g, lr=lr)
        opt_d = torch.optim.AdamW(params_d, lr=lr)
        if self.lr_scheduler == 'CosineAnnealingRestartCyclicLR':
            print(f'Setup LR scheduler {self.lr_scheduler}')
            # fixed lr in 1st and last period, cosine annealing in between
            sch_g = lr_scheduler.CosineAnnealingRestartCyclicLR(opt_g, periods=[200000, 100000, 1000000], eta_mins=[lr, 1e-5, 1e-5], restart_weights=[1, 1, 0.])
            sch_d = lr_scheduler.CosineAnnealingRestartCyclicLR(opt_d, periods=[200000, 100000, 1000000], eta_mins=[lr, 1e-5, 1e-5], restart_weights=[1, 1, 0.])
            return [opt_g, opt_d], [sch_g, sch_d]
        return [opt_g, opt_d], []

