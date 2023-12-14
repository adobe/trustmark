# Copyright 2022 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import os
from .transformations import jpeg_compress_decompress, round_only_at_0
import torch
import numpy as np
from torch import nn
import torch.nn.functional as thf
from PIL import Image
import kornia as ko
import albumentations as ab
from torchvision import transforms
from typing import Any, Dict, Optional, Tuple, Union
from kornia.constants import Resample
from kornia.core import Tensor
from .augment_imagenetc import RandomImagenetC


class IdentityAugment(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, **kwargs):
        return x


class BaseTransform(nn.Module):
    # base transform for kornia augmentation
    # support ramping in transform severity
    def __init__(self, tform):
        super().__init__()
        self.tform = tform
    
    def get_params(self, input, params = None, **kwargs):
        in_tensor = self.tform.__unpack_input__(input)
        in_tensor = self.tform.transform_tensor(in_tensor)
        batch_shape = in_tensor.shape
        if params is None:
            params = self.tform.forward_parameters(batch_shape)

        if 'batch_prob' not in params:
            params['batch_prob'] = ko.core.tensor([True] * batch_shape[0])

        params, flags = self.tform._process_kwargs_to_params_and_flags(params, self.tform.flags, **kwargs)
        return params, flags, in_tensor
    
    def apply_ramp(self, params, flags, ramp):
        # override this method to apply ramping
        raise NotImplementedError

    def forward(self, x, **kwargs):
        if 'ramp' in kwargs:
            params, flags, x_tensor = self.get_params(x, **kwargs)
            params, flags = self.apply_ramp(params, flags, kwargs['ramp'])
            output = self.tform.apply_func(x_tensor, params, flags)
            return self.transform_output_tensor(output, x_tensor.shape) if self.tform.keepdim else output
        else:
            return self.tform(x, **kwargs)
        

class RandomCompress(nn.Module):
    def __init__(self, severity='medium', p=0.5):
        super().__init__()
        self.p = p
        if severity == 'low':
            self.jpeg_quality = 70
        elif severity == 'medium':
            self.jpeg_quality = 50
        elif severity == 'high':
            self.jpeg_quality = 40
    
    def forward(self, x, ramp=1.):
        # x (B, C, H, W) in range [0, 1]
        # ramp: adjust the ramping of the compression, 1.0 means min quality = self.jpeg_quality
        if torch.rand(1)[0] >= self.p:
            return x
        jpeg_quality = 100. - torch.rand(1)[0] * ramp * (100. - self.jpeg_quality)
        x = jpeg_compress_decompress(x, rounding=round_only_at_0, quality=jpeg_quality)
        return x

class ImagenetCTransform(nn.Module):
    def __init__(self, severity='medium', p=0.5):
        super().__init__()
        self.p = p
        if severity == 'low':
            self.severity = 2
        elif severity == 'medium':
            self.severity = 3
        elif severity == 'high':
            self.severity = 5
        self.tform = RandomImagenetC(max_severity=self.severity, phase='train')
    
    def forward(self, x, ramp=1.):
        # x (B, C, H, W) in range [0, 1]
        if torch.rand(1)[0] >= self.p:
            return x
        img0 = x.detach().cpu().numpy()
        img = img0 * 255  # [0, 1] -> [0, 255]
        img = img.transpose(0, 2, 3, 1).astype(np.uint8)
        img = [Image.fromarray(i) for i in img]
        img = [self.tform(i) for i in img]
        img = np.array([np.array(i) for i in img], dtype=np.float32)
        img = img.transpose(0, 3, 1, 2) / 255.  # [0, 255] -> [0, 1]
        residual = torch.from_numpy(img - img0).to(x.device)
        x = x + residual
        return x 
    

class RandomResizedCrop(ko.augmentation.RandomResizedCrop):
    def __init__(
        self,
        size = (224, 224),
        scale = (0.7, 1.0),
        ratio = (3.0 / 4.0, 4.0 / 3.0),
        resample = 'BILINEAR',
        same_on_batch = False,
        align_corners = True,
        p = 1.0,
        keepdim = False,
        cropping_mode = "resample",
    ) -> None:
        super().__init__(size, scale, ratio, resample, same_on_batch, align_corners, p, keepdim, cropping_mode)
        self.p_size = size
        self.p_scale = scale
        self.p_ratio = ratio

    def forward(self, x, ramp=1.0):
        # apply ramp to scale and ratio
        def ramping(param):
            return 1.0 + ramp * (param - 1.0)
        self._param_generator.scale = (ramping(self.p_scale[0]), ramping(self.p_scale[1]))
        self._param_generator.ratio = (ramping(self.p_ratio[0]), ramping(self.p_ratio[1]))
        return super().forward(x)

class RandomResizedCrop2(ko.augmentation.RandomResizedCrop):
    # also vary size
    def __init__(
        self,
        size = (0.5, 1.0),
        scale = (0.7, 1.0),
        ratio = (3.0 / 4.0, 4.0 / 3.0),
        resample = 'BILINEAR',
        same_on_batch = False,
        align_corners = True,
        p = 1.0,
        keepdim = False,
        cropping_mode = "resample",
    ) -> None:
        super().__init__((224,224), scale, ratio, resample, same_on_batch, align_corners, p, keepdim, cropping_mode)
        self.p_size = size
        self.p_scale = scale
        self.p_ratio = ratio

    def forward(self, x, ramp=1.0):
        # apply ramp to scale and ratio
        def ramping(param):
            return 1.0 + ramp * (param - 1.0)
        s0 = x.shape[-1]
        p_low = min(0.99, self.p_size[1] - (self.p_size[1] - self.p_size[0]) * ramp)
        s = torch.randint(int(p_low * s0), int(s0), (1,))[0]
        self._param_generator.size = (s, s)
        self._param_generator.scale = (ramping(self.p_scale[0]), ramping(self.p_scale[1]))
        self._param_generator.ratio = (ramping(self.p_ratio[0]), ramping(self.p_ratio[1]))
        return super().forward(x)

class RandomFlip(BaseTransform):
    def __init__(self, severity='medium', p=0.5):
        self.p = p
        tform = ko.augmentation.RandomHorizontalFlip(p=p)
        super().__init__(tform)
    
    def apply_ramp(self, params, flags, ramp):
        p = self.p * ramp
        params['batch_prob'] = (torch.zeros_like(params['batch_prob']).uniform_(0,1) < p).float()
        return params, flags
    

class RandomGrayscale(BaseTransform):
    def __init__(self, severity='medium', p=0.5):
        self.p = p
        tform = ko.augmentation.RandomGrayscale(p=p)
        super().__init__(tform)
    
    def apply_ramp(self, params, flags, ramp):
        p = self.p * ramp
        params['batch_prob'] = (torch.zeros_like(params['batch_prob']).uniform_(0,1) < p).float()
        return params, flags


class RandomRotate(BaseTransform):
    def __init__(self, severity='medium', p=0.5):
        self.p = p
        if severity == 'low':
            self.degrees = (-5, 5)
        elif severity == 'medium':
            self.degrees = (-10, 10)
        elif severity == 'high':
            self.degrees = (-15, 15)
        tform = ko.augmentation.RandomAffine(self.degrees, padding_mode='border', p=p)
        super().__init__(tform)
    
    def apply_ramp(self, params, flags, ramp):
        params['angle'] = params['angle'] * ramp
        return params, flags


class RandomBoxBlur(BaseTransform):
    def __init__(self, severity='medium', border_type='reflect', normalized=True, p=0.5):
        self.p = p
        if severity == 'low':
            kernel_size = 3
        elif severity == 'medium':
            kernel_size = 5
        elif severity == 'high':
            kernel_size = 7
        
        tform = ko.augmentation.RandomBoxBlur(kernel_size=(kernel_size, kernel_size), border_type=border_type, normalized=normalized, p=self.p)
        super().__init__(tform)
    
    def apply_ramp(self, params, flags, ramp):
        return params, flags


class RandomMedianBlur(BaseTransform):
    def __init__(self, severity='medium', p=0.5):
        self.p = p
        tform = ko.augmentation.RandomMedianBlur(kernel_size=(3,3), p=p)
        super().__init__(tform)
    
    def apply_ramp(self, params, flags, ramp):
        return params, flags


class RandomBrightness(BaseTransform):
    def __init__(self, severity='medium', p=0.5):
        self.p = p
        if severity == 'low':
            brightness = (0.9, 1.1)
        elif severity == 'medium':
            brightness = (0.75, 1.25)
        elif severity == 'high':
            brightness = (0.5, 1.5)
        tform = ko.augmentation.RandomBrightness(brightness=brightness, p=p)
        super().__init__(tform)

    def apply_ramp(self, params, flags, ramp):
        params['brightness_factor'] = 1 + (params['brightness_factor']-1) * ramp
        return params, flags


class RandomContrast(BaseTransform):
    def __init__(self, severity='medium', p=0.5):
        self.p = p
        if severity == 'low':
            contrast = (0.9, 1.1)
        elif severity == 'medium':
            contrast = (0.75, 1.25)
        elif severity == 'high':
            contrast = (0.5, 1.5)
        tform = ko.augmentation.RandomContrast(contrast=contrast, p=p)
        super().__init__(tform)

    def apply_ramp(self, params, flags, ramp):
        params['contrast_factor'] = 1 + (params['contrast_factor']-1) * ramp
        return params, flags


class RandomSaturation(BaseTransform):
    def __init__(self, severity='medium', p=0.5):
        self.p = p
        if severity == 'low':
            sat = (0.9, 1.1)
        elif severity == 'medium':
            sat = (0.75, 1.25)
        elif severity == 'high':
            sat = (0.5, 1.5)
        tform = ko.augmentation.RandomSaturation(saturation=sat, p=p)
        super().__init__(tform)

    def apply_ramp(self, params, flags, ramp):
        params['saturation_factor'] = 1 + (params['saturation_factor']-1) * ramp
        return params, flags

class RandomSharpness(BaseTransform):
    def __init__(self, severity='medium', p=0.5):
        self.p = p
        if severity == 'low':
            sharpness = 0.5
        elif severity == 'medium':
            sharpness = 1.0
        elif severity == 'high':
            sharpness = 2.5
        tform = ko.augmentation.RandomSharpness(sharpness=sharpness, p=p)
        super().__init__(tform)

    def apply_ramp(self, params, flags, ramp):
        params['sharpness'] *= ramp
        return params, flags

class RandomColorJiggle(BaseTransform):
    def __init__(self, severity='medium', p=0.5):
        self.p = p
        if severity == 'low':
            factor = (0.05, 0.05, 0.05, 0.01)
        elif severity == 'medium':
            factor = (0.1, 0.1, 0.1, 0.02)
        elif severity == 'high':
            factor = (0.1, 0.1, 0.1, 0.05)
        tform = ko.augmentation.ColorJiggle(*factor, p=p)
        super().__init__(tform)

    def apply_ramp(self, params, flags, ramp):
        for key in ['brightness_factor', 'contrast_factor', 'saturation_factor']:
            params[key] = 1 + (params[key]-1) * ramp
        params['hue_factor'] = params['hue_factor'] * ramp
        return params, flags
    

class RandomHue(BaseTransform):
    def __init__(self, severity='medium', p=0.5):
        self.p = p
        if severity == 'low':
            hue = 0.01
        elif severity == 'medium':
            hue = 0.02
        elif severity == 'high':
            hue = 0.05
        tform = ko.augmentation.RandomHue(hue=(-hue, hue), p=p)
        super().__init__(tform)

    def apply_ramp(self, params, flags, ramp):
        params['hue_factor'] *= ramp
        return params, flags
    

class RandomGamma(BaseTransform):
    def __init__(self, severity='medium', p=0.5):
        self.p = p
        if severity == 'low':
            gamma, gain = (0.9, 1.1), (0.9,1.1)
        elif severity == 'medium':
            gamma, gain = (0.75, 1.25), (0.75,1.25)
        elif severity == 'high':
            gamma, gain = (0.5, 1.5), (0.5,1.5)
        tform = ko.augmentation.RandomGamma(gamma, gain, p=p)
        super().__init__(tform)

    def apply_ramp(self, params, flags, ramp):
        params['gamma_factor'] = 1 + (params['gamma_factor']-1) * ramp
        params['gain_factor'] = 1 + (params['gain_factor']-1) * ramp
        return params, flags


class RandomGaussianBlur(BaseTransform):
    def __init__(self, severity='medium', p=0.5):
        self.p = p
        if severity == 'low':
            kernel_size, sigma = 3, (0.1, 1.0)
        elif severity == 'medium':
            kernel_size, sigma = 5, (0.1, 1.5)
        elif severity == 'high':
            kernel_size, sigma = 7, (0.1, 2.0)
        tform = ko.augmentation.RandomGaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=sigma, p=self.p)
        super().__init__(tform)

    def apply_ramp(self, params, flags, ramp):
        params['sigma'] *= ramp
        return params, flags


class RandomGaussianNoise(BaseTransform):
    def __init__(self, severity='medium', p=0.5):
        self.p = p
        if severity == 'low':
            std = 0.02
        elif severity == 'medium':
            std = 0.04
        elif severity == 'high':
            std = 0.08
        tform = ko.augmentation.RandomGaussianNoise(mean=0., std=std, p=p)
        super().__init__(tform)

    def apply_ramp(self, params, flags, ramp):
        return params, flags
    
    
class RandomMotionBlur(BaseTransform):
    def __init__(self, severity='medium', p=0.5):
        self.p = p
        if severity == 'low':
            kernel_size, angle, direction = (3, 5), (-25, 25), (-0.25, 0.25)
        elif severity == 'medium':
            kernel_size, angle, direction = (3, 7), (-45, 45), (-0.5, 0.5)
        elif severity == 'high':
            kernel_size, angle, direction = (3, 9), (-90, 90), (-1.0, 1.0)
        tform = ko.augmentation.RandomMotionBlur(kernel_size, angle, direction, p=p)
        super().__init__(tform)

    def apply_ramp(self, params, flags, ramp):
        params['angle_factor'] *= ramp
        params['direction_factor'] *= ramp
        return params, flags


class RandomPosterize(BaseTransform):
    def __init__(self, severity='medium', p=0.5):
        self.p = p
        if severity == 'low':
            bits = 5
        elif severity == 'medium':
            bits = 4
        elif severity == 'high':
            bits = 3
        tform = ko.augmentation.RandomPosterize(bits=bits, p=p)
        super().__init__(tform)

    def apply_ramp(self, params, flags, ramp):
        return params, flags


class RandomRGBShift(BaseTransform):
    def __init__(self, severity='medium', p=0.5):
        self.p = p
        if severity == 'low':
            rgb = 0.02
        elif severity == 'medium':
            rgb = 0.05
        elif severity == 'high':
            rgb = 0.1
        tform = ko.augmentation.RandomRGBShift(r_shift_limit=rgb, g_shift_limit=rgb, b_shift_limit=rgb, p=p)
        super().__init__(tform)

    def apply_ramp(self, params, flags, ramp):
        for key in ['r_shift', 'g_shift', 'b_shift']:
            params[key] = params[key] * ramp
        return params, flags


class TransformNet(nn.Module):
    def __init__(self, flip=True, crop_mode='random_crop', rotate=False, compress=True, brightness=True, contrast=True, color_jiggle=True, gamma=False, grayscale=True, gaussian_blur=True, gaussian_noise=True, hue=True, motion_blur=True, posterize=True, rgb_shift=True, saturation=True, sharpness=True, median_blur=True, box_blur=True, imagenetc=False, severity='medium', n_optional=2, ramp=1000, p=0.5):
        super().__init__()
        self.n_optional = n_optional
        self.p = p
        self.ramp = ramp
        self.register_buffer('step0', torch.tensor(0))

        # flip
        p_flip = 0.5 if flip else 0
        rnd_flip_layer = RandomFlip(p_flip)
        self.register(rnd_flip_layer, 'Random Flip', 'fixed')

        # crop/resized crop
        self.crop_mode = crop_mode
        assert crop_mode in ['random_crop', 'resized_crop']
        if crop_mode == 'random_crop':
            rnd_crop_layer = ko.augmentation.RandomCrop((224,224), cropping_mode="resample")
        elif crop_mode == 'resized_crop':
            rnd_crop_layer = RandomResizedCrop(size=(224,224), scale=(0.7, 1.0), ratio=(3.0/4, 4.0/3), cropping_mode='resample')
        self.register(rnd_crop_layer, 'Random Crop', 'fixed')

        if rotate:
            rnd_rotate = RandomRotate(severity, p=p)
            self.register(rnd_rotate, 'Random Rotate', 'fixed')
        
        if compress:
            self.register(RandomCompress(severity, p=p), 'Random Compress')
        if brightness:
            self.register(RandomBrightness(severity, p=p), 'Random Brightness')
        if contrast:
            self.register(RandomContrast(severity, p=p), 'Random Contrast')
        if color_jiggle:
            self.register(RandomColorJiggle(severity, p=p), 'Random Color')
        if gamma:
            self.register(RandomGamma(severity, p=p), 'Random Gamma')
        if grayscale:
            self.register(RandomGrayscale(p=p), 'Grayscale')
        if gaussian_blur:
            self.register(RandomGaussianBlur(severity, p=p), 'Random Gaussian Blur')
        if gaussian_noise:
            self.register(RandomGaussianNoise(severity, p=p), 'Random Gaussian Noise')
        if hue:
            self.register(RandomHue(severity, p=p), 'Random Hue')
        if motion_blur:
            self.register(RandomMotionBlur(severity, p=p), 'Random Motion Blur')
        if posterize:
            self.register(RandomPosterize(severity, p=p), 'Random Posterize')
        if rgb_shift:
            self.register(RandomRGBShift(severity, p=p), 'Random RGB Shift')
        if saturation:
            self.register(RandomSaturation(severity, p=p), 'Random Saturation')
        if sharpness:
            self.register(RandomSharpness(severity, p=p), 'Random Sharpness')
        if median_blur:
            self.register(RandomMedianBlur(severity, p=p), 'Random Median Blur')
        if box_blur:
            self.register(RandomBoxBlur(severity, p=p), 'Random Box Blur')
        if imagenetc:
            self.register(ImagenetCTransform(severity, p=p), 'Random ImageNetC')

    def register(self, tform, name, mode='optional'):
        # register a new fixed or optional transform
        assert mode in ['fixed', 'optional']
        if mode == 'fixed':
            if not hasattr(self, 'fixed_transforms'):
                self.fixed_transforms = []
                self.fixed_names = []
            self.fixed_transforms.append(tform)
            self.fixed_names.append(name)
        else:
            if not hasattr(self, 'optional_transforms'):
                self.optional_transforms = []
                self.optional_names = []
            self.optional_transforms.append(tform)
            self.optional_names.append(name)

    def activate(self, global_step):
        if self.step0 == 0:
            print(f'[TRAINING] Activating TransformNet at step {global_step}')
            self.step0 = torch.tensor(global_step)
    
    def is_activated(self):
        return self.step0 > 0
    
    def forward(self, x, global_step, p=0.9):
        # x: [batch_size, 3, H, W] in range [-1, 1]
        x = x * 0.5 + 0.5  # [-1, 1] -> [0, 1]
        ramp = np.min([(global_step-self.step0.cpu().item()) / self.ramp, 1.])

        # fixed transforms
        for tform in self.fixed_transforms:
            x = tform(x, ramp=ramp)
            if isinstance(x, tuple):
                x = x[0]
        
        # optional transforms
        if len(self.optional_transforms) > 0:
            tform_ids = torch.randint(len(self.optional_transforms), (self.n_optional,)).numpy()
            for tform_id in tform_ids:
                tform = self.optional_transforms[tform_id]
                x = tform(x, ramp=ramp)
                if isinstance(x, tuple):
                    x = x[0]

        return x * 2 - 1  # [0, 1] -> [-1, 1]
    
    def transform_by_id(self, x, tform_id):
        # x: [batch_size, 3, H, W] in range [-1, 1]
        x = x * 0.5 + 0.5  # [-1, 1] -> [0, 1]
        # fixed transforms
        for tform in self.fixed_transforms:
            x = tform(x)
            if isinstance(x, tuple):
                x = x[0]
        
        # optional transforms
        tform = self.optional_transforms[tform_id]
        x = tform(x)
        if isinstance(x, tuple):
            x = x[0]
        return x * 2 - 1  # [0, 1] -> [-1, 1]
    
    def transform_by_name(self, x, tform_name):
        assert tform_name in self.fixed_names + self.optional_names
        if tform_name in self.fixed_names:
            tform = self.fixed_transforms[self.fixed_names.index(tform_name)]
        else:
            tform = self.optional_transforms[self.optional_names.index(tform_name)]
        x = x * 0.5 + 0.5  # [-1, 1] -> [0, 1]
        x = tform(x)
        if isinstance(x, tuple):
            x = x[0]
        return x * 2 - 1  # [0, 1] -> [-1, 1]
        # tform_id = self.optional_names.index(tform_name)
        # return self.transform_by_id(x, tform_id)

    def apply_transform_on_pil_image(self, x, tform_name):
        # x: PIL image
        # return: PIL image
        assert tform_name in self.optional_names + ['Fixed Augment']
        # if tform_name == 'Random Crop':  # the only transform dependent on image size
        #     # crop equivalent to 224/256
        #     w, h = x.size
        #     new_w, new_h = int(224 / 256 * w), int(224 / 256 * h)
        #     x = transforms.RandomCrop((new_h, new_w))(x)
        #     return x

        # x = np.array(x).astype(np.float32) / 255.  # [0, 255] -> [0, 1]
        # x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        # if tform_name == 'Random Flip':
        #     x = self.fixed_transforms[0](x)
        # else:
        #     tform_id = self.optional_names.index(tform_name)
        #     tform = self.optional_transforms[tform_id]
        #     x = tform(x)
        #     if isinstance(x, tuple):
        #         x = x[0]
        # x = x.detach().squeeze(0).permute(1, 2, 0).numpy() * 255  # [0, 1] -> [0, 255]
        # return Image.fromarray(x.astype(np.uint8))

        w, h = x.size
        x = x.resize((256, 256), Image.BILINEAR)
        x = np.array(x).astype(np.float32) / 255.  # [0, 255] -> [0, 1]
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        if tform_name == 'Fixed Augment':
            for tform in self.fixed_transforms:
                x = tform(x)
                if isinstance(x, tuple):
                    x = x[0]
        else:
            tform_id = self.optional_names.index(tform_name)
            tform = self.optional_transforms[tform_id]
            x = tform(x)
            if isinstance(x, tuple):
                x = x[0]
        x = x.detach().squeeze(0).permute(1, 2, 0).numpy() * 255  # [0, 1] -> [0, 255]
        x = Image.fromarray(x.astype(np.uint8))
        if (tform_name == 'Random Crop') and (self.crop_mode == 'random_crop'):
            w, h = int(224 / 256 * w), int(224 / 256 * h)
        x = x.resize((w, h), Image.BILINEAR)
        return x


class TransformNetHidden(TransformNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fixed_transforms[-1] = RandomResizedCrop2(size=(0.5,1.0), scale=(0.7, 1.0), ratio=(3.0/4, 4.0/3), cropping_mode='resample')
    
    def forward(self, x, global_step, p=0.9):
        # x: [batch_size, 3, H, W] in range [-1, 1]
        x = x * 0.5 + 0.5  # [-1, 1] -> [0, 1]
        ramp = np.min([(global_step-self.step0.cpu().item()) / self.ramp, 1.])

        # fixed transforms
        for tform in self.fixed_transforms[:-1]:
            x = tform(x, ramp=ramp)
            if isinstance(x, tuple):
                x = x[0]
        
        # optional transforms
        if len(self.optional_transforms) > 0:
            tform_ids = torch.randint(len(self.optional_transforms), (self.n_optional,)).numpy()
            for tform_id in tform_ids:
                tform = self.optional_transforms[tform_id]
                x = tform(x, ramp=ramp)
                if isinstance(x, tuple):
                    x = x[0]
        # apply crop
        x = self.fixed_transforms[-1](x, ramp=ramp)
        if isinstance(x, tuple):
            x = x[0]
        return x * 2 - 1  # [0, 1] -> [-1, 1]


if __name__ == '__main__':
    pass
