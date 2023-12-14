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
import random
import numpy as np 
from PIL import Image 
from imagenet_c import corrupt, corruption_dict


class IdentityAugment(object):
    def __call__(self, x):
        return x 

    def __repr__(self):
        s = f'()'
        return self.__class__.__name__ + s

class RandomImagenetC(object):
    # transform id 5 (motion blur) and 7 (snow) requires WandImage which is not fork-safe, while id 4 (glass blur) and 6 (zoom blur) are super slow thus we move it to validation (unseen), 12 (elastic transform) is non realistic
    methods = {'train': np.array([0,1,2,3,8,9,10,11,13,14,15, 16, 17, 18]),#np.arange(15),
               'val': np.array([4, 5, 6, 7, 12]),
               'test': np.array([0,1,2,3,8,9,10,11,13,14,15, 16, 17, 18])
    }
    method_names = list(corruption_dict.keys())
    def __init__(self, min_severity=1, max_severity=5, phase='all', p=1.0,n=19):
        assert phase in ['train', 'val', 'test', 'all'], ValueError(f'{phase} not recognised. Must be one of [train, val, all]')
        if phase == 'all':
            self.corrupt_ids = np.concatenate(list(self.methods.values()))
        else:
            self.corrupt_ids = self.methods[phase]
        self.corrupt_ids = self.corrupt_ids[:n]  # first n tforms
        self.phase = phase
        self.severity = np.arange(min_severity, max_severity+1)
        self.p = p  # probability to apply a transformation

    def __call__(self, x, corrupt_id=None, corrupt_strength=None):
        # input: x PIL image
        if corrupt_id is None:
            if len(self.corrupt_ids)==0:  # do nothing
                return x
            corrupt_id = np.random.choice(self.corrupt_ids)
        else:
            assert corrupt_id in range(19)

        severity = np.random.choice(self.severity) if corrupt_strength is None else corrupt_strength
        assert severity in self.severity, f'Error! Corrupt strength {severity} isnt supported.'
        
        if np.random.rand() < self.p:
            org_size = x.size 
            x = np.asarray(x.convert('RGB').resize((224, 224), Image.BILINEAR))[:,:,::-1]
            x = corrupt(x, severity, corruption_number=corrupt_id)
            x = Image.fromarray(x[:,:,::-1])
            if x.size != org_size:
                x = x.resize(org_size, Image.BILINEAR)
        return x 

    def transform_with_fixed_severity(self, x, severity, corrupt_id=None):
        if corrupt_id is None:
            corrupt_id = np.random.choice(self.corrupt_ids)
        else:
            assert corrupt_id in self.corrupt_ids
        assert severity > 0 and severity < 6
        org_size = x.size 
        x = np.asarray(x.convert('RGB').resize((224, 224), Image.BILINEAR))[:,:,::-1]
        x = corrupt(x, severity, corruption_number=corrupt_id)
        x = Image.fromarray(x[:,:,::-1])
        if x.size != org_size:
            x = x.resize(org_size, Image.BILINEAR)
        return x

    def __repr__(self):
        s = f'(severity={self.severity}, phase={self.phase}, p={self.p},ids={self.corrupt_ids})'
        return self.__class__.__name__ + s


class NoiseResidual(object):
    def __init__(self, k=16):
        self.k = k 
    def __call__(self, x):
        h, w = x.height, x.width
        x1 = x.resize((w//self.k,h//self.k), Image.BILINEAR).resize((w, h), Image.BILINEAR)
        x1 = np.abs(np.array(x).astype(np.float32) - np.array(x1).astype(np.float32))
        x1 = (x1 - x1.min())/(x1.max() - x1.min() + np.finfo(np.float32).eps)
        x1 = Image.fromarray((x1*255).astype(np.uint8))
        return x1
    def __repr__(self):
        s = f'(k={self.k}'
        return self.__class__.__name__ + s


def get_transforms(img_mean=[0.5, 0.5, 0.5], img_std=[0.5, 0.5, 0.5], rsize=256, csize=224, pertubation=True, dct=False, residual=False, max_c=19):
    from torchvision import transforms
    prep = transforms.Compose([
            transforms.Resize(rsize),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(csize)])
    if pertubation:
        pertubation_train = RandomImagenetC(max_severity=5, phase='train', p=0.95,n=max_c)
        pertubation_val = RandomImagenetC(max_severity=5, phase='train', p=1.0,n=max_c)
        pertubation_test = RandomImagenetC(max_severity=5, phase='val', p=1.0,n=max_c)
    else:
        pertubation_train = pertubation_val = pertubation_test = IdentityAugment()
    if dct:
        from .image_tools import DCT 
        norm = [
                DCT(),
                transforms.ToTensor(),
                transforms.Normalize(mean=img_mean, std=img_std)]
    else:
        norm = [
                transforms.ToTensor(),
                transforms.Normalize(mean=img_mean, std=img_std)]
    if residual:
        norm.insert(0, NoiseResidual())

    preprocess = {
        'train': [prep, pertubation_train, transforms.Compose(norm)],

        'val': [prep, pertubation_val, transforms.Compose(norm)],

        'test_unseen': [prep, pertubation_test, transforms.Compose(norm)],

        'clean': transforms.Compose([transforms.Resize(csize)] + norm)
        }
    return preprocess


# ## example
# from PIL import Image 
# import numpy as np 
# import time
# from imagenet_c import corrupt, corruption_dict
# im = Image.open('/vol/research/tubui1/projects/gan_prov/gan_models/stargan2/test.jpg').convert('RGB').resize((224,224), Image.BILINEAR)
# im.save('original.jpg')
# im = np.array(im)[:,:,::-1]  # BRG
# t = np.zeros(19)
# for i, key in enumerate(corruption_dict.keys()):
#     begin = time.time()
#     for j in range(10):
#         out = corrupt(im, 5, corruption_number=i)
#     end = time.time()
#     t[i] = end-begin
#     # Image.fromarray(out[:,:,::-1]).save(f'imc_{key}.jpg')
#     print(f'{i} - {key}: {end-begin}')

# for i,k in enumerate(corruption_dict.keys()):
#     print(i, k, t[i])
