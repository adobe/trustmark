# Copyright 2023 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

from __future__ import absolute_import

import torch
import os
import pathlib
import re
import time
import importlib

from omegaconf import OmegaConf
from .datalayer import DataLayer
from PIL import Image
from torchvision import transforms
import numpy as np
import urllib.request


# for model checking
from hashlib import md5
from mmap import mmap, ACCESS_READ

# Content Autenticity Initiative (CAI) Content Delivery Network
MODEL_REMOTE_HOST = "https://cc-assets.netlify.app/watermarking/trustmark-models/"

MODEL_CHECKSUMS=dict()
MODEL_CHECKSUMS['trustmark_C.yaml']="4ee4a79c091f9263c949bd0cb590eb74"
MODEL_CHECKSUMS['decoder_C.ckpt']="ab3fa5678a86c006bb162e5cc90501d3"
MODEL_CHECKSUMS['encoder_C.ckpt']="c22bd5f675ee2cf2a6b18f3c2cbcc507"
MODEL_CHECKSUMS['trustmark_rm_C.yaml']="8476bcd4092abf302272868f3b4c2e38"
MODEL_CHECKSUMS['trustmark_rm_C.ckpt']="5ca3d651d9cde175433cebdf437e412f"

MODEL_CHECKSUMS['trustmark_Q.yaml']="fe40df84a7feeebfceb7a7678d7e6ec6"
MODEL_CHECKSUMS['decoder_Q.ckpt']="4ced90e9cfe13e3295ad082887fe9187"
MODEL_CHECKSUMS['encoder_Q.ckpt']="700328b8754db934b2f6cb5e5185d81f"
MODEL_CHECKSUMS['trustmark_rm_Q.yaml']="8476bcd4092abf302272868f3b4c2e38"
MODEL_CHECKSUMS['trustmark_rm_Q.ckpt']="760337a5596e665aed2ab5c49aa5284f"

MODEL_CHECKSUMS['trustmark_B.yaml']="fe40df84a7feeebfceb7a7678d7e6ec6"
MODEL_CHECKSUMS['decoder_B.ckpt']="c4aaa4a86e551e6aac7f309331191971"
MODEL_CHECKSUMS['encoder_B.ckpt']="e6ab35b3f2d02f37b418726a2dc0b9c9"
MODEL_CHECKSUMS['trustmark_rm_B.yaml']="0952cd4de245c852840f22d096946db8"
MODEL_CHECKSUMS['trustmark_rm_B.ckpt']="eb4279e0301973112b021b1440363401"


ASPECT_RATIO_LIM = 2.0

class TrustMark():

    class Encoding:
       Undefined=-1
       BCH_SUPER=0
       BCH_3=3
       BCH_4=2
       BCH_5=1

    def __init__(self, use_ECC=True, verbose=True, secret_len=100, device='', model_type='Q', encoding_type=Encoding.BCH_5):
        """ Initializes the TrustMark watermark encoder/decoder/remover module

        Parameters (default listed first)
        ---------------------------------

        use_ECC : bool
            [True] will use BCH error correction on the payload, reducing payload size (default)
            [False] will disable error correction, increasing payload size
        verbose : bool
            [True] will output status messages during use (default)
            [False] will run silent except for error messages
        """

        super(TrustMark, self).__init__()

        if not device:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        if(verbose):
            print('Initializing TrustMark watermarking %s ECC using [%s]' % ('with' if use_ECC else 'without',self.device))

        # the location of three models
        assert model_type in ['C', 'Q', 'B']
        self.model_type = model_type


        locations={'config' : os.path.join(pathlib.Path(__file__).parent.resolve(),f'models/trustmark_{self.model_type}.yaml'), 
                   'config-rm' : os.path.join(pathlib.Path(__file__).parent.resolve(),f'models/trustmark_rm_{self.model_type}.yaml'), 
                   'decoder': os.path.join(pathlib.Path(__file__).parent.resolve(),f'models/decoder_{self.model_type}.ckpt'), 
                   'remover': os.path.join(pathlib.Path(__file__).parent.resolve(),f'models/trustmark_rm_{self.model_type}.ckpt'),
                   'encoder': os.path.join(pathlib.Path(__file__).parent.resolve(),f'models/encoder_{self.model_type}.ckpt')}

        self.use_ECC=use_ECC
        self.secret_len=secret_len
        self.ecc = DataLayer(secret_len, verbose=verbose, encoding_mode=encoding_type)
        self.enctyp=encoding_type
        
        self.decoder = self.load_model(locations['config'], locations['decoder'], self.device, secret_len, part='decoder')
        self.encoder = self.load_model(locations['config'], locations['encoder'], self.device, secret_len, part='encoder')
        self.removal = self.load_model(locations['config-rm'], locations['remover'], self.device, secret_len, part='remover')


    def schemaCapacity(self):
        if self.use_ECC:
            return self.ecc.schemaCapacity(self.enctyp)
        else:
            return self.secret_len

    def check_and_download(self, filename):
        valid=False
        if os.path.isfile(filename) and os.path.getsize(filename)>0:
            with open(filename) as file, mmap(file.fileno(), 0, access=ACCESS_READ) as file:
#                print(filename+'-> '+md5(file).hexdigest())
                 valid= (MODEL_CHECKSUMS[pathlib.Path(filename).name]==md5(file).hexdigest())

        if not valid:
            print('Fetching model file (once only): '+filename)
            urld=MODEL_REMOTE_HOST+os.path.basename(filename)
 
            urllib.request.urlretrieve(urld, filename=filename)

    def load_model(self, config_path, weight_path, device, secret_len, part='all'):
        assert part in ['all', 'encoder', 'decoder', 'remover']
        self.check_and_download(config_path)
        self.check_and_download(weight_path)
        config = OmegaConf.load(config_path).model
        if part == 'encoder':
            # replace all other components with identity
            config.params.secret_decoder_config.target = 'trustmark.model.Identity'
            config.params.discriminator_config.target = 'trustmark.model.Identity'
            config.params.loss_config.target = 'trustmark.model.Identity'
            config.params.noise_config.target = 'trustmark.model.Identity'
        elif part == 'decoder':
            # replace all other components with identity
            config.params.secret_encoder_config.target = 'trustmark.model.Identity'
            config.params.discriminator_config.target = 'trustmark.model.Identity'
            config.params.loss_config.target = 'trustmark.model.Identity'
            config.params.noise_config.target = 'trustmark.model.Identity'

        elif part == 'remover':
            config.params.is_train = False  # inference mode, only load denoise module
    
        model = instantiate_from_config(config)
        state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
        
        if 'global_step' in state_dict:
            print(f'Global step: {state_dict["global_step"]}, epoch: {state_dict["epoch"]}')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            
        misses, ignores = model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()

        return model

    def get_the_image_for_processing(self, in_image):

        # get the aspect ratio
        width, height = in_image.size
        if width>height:
            aspect_ratio = width/height
        else:
            aspect_ratio = height/width

        out_im = in_image.copy()
        if aspect_ratio>ASPECT_RATIO_LIM:
            # crop the image in the center
            size = min([width, height])
            left = (width - size) // 2
            top = (height - size) // 2
            right = (width + size) // 2
            bottom = (height + size) // 2
            out_im = out_im.crop((left, top, right, bottom))

        return out_im


    def put_the_image_after_processing(self, wm_image, cover_im):

        # get the aspect ratio
        height, width, _ = cover_im.shape
        if width>height:
             aspect_ratio = width/height
        else:
             aspect_ratio = height/width

        out_im = wm_image.copy()
        if aspect_ratio>ASPECT_RATIO_LIM:
            # crop the image in the center
            size = min([width, height])
            left = (width - size) // 2
            top = (height - size) // 2
            right = (width + size) // 2
            bottom = (height + size) // 2

            out_im = cover_im.copy()
            out_im[top:bottom, left:right, :] = wm_image.copy()

        return out_im


    def decode(self, in_stego_image, MODE='text'):
        # Inputs
        # stego_image: PIL image
        # Outputs: secret numpy array (1, secret_len)
        stego_image = self.get_the_image_for_processing(in_stego_image)
        if min(stego_image.size) > 256:
            stego_image = stego_image.resize((256,256), Image.BILINEAR)
        stego = transforms.ToTensor()(stego_image).unsqueeze(0).to(self.decoder.device) * 2.0 - 1.0 # (1,3,256,256) in range [-1, 1]
        with torch.no_grad():
            secret_binaryarray = (self.decoder.decoder(stego) > 0).cpu().numpy()  # (1, secret_len)
        if self.use_ECC:
            secret_pred, detected, version = self.ecc.decode_bitstream(secret_binaryarray, MODE)[0]
            if not detected:
                # last ditch attempt to recover a possible corruption of the version bits by trying all other schema types
                modeset= [x for x in range(0,3) if x not in [version]] # not bch_3   
                for m in modeset:
                     if m==0:
                        secret_binaryarray[0][-2]=False
                        secret_binaryarray[0][-1]=False
                     if m==1:
                        secret_binaryarray[0][-2]=False
                        secret_binaryarray[0][-1]=True
                     if m==2:
                        secret_binaryarray[0][-2]=True
                        secret_binaryarray[0][-1]=False
                     if m==3: 
                        secret_binaryarray[0][-2]=True  
                        secret_binaryarray[0][-1]=True
                     secret_pred, detected, version = self.ecc.decode_bitstream(secret_binaryarray, MODE)[0]
                     if (detected):
                          return secret_pred, detected, version   
                     else:
                          return '', False, -1
            else:
                return secret_pred, detected, version
        else:
            assert len(secret_binaryarray.shape)==2
            secret_pred = ''.join(str(int(x)) for x in secret_binaryarray[0])
            return secret_pred, True, -1
         
    def encode(self, in_cover_image, string_secret, MODE='text', WM_STRENGTH=0.95, WM_MERGE='bilinear'):
        # Inputs
        #   cover_image: PIL image
        #   secret_tensor: (1, secret_len)
        # Outputs: stego image (PIL image)
        
        # secrets
        if not self.use_ECC:
            if MODE=="binary":
                secret = [int(x) for x in string_secret]
                secret = np.array(secret, dtype=np.float32)
            else:
                secret = self.ecc.encode_text_ascii(string_secret)  # bytearray
                secret = ''.join(format(x, '08b') for x in secret)
                secret = [int(x) for x in secret]
                secret = np.array(secret, dtype=np.float32)
        else:
            if MODE=="binary":
                secret = self.ecc.encode_binary([string_secret])
            else:
                secret = self.ecc.encode_text([string_secret])
        secret = torch.from_numpy(secret).float().to(self.device)
        
        cover_image = self.get_the_image_for_processing(in_cover_image)
        w, h = cover_image.size
        cover = cover_image.resize((256,256), Image.BILINEAR)
        tic=time.time()
        cover = transforms.ToTensor()(cover).unsqueeze(0).to(self.encoder.device) * 2.0 - 1.0 # (1,3,256,256) in range [-1, 1]
        with torch.no_grad():
            stego, _ = self.encoder(cover, secret)
            residual = stego.clamp(-1, 1) - cover
            residual = torch.nn.functional.interpolate(residual, size=(h, w), mode=WM_MERGE)
            residual = residual.permute(0,2,3,1).cpu().numpy().astype('f4')  # (1,256,256,3)
            stego = np.clip(residual[0]*WM_STRENGTH + np.array(cover_image)/127.5-1., -1, 1)*127.5+127.5  # (256, 256, 3), ndarray, uint8
            stego = self.put_the_image_after_processing(stego, np.asarray(in_cover_image).astype(np.uint8))

        return Image.fromarray(stego.astype(np.uint8))

    @torch.no_grad()
    def remove_watermark(self, stego):
        """Remove watermark from stego image"""
        W, H = stego.size
        stego256 = stego.resize((256,256), Image.BILINEAR)
        stego256 = transforms.ToTensor()(stego256).unsqueeze(0).to(self.removal.device) * 2.0 - 1.0 # (1,3,256,256) in range [-1, 1]
        img256 = self.removal(stego256).clamp(-1, 1)
        res = img256 - stego256
        res = torch.nn.functional.interpolate(res, (H,W), mode='bilinear').permute(0,2,3,1).cpu().numpy()   # (B,3,H,W) no need antialias since this op is mostly upsampling
        out = np.clip(res[0] + np.asarray(stego)/127.5-1., -1, 1)*127.5+127.5  # (256, 256, 3), ndarray, uint8
        return Image.fromarray(out.astype(np.uint8))



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
  


