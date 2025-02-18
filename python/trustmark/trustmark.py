# Copyright 2023 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

from __future__ import absolute_import

import torch
import os
import pathlib
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

# C variant is a compact version of TrustMark, using a ResNet-18 decoder
# This is convenient for resource constrained deployments but lowers the PSNR to 39 typical
MODEL_CHECKSUMS['trustmark_C.yaml']="4ee4a79c091f9263c949bd0cb590eb74"
MODEL_CHECKSUMS['decoder_C.ckpt']="ab3fa5678a86c006bb162e5cc90501d3"
MODEL_CHECKSUMS['encoder_C.ckpt']="c22bd5f675ee2cf2a6b18f3c2cbcc507"
MODEL_CHECKSUMS['trustmark_rm_C.yaml']="8476bcd4092abf302272868f3b4c2e38"
MODEL_CHECKSUMS['trustmark_rm_C.ckpt']="5ca3d651d9cde175433cebdf437e412f"

# Q variant was published with the original 2023 paper and strikes a good robustness/quality tradeoff
# using a ResNet50 backbone.  Q yields typical PSNR around 43 and most people use this as default.
MODEL_CHECKSUMS['trustmark_Q.yaml']="fe40df84a7feeebfceb7a7678d7e6ec6"
MODEL_CHECKSUMS['decoder_Q.ckpt']="4ced90e9cfe13e3295ad082887fe9187"
MODEL_CHECKSUMS['encoder_Q.ckpt']="700328b8754db934b2f6cb5e5185d81f"
MODEL_CHECKSUMS['trustmark_rm_Q.yaml']="8476bcd4092abf302272868f3b4c2e38"
MODEL_CHECKSUMS['trustmark_rm_Q.ckpt']="760337a5596e665aed2ab5c49aa5284f"

# B variant is very similar to Q variant, but had slightly higher robustness vs. quality it is included
# here as presented in the original 2023 paper for purposes of reproducing the results
MODEL_CHECKSUMS['trustmark_B.yaml']="fe40df84a7feeebfceb7a7678d7e6ec6"
MODEL_CHECKSUMS['decoder_B.ckpt']="c4aaa4a86e551e6aac7f309331191971"
MODEL_CHECKSUMS['encoder_B.ckpt']="e6ab35b3f2d02f37b418726a2dc0b9c9"
MODEL_CHECKSUMS['trustmark_rm_B.yaml']="0952cd4de245c852840f22d096946db8"
MODEL_CHECKSUMS['trustmark_rm_B.ckpt']="eb4279e0301973112b021b1440363401"

# P variant is trained with higher weight on perceptual loss over diverse data and is the highest visual
# quality variant of TrustMark whilst still retaining good robustness, PSNR typically 48
MODEL_CHECKSUMS['trustmark_P.yaml']="fe40df84a7feeebfceb7a7678d7e6ec6"
MODEL_CHECKSUMS['decoder_P.ckpt']="9450972bc0c3c217cb7b8220dd2f7a3c"
MODEL_CHECKSUMS['encoder_P.ckpt']="0a18f6de6d57c6ef7dda30ce6154a775"
MODEL_CHECKSUMS['trustmark_rm_P.yaml']="654cabd3ac8339397fbc611ca7464780"
MODEL_CHECKSUMS['trustmark_rm_P.ckpt']="8b8f4715ea474327921ee9d0f46d2c3f"


CONCENTRATE_WM_REGION = 1.0
ASPECT_RATIO_LIM = 2.0
FALLBACK_ALL_SCHEMAS = True
FEATHERING_RESIDUAL=0.01

class TrustMark():

    class Encoding:
       Undefined=-1
       BCH_SUPER=0
       BCH_3=3
       BCH_4=2
       BCH_5=1

    def __init__(self, use_ECC=True, verbose=True, secret_len=100, device='', model_type='Q', encoding_type=Encoding.BCH_5, concentrate_wm_region=CONCENTRATE_WM_REGION):
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
        assert model_type in ['C', 'Q', 'B', 'P']
        self.model_type = model_type


        locations={'config' : os.path.join(pathlib.Path(__file__).parent.resolve(),f'models/trustmark_{self.model_type}.yaml'), 
                   'config-rm' : os.path.join(pathlib.Path(__file__).parent.resolve(),f'models/trustmark_rm_{self.model_type}.yaml'), 
                   'decoder': os.path.join(pathlib.Path(__file__).parent.resolve(),f'models/decoder_{self.model_type}.ckpt'), 
                   'remover': os.path.join(pathlib.Path(__file__).parent.resolve(),f'models/trustmark_rm_{self.model_type}.ckpt'),
                   'encoder': os.path.join(pathlib.Path(__file__).parent.resolve(),f'models/encoder_{self.model_type}.ckpt')}

        self.use_ECC = use_ECC
        self.secret_len = secret_len
        self.ecc = DataLayer(secret_len, verbose=verbose, encoding_mode=encoding_type)
        self.enctyp = encoding_type
        self.aspect_ratio_lim=ASPECT_RATIO_LIM
        self.concentrate_wm_region=concentrate_wm_region

        if model_type=='P':
           self.model_resolution_enc = 256
           self.model_resolution_dec = 224
           self.aspect_ratio_lim=0 # always force to centre square crop
        else:
           self.model_resolution_enc = 256
           self.model_resolution_dec = 245
        self.model_resolution_remove = 256
        
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
        scale=self.concentrate_wm_region
        width, height = in_image.size
        
        # Compute aspect ratio (≥ 1.0)
        if width > height:
            aspect_ratio = width / height
        else:
            aspect_ratio = height / width
        
        # Make a copy of the image (PIL)
        out_im = in_image.copy()

        if (aspect_ratio > self.aspect_ratio_lim):
            # We do a center-square approach, but scaled
            square_size = min(width, height)  # largest possible square dimension
            scaled_size = int(square_size * scale)  # scale that dimension

            # Compute bounding box
            left   = (width  - scaled_size) // 2
            top    = (height - scaled_size) // 2
            right  = left + scaled_size
            bottom = top  + scaled_size
            
            out_im = out_im.crop((left, top, right, bottom))
        
        else:
            # The aspect ratio is normal, so we consider
            # the *entire* image dimension. Then scale that region
            scaled_w = int(width  * scale)
            scaled_h = int(height * scale)
            
            # Center the smaller (or bigger) rectangle
            left   = (width  - scaled_w) // 2
            top    = (height - scaled_h) // 2
            right  = left + scaled_w
            bottom = top  + scaled_h
            
            out_im = out_im.crop((left, top, right, bottom))

        return out_im


    def put_the_image_after_processing(self, wm_image, cover_im, feather=True):

        scale = self.concentrate_wm_region
        cover_h, cover_w, _ = cover_im.shape

        if cover_w > cover_h:
            aspect_ratio = cover_w / cover_h
        else:
            aspect_ratio = cover_h / cover_w

        out_im = cover_im.copy()

        if (aspect_ratio > self.aspect_ratio_lim):
            # Square region, scaled
            square_size = min(cover_w, cover_h)
            scaled_size = int(square_size * scale)

            left   = (cover_w - scaled_size) // 2
            top    = (cover_h - scaled_size) // 2
            right  = left + scaled_size
            bottom = top  + scaled_size

            region_w = scaled_size
            region_h = scaled_size

        else:
            # Normal ratio, scaled
            scaled_w = int(cover_w * scale)
            scaled_h = int(cover_h * scale)
            left   = (cover_w - scaled_w) // 2
            top    = (cover_h - scaled_h) // 2
            right  = left + scaled_w
            bottom = top  + scaled_h

            region_w = scaled_w
            region_h = scaled_h

        if feather:

            feather_size = int(min(region_w, region_h) * FEATHERING_RESIDUAL)
        
            feather_size = max(1, feather_size)      
            feather_size = min(feather_size, 50)

            self.feather_paste(
                out_im,       # destination (modified in-place)
                cover_im,     # original for reference
                wm_image,     # watermark patch
                top, bottom, left, right,
                feather_size=feather_size
            )
        else:
            out_im[top:bottom, left:right, :] = wm_image

        return out_im

    def feather_paste(self,
        out_im: np.ndarray,     # Output image (modified in-place)
        cover_im: np.ndarray,   # Original cover image (same shape)
        wm_image: np.ndarray,   # Watermarked patch to paste
        top: int, bottom: int,
        left: int, right: int,
        feather_size: int = 9):
    
        out_im[top:bottom, left:right, :] = wm_image
        alpha_vals = [ (i+1) / feather_size for i in range(feather_size) ]
    
        feather_size = min(feather_size, (bottom - top), (right - left))
    
        for i in range(feather_size):
            alpha = alpha_vals[i]
            row = top + i
            # Blend that entire row from left..right
            out_im[row, left:right, :] = (
                alpha * wm_image[i, :, :] +
                (1.0 - alpha) * cover_im[row, left:right, :]
            )
        
        for i in range(feather_size):
            alpha = alpha_vals[i]
            row = bottom - 1 - i
            wm_row = (bottom - top - 1) - i
            out_im[row, left:right, :] = (
                alpha * wm_image[wm_row, :, :] +
                (1.0 - alpha) * cover_im[row, left:right, :]
            )
    
        for i in range(feather_size):
            alpha = alpha_vals[i]
            col = left + i
            out_im[top:bottom, col, :] = (
                alpha * wm_image[:, i, :] +
                (1.0 - alpha) * cover_im[top:bottom, col, :]
            )
    
        for i in range(feather_size):
            alpha = alpha_vals[i]
            col = right - 1 - i
            wm_col = (right - left - 1) - i
            out_im[top:bottom, col, :] = (
                alpha * wm_image[:, wm_col, :] +
                (1.0 - alpha) * cover_im[top:bottom, col, :]
            )
    


    def decode(self, in_stego_image, MODE='text'):
        # Inputs
        # stego_image: PIL image
        # Outputs: secret numpy array (1, secret_len)
        stego_image = self.get_the_image_for_processing(in_stego_image)
        stego_image = stego_image.resize((self.model_resolution_dec,self.model_resolution_dec), Image.BILINEAR)
        stego = transforms.ToTensor()(stego_image).unsqueeze(0).to(self.decoder.device) * 2.0 - 1.0 # (1,3,modelres,modelres) in range [-1, 1]
        with torch.no_grad():
            secret_binaryarray = (self.decoder.decoder(stego) > 0).cpu().numpy()  # (1, secret_len)
        if self.use_ECC:
            secret_pred, detected, version = self.ecc.decode_bitstream(secret_binaryarray, MODE)[0]
            if not detected and FALLBACK_ALL_SCHEMAS:
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
         
    def encode(self, in_cover_image, string_secret, MODE='text', WM_STRENGTH=1.0, WM_MERGE='bilinear'):
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
        if self.model_type == 'P':
            WM_STRENGTH = WM_STRENGTH * 1.25
        secret = torch.from_numpy(secret).float().to(self.device)
        
        cover_image = self.get_the_image_for_processing(in_cover_image)
        w, h = cover_image.size
        cover = cover_image.resize((self.model_resolution_enc,self.model_resolution_enc), Image.BILINEAR)
        tic=time.time()
        cover = transforms.ToTensor()(cover).unsqueeze(0).to(self.encoder.device) * 2.0 - 1.0 # (1,3,modelres,modelres) in range [-1, 1]
        with torch.no_grad():
            stego, _ = self.encoder(cover, secret)
            residual = stego.clamp(-1, 1) - cover

            residual_mean_c = residual.mean(dim=(2,3), keepdim=True)  # remove color shifts per channel
            residual = residual - residual_mean_c

            residual = torch.nn.functional.interpolate(residual, size=(h, w), mode=WM_MERGE)
            residual = residual.permute(0,2,3,1).cpu().numpy().astype('f4')  # (1,modelres,modelres,3)
            stego = np.clip(residual[0]*WM_STRENGTH + np.array(cover_image)/127.5-1., -1, 1)*127.5+127.5  # (modelres, modelres, 3), ndarray, uint8
            stego = self.put_the_image_after_processing(stego, np.asarray(in_cover_image).astype(np.uint8))

        return Image.fromarray(stego.astype(np.uint8))

    @torch.no_grad()
    def remove_watermark(self, in_cover_image, WM_STRENGTH=1.0, WM_MERGE='bilinear'):
        """Remove watermark from stego image"""
        stego = self.get_the_image_for_processing(in_cover_image)
        W, H = stego.size
        if self.model_type == 'P':
            WM_STRENGTH = WM_STRENGTH * 1.25
        stego256 = stego.resize((self.model_resolution_remove,self.model_resolution_remove), Image.BILINEAR)
        stego256 = transforms.ToTensor()(stego256).unsqueeze(0).to(self.removal.device) * 2.0 - 1.0 # (1,3,modelres,modelres) in range [-1, 1]
        img256 = self.removal(stego256).clamp(-1, 1)
        res = img256 - stego256
        res = torch.nn.functional.interpolate(res, (H,W), mode=WM_MERGE).permute(0,2,3,1).cpu().numpy()   # (B,3,H,W) no need antialias since this op is mostly upsampling
        out = np.clip(res[0]*WM_STRENGTH + np.asarray(stego)/127.5-1., -1, 1)*127.5+127.5  # (modelres, modelres, 3), ndarray, uint8
        stego = self.put_the_image_after_processing(out, np.asarray(in_cover_image).astype(np.uint8))
        return Image.fromarray(stego.astype(np.uint8))



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
  


