# Copyright 2022 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import os, sys, torch 
import argparse
from pathlib import Path


def main(args):
    model_name = Path(args.weight).name
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    state_dict = torch.load(args.weight, map_location=torch.device('cpu'))
    if 'global_step' in state_dict:
        print(f'Global step: {state_dict["global_step"]}, epoch: {state_dict["epoch"]}')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    # get encoder
    encoder_dict = {key: val for key, val in state_dict.items() if key.startswith('encoder.')}
    # get decoder
    decoder_dict = {key: val for key, val in state_dict.items() if key.startswith('decoder.')}
    # get watermark removal
    removal_dict = {key: val for key, val in state_dict.items() if key.startswith('denoise.')}
    # save
    if len(encoder_dict) > 0:
        torch.save(encoder_dict, outdir/ f'encoder_{model_name}')
    if len(decoder_dict) > 0:
        torch.save(decoder_dict, outdir/ f'decoder_{model_name}')
    if len(removal_dict) > 0:
        torch.save(removal_dict, outdir/ f'removal_{model_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight', default='models/epoch=000079-step=000500000.ckpt')
    parser.add_argument('-o', '--output', default='models/')
    args = parser.parse_args()
    main(args)
