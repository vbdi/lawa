#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, torch 
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf

def trainer_settings(config, output_dir):
    ## This function is adapted from RoSteALS paper
    out = {}
    ckptdir = os.path.join(output_dir, 'checkpoints')
    cfgdir = os.path.join(output_dir, 'configs')
    
    resumedir = ''

    pl_config = config.get("lightning", OmegaConf.create())
    # callbacks
    callbacks = {
        'generic': dict(target='models.logger.SetupCallback', 
        params={'resume': resumedir, 'now': '', 'logdir': output_dir, 'ckptdir': ckptdir, 'cfgdir': cfgdir, 'config': config, 'lightning_config': pl_config}),
        
        'cuda': dict(target='models.logger.CUDACallback', params={}),

        'ckpt': dict(target='pytorch_lightning.callbacks.ModelCheckpoint',
        params={'dirpath': ckptdir, 'filename': '{epoch:06}', 'verbose': True, 'save_top_k': -1, 'save_last': True}),
     
    }
    if 'checkpoint' in pl_config.callbacks:
        callbacks['ckpt'] = OmegaConf.merge(callbacks['ckpt'], pl_config.callbacks.checkpoint)

    if 'progress_bar' in pl_config.callbacks:
        callbacks['probar'] = pl_config.callbacks.progress_bar

    if 'image_logger' in pl_config.callbacks:
        callbacks['img_log'] = pl_config.callbacks.image_logger
    

    callbacks = [instantiate_from_config(c) for k, c in callbacks.items()]
    out['callbacks'] = callbacks

    # logger
    logger = dict(target='pytorch_lightning.loggers.TestTubeLogger', params={'name': 'testtube', 'save_dir': output_dir})
    logger = instantiate_from_config(logger)
    out['logger'] = logger
    out['accelerator'] = "gpu"
    out['strategy'] = "ddp"
    out['num_sanity_val_steps'] = 0
    
    return out

def main(args):
    config = OmegaConf.load(args.config)
    
    #### data
    data_config = config.get("data", OmegaConf.create()) 
    data_config.params.batch_size = args.batch_size
    data_config.params.train.params.message_len = args.message_len
    data_config.params.validation.params.message_len = args.message_len
    data = instantiate_from_config(data_config)
    data.setup()
    
    
    #### model
    config.model.params.decoder_config.params.message_len = args.message_len
    if args.learning_rate != 0:
        config.model.learning_rate = args.learning_rate
    model = instantiate_from_config(config.model).cpu()
    if args.checkpoint != '':
        print("Loading model weights from checkpoint!")
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        model.fine_tune_decoder = torch.tensor(False)
        model.ae.encoder.eval()
        for p in model.ae.encoder.parameters():
            p.requires_grad = False
        model.ae.decoder.eval()
        for p in model.ae.decoder.parameters():
            p.requires_grad = False
        del checkpoint
    
    #### trainer
    trainer_kwargs = dict(devices=args.devices, precision=32, max_epochs=args.max_epochs, num_nodes=args.num_nodes)
    temp_config_dict = trainer_settings(config, args.output)
    trainer_kwargs.update(temp_config_dict)
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.logdir = args.output

    ### Train
    trainer.fit(model, data)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', type=str, default='configs/SD14_LaWa.yaml')
    parser.add_argument('-o', '--output', type=str, default='outputs/train_result')
    parser.add_argument('--checkpoint',type=str, default= '', help="If continue from a checkpoint")
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of available nodes for multi-node training')
    parser.add_argument('--gpus', type=int, default=1, help='Number of available gpus on each node')
    parser.add_argument('--devices', type=int, default=[0])
    parser.add_argument('--message_len', type=int, default=48, help='Length of watermark message')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--max_epochs', default=100)
    parser.add_argument('--learning_rate', default=0, help="If set to 0, learning rate is set to the value in the config file")
    
    args = parser.parse_args()
    main(args)