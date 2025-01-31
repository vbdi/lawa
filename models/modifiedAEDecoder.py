import numpy as np
import einops
import torch
import torch.nn as nn
# from torch.nn import functional as thf
import pytorch_lightning as pl
import torchvision
from ldm.modules.diffusionmodules.util import (
    zero_module,
)
import copy
# from contextlib import contextmanager
# from torchvision.utils import make_grid
# from ldm.modules.attention import SpatialTransformer
# from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
# from ldm.models.diffusion.ddpm import LatentDiffusion
# from ldm.util import log_txt_as_img, exists, instantiate_from_config, default
from ldm.util import instantiate_from_config
# from ldm.models.diffusion.ddim import DDIMSampler
# from ldm.modules.ema import LitEma
# from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
# from ldm.modules.diffusionmodules.model import Encoder
import lpips
import torch.nn.functional as F
# from torch.autograd.variable import Variable

from torchvision import transforms
# from torchvision import models
import random
from lpips.lpips import LPIPS

from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class LaWa(pl.LightningModule):
    def __init__(self,
                 first_stage_config,
                 decoder_config,
                 discriminator_config,
                 recon_type,
                 learning_rate = 0.0001,
                 epoch_num = 100,
                 recon_loss_weight = 2.0,
                 adversarial_loss_weight = 2.0,
                 perceptual_loss_weight = 0.0,
                 lpips_loss_weights_path = None,
                 message_absolute_loss_weight = None,
                 ramp= 100000,
                 watermark_addition_weight = 0.1,
                 noise_config='__none__',
                 use_ema=False,
                 scale_factor=1.,
                 ckpt_path="__none__",
                 extraction_resize = False,
                 addition_network_config = None,
                 start_attack_acc_thresh = 0.85,
                 dis_update_freq = 1,
                 clamp_during_training = False, 
                 noise_block_size = 8,
                 ):
        super().__init__()
        self.learning_rate = learning_rate
        self.epoch_num = epoch_num
        self.scale_factor = scale_factor

        self.extraction_resize = extraction_resize
        self.ae = instantiate_from_config(first_stage_config)
        self.decoder = instantiate_from_config(decoder_config)
        self.discriminator = instantiate_from_config(discriminator_config)
        if addition_network_config != None:
            self.addition_net = instantiate_from_config(addition_network_config)
        else:
            self.addition_net = None

        # self.decoder_latent = instantiate_from_config(decoder_latent_config)
        if noise_config != '__none__':
            print('Using noise')
            self.noise = instantiate_from_config(noise_config)
        # copy weights from first stage
        # freeze first stage
        
        self.ae.eval()
        self.ae.train = disabled_train
        for p in self.ae.parameters():
            p.requires_grad = False
        

        self.watermark_addition_weight = watermark_addition_weight

        # early training phase
        self.message_len = decoder_config.params.message_len
        self.fixed_x = None
        self.fixed_img = None
        self.fixed_input_recon = None
        self.fixed_control = None
        self.register_buffer("fixed_input", torch.tensor(False))
        self.register_buffer("noise_activated", torch.tensor(False))
        self.noise_active_step = 0
        self.ramp_step = 1e9
        self.start_attack_acc_thresh = start_attack_acc_thresh
        self.dis_update_freq = dis_update_freq 
        self.tanh_activation = nn.Tanh()

        self.use_ema = use_ema

        if ckpt_path != '__none__':
            print("###############################################################################")
            print("Using provided model weights!")
            self.init_from_ckpt(ckpt_path, ignore_keys=[])
        
        #### image normalization parameters:
        self.normalize_vqgan_to_imagenet = transforms.Compose([transforms.Normalize(mean=[-1, -1, -1], std=[1/0.5, 1/0.5, 1/0.5]),
                                                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        #### loss parameters:
        self.recon_type = recon_type
        if self.recon_type == 'yuv':
            self.register_buffer('yuv_scales', torch.tensor([1,100,100]).unsqueeze(1).float())  # [3,1]

        # elif self.recon_type == 'watson_vgg':
        #     provider = LossProvider()
        #     self.loss_w_vgg = provider.get_loss_function('Watson-VGG', colorspace='RGB', pretrained=True, reduction='sum')
        # provider = LossProvider()
        # self.loss_w_vgg = provider.get_loss_function('Watson-VGG', colorspace='RGB', pretrained=True, reduction='sum')
            
        self.recon_weight = recon_loss_weight
        self.adversarial_loss_weight = adversarial_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight
        if lpips_loss_weights_path != None:
            self.perceptual_loss = LPIPS(weights_path= lpips_loss_weights_path)
            self.perceptual_loss.eval()
        elif self.perceptual_loss_weight > 0 and lpips_loss_weights_path == None:
            self.perceptual_loss = LPIPS()
            self.perceptual_loss.eval()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.adversarial_loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.message_absolute_loss_weight = message_absolute_loss_weight


        ### watermark layers:
        # watermark
        self.noise_block_size = noise_block_size
        self.watermark_initial_0 = nn.Sequential(
            nn.Linear(self.message_len, 4 * self.noise_block_size * self.noise_block_size),
            nn.SiLU(),
            View(-1, 4 , self.noise_block_size, self.noise_block_size),
        )
        for p in self.watermark_initial_0.parameters():
            p.requires_grad = True
        self.watermark_initial_0_conv =  zero_module(torch.nn.Conv2d(4, 4, 3, padding=1))
        for p in self.watermark_initial_0_conv.parameters():
            p.requires_grad = True

        
        self.watermark_initial = nn.Sequential(
            nn.Linear(self.message_len, 512 * self.noise_block_size * self.noise_block_size),
            nn.SiLU(),
            View(-1, 512 , self.noise_block_size, self.noise_block_size),
        )
        for p in self.watermark_initial.parameters():
            p.requires_grad = True
        self.watermark_initial_conv = zero_module(torch.nn.Conv2d(512,512,3,padding=1))
        for p in self.watermark_initial_conv.parameters():
            p.requires_grad = True
        
        self.watermark = nn.ModuleList()
        self.watermark_conv = nn.ModuleList()
        layer_4 = nn.Sequential(
            nn.Linear(self.message_len, 128 * self.noise_block_size * self.noise_block_size),
            nn.SiLU(),
            View(-1, 128 , self.noise_block_size, self.noise_block_size),
        )
        for p in layer_4.parameters():
            p.requires_grad = True
        self.watermark.append(layer_4)
        layer_4_conv = zero_module(torch.nn.Conv2d(128,128,3,padding=1))
        for p in layer_4_conv.parameters():
            p.requires_grad = True
        self.watermark_conv.append(layer_4_conv)
        layer_3 = nn.Sequential(
            nn.Linear(self.message_len, 256 * self.noise_block_size * self.noise_block_size),
            nn.SiLU(),
            View(-1, 256 , self.noise_block_size, self.noise_block_size),
        )
        for p in layer_3.parameters():
            p.requires_grad = True
        self.watermark.append(layer_3)
        layer_3_conv = zero_module(torch.nn.Conv2d(256,256,3,padding=1))
        for p in layer_3_conv.parameters():
            p.requires_grad = True
        self.watermark_conv.append(layer_3_conv)
        layer_2 = nn.Sequential(
            nn.Linear(self.message_len, 512 * self.noise_block_size * self.noise_block_size),
            nn.SiLU(),
            View(-1, 512 , self.noise_block_size, self.noise_block_size),
        )
        for p in layer_2.parameters():
            p.requires_grad = True
        self.watermark.append(layer_2)
        layer_2_conv = zero_module(torch.nn.Conv2d(512,512,3,padding=1))
        for p in layer_2_conv.parameters():
            p.requires_grad = True
        self.watermark_conv.append(layer_2_conv)
        layer_1 = nn.Sequential(
            nn.Linear(self.message_len, 512 * self.noise_block_size * self.noise_block_size),
            nn.SiLU(),
            View(-1, 512 , self.noise_block_size, self.noise_block_size),
        )
        for p in layer_1.parameters():
            p.requires_grad = True
        self.watermark.append(layer_1)
        layer_1_conv = zero_module(torch.nn.Conv2d(512,512,3,padding=1))
        for p in layer_1_conv.parameters():
            p.requires_grad = True
        self.watermark_conv.append(layer_1_conv)

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

   
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.control_ema(self.control)
            self.decoder_ema(self.decoder)

    def forward(self, x, image, c):
        
        H = x.shape[2]
        W = x.shape[3]
        dx = self.watermark_initial_0(c)
        dx = dx.repeat(1,1,int(H/self.noise_block_size),int(W/self.noise_block_size))
        dx = self.watermark_initial_0_conv(dx)
        x_new = x + dx
          
        #assert z.shape[1:] == self.z_shape[1:]
        self.ae.decoder.last_z_shape = x_new.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.ae.decoder.conv_in(x_new)

        # middle
        h = self.ae.decoder.mid.block_1(h, temb)
        h = self.ae.decoder.mid.attn_1(h)
        h = self.ae.decoder.mid.block_2(h, temb)

        H = h.shape[2]
        W = h.shape[3]
        dh = self.watermark_initial(c)
        dh = dh.repeat(1,1,int(H/self.noise_block_size),int(W/self.noise_block_size))
        dh = self.watermark_initial_conv(dh)
        h1 = h + dh

        # upsampling
        i_level = 3
        for i_block in range(self.ae.decoder.num_res_blocks+1):
            h1 = self.ae.decoder.up[i_level].block[i_block](h1, temb)
            if len(self.ae.decoder.up[i_level].attn) > 0:
                h1 = self.ae.decoder.up[i_level].attn[i_block](h1)
        if i_level != 0:
            h1 = self.ae.decoder.up[i_level].upsample(h1)
        H = h1.shape[2]
        W = h1.shape[3]
        dh1 = self.watermark[i_level](c)
        dh1 = dh1.repeat(1,1,int(H/self.noise_block_size),int(W/self.noise_block_size))
        dh1 = self.watermark_conv[i_level](dh1)
        h2 = h1 + dh1

        i_level = 2
        for i_block in range(self.ae.decoder.num_res_blocks+1):
            h2 = self.ae.decoder.up[i_level].block[i_block](h2, temb)
            if len(self.ae.decoder.up[i_level].attn) > 0:
                h2 = self.ae.decoder.up[i_level].attn[i_block](h2)
        if i_level != 0:
            h2 = self.ae.decoder.up[i_level].upsample(h2)
        H = h2.shape[2]
        W = h2.shape[3]
        dh2 = self.watermark[i_level](c)
        dh2 = dh2.repeat(1,1,int(H/self.noise_block_size),int(W/self.noise_block_size))
        dh2 = self.watermark_conv[i_level](dh2)
        h3 = h2 + dh2

        i_level = 1
        for i_block in range(self.ae.decoder.num_res_blocks+1):
            h3 = self.ae.decoder.up[i_level].block[i_block](h3, temb)
            if len(self.ae.decoder.up[i_level].attn) > 0:
                h3 = self.ae.decoder.up[i_level].attn[i_block](h3)
        if i_level != 0:
            h3 = self.ae.decoder.up[i_level].upsample(h3)
        H = h3.shape[2]
        W = h3.shape[3]
        dh3 = self.watermark[i_level](c)
        dh3 = dh3.repeat(1,1,int(H/self.noise_block_size),int(W/self.noise_block_size))
        dh3 = self.watermark_conv[i_level](dh3)
        h4 = h3 + dh3

        i_level = 0
        for i_block in range(self.ae.decoder.num_res_blocks+1):
            h4 = self.ae.decoder.up[i_level].block[i_block](h4, temb)
            if len(self.ae.decoder.up[i_level].attn) > 0:
                h4 = self.ae.decoder.up[i_level].attn[i_block](h4)
        if i_level != 0:
            h4 = self.ae.decoder.up[i_level].upsample(h4)
        H = h4.shape[2]
        W = h4.shape[3]
        dh4 = self.watermark[i_level](c)
        dh4 = dh4.repeat(1,1,int(H/self.noise_block_size),int(W/self.noise_block_size))
        dh4 = self.watermark_conv[i_level](dh4)
        h5 = h4 + dh4

        # end
        if self.ae.decoder.give_pre_end:
            return h5

        h5 = self.ae.decoder.norm_out(h5)
        h5 = self.nonlinearity(h5)
        h5 = self.ae.decoder.conv_out(h5)
        if self.ae.decoder.tanh_out:
            h5 = torch.tanh(h5)
        
        
        return None, h5 

    @torch.no_grad()
    def get_input(self, batch, bs=None):
        image = batch['image']
        message = batch['message']
        if bs is not None:
            image = image[:bs]
            message = message[:bs]
        else:
            bs = image.shape[0]
        
        image = einops.rearrange(image, "b h w c -> b c h w").contiguous()
        x = self.encode_first_stage(image).detach()
        x = self.ae.post_quant_conv(1./self.scale_factor * x).detach()
        image_rec = self.ae.decoder(x).detach()
        image_rec = torch.clamp(image_rec, min=-1., max=1.)
        
        out = [x, message, image, image_rec]
        return out

    def training_step(self, batch, batch_idx, optimizer_idx):
        # print("Next Step!")
        x, c, img, img_rec_gt = self.get_input(batch)
        real_labels = torch.ones(img_rec_gt.size(0), 1)
        real_labels = real_labels.type_as(x)
        fake_labels = torch.zeros(img_rec_gt.size(0), 1)
        fake_labels = fake_labels.type_as(x)

        # train watermark embedder and extractor:
        if optimizer_idx == 0:
            loss_dict = {}
            
            ###### create the image:
            x_w, image_rec = self(x, img_rec_gt, c)

            #### recreation loss calculation:
            rec_loss = self.compute_recon_loss(img_rec_gt.contiguous(), image_rec.contiguous())

            #### calculate psnr
            psnr = self.calculate_psnr(image_rec, img_rec_gt)

            if self.perceptual_loss_weight > 0:
                p_loss = self.perceptual_loss(img_rec_gt.contiguous() , image_rec.contiguous()).mean()
                loss_dict['emb_p_loss'] = p_loss
            else:
                p_loss = 0 
 
            ####### calculate discriminator loss:
            ## here we give cover image label to the watermarked images:
            d_on_watermarked_image = self.discriminator(image_rec)
            loss_adversarial = (-1) * d_on_watermarked_image.mean()

            
            ####### applying noise:
            if hasattr(self, 'noise') and self.noise_activated:
                image_rec = self.noise(image_rec, self.global_step, active_step = self.noise_active_step, p=0.9)
            
            
            #### extract the watermark
            pred = self.decoder(image_rec)

            #### secret loss calculation:
            secret_loss = self.bce_loss(pred, 0.5*(c+1))
            
            #### apply loss weights: 
            w_emb_loss =  rec_loss * self.recon_weight + self.adversarial_loss_weight * loss_adversarial + \
                              secret_loss * self.message_absolute_loss_weight + self.perceptual_loss_weight * p_loss
            
            ## final loss is average over batch size:
            # w_emb_loss = w_emb_loss
            diff = (~torch.logical_xor(pred>0, 0.5*(c+1)>0)) # b k -> b k
            bit_acc = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
            bit_acc = torch.mean(bit_acc)

            ##### loss dict update
            loss_dict['bit_acc'] = bit_acc
            loss_dict['psnr'] = psnr
            loss_dict['emb_loss'] = w_emb_loss
            loss_dict['emb_rec_loss'] = rec_loss
            loss_dict['emb_secret_loss'] = secret_loss
            loss_dict['emb_adversarial_loss'] = loss_adversarial
            #loss_dict['optimizer_id'] = optimizer_idx

            ### change the dataset for training or start attacking if accuracy has increased
            bit_acc_ = bit_acc.item()
            
            if (bit_acc_ > self.start_attack_acc_thresh) and (not self.noise_activated):
                print(f'Activating attack at step {self.global_step}')
                self.noise_activated = ~self.noise_activated
                self.noise_active_step = self.global_step
        
            #### logging:
            loss_dict = {f"train/{key}": val for key, val in loss_dict.items()}

            self.log_dict(loss_dict, prog_bar=True,
                        logger=True, on_step=True, on_epoch=True)
            
            self.log("global_step", self.global_step,
                    prog_bar=True, logger=True, on_step=True, on_epoch=False)
            
            return w_emb_loss
        
        # train image discriminator:
        if optimizer_idx == 1:
            loss_dict = {}

            # generate watermarked images:
            x_w, image_rec = self(x, img_rec_gt, c)

            ###### normalize the watermarked image:
            # image_rec = self.normalize_vqgan_to_imagenet(image_rec)

            # watermarked image prediction
            fake_preds = self.discriminator(image_rec).mean()
            
            ###### normalize the real image:
            # img_rec_gt = self.normalize_vqgan_to_imagenet(img_rec_gt)
            
            # real prediction
            real_preds = self.discriminator(img_rec_gt).mean()

            # clip weights:
            for p in self.discriminator.parameters():
                p.data.clamp_(-0.1, 0.1)
            
            ### total loss:
            d_loss = fake_preds - real_preds
            # d_loss = d_loss.mean()
            ##### loss dict update
            loss_dict['dis_loss'] = d_loss
            loss_dict['dis_real_loss'] = real_preds#.mean()
            loss_dict['dis_fake_loss'] = fake_preds#.mean()
            #loss_dict['optimizer_id'] = optimizer_idx

            loss_dict = {f"train/{key}": val for key, val in loss_dict.items()}

            self.log_dict(loss_dict, prog_bar=True,
                        logger=True, on_step=True, on_epoch=True)
            
            return d_loss

    
    def validation_step(self, batch, batch_idx):
        
        with torch.no_grad():
            loss_dict = {}
            x, c, img, img_rec_gt = self.get_input(batch)
            
            x_w, image_rec = self(x, img_rec_gt, c)
            
            rec_loss = self.compute_recon_loss(img_rec_gt.contiguous(), image_rec.contiguous())

            #### calculate psnr
            psnr = self.calculate_psnr(image_rec, img_rec_gt)
                
            if self.perceptual_loss_weight > 0:

                p_loss = self.perceptual_loss(img_rec_gt.contiguous() , image_rec.contiguous()).mean()
                loss_dict['emb_p_loss'] = p_loss
            else:
                p_loss = 0 

            ####### calculate discriminator loss:
            ## here we give cover image label to the watermarked images:
            d_on_watermarked_image = self.discriminator(image_rec)
            loss_adversarial = (-1.) * d_on_watermarked_image.mean()

            real_preds = self.discriminator(img_rec_gt).mean()

            fake_preds = self.discriminator(image_rec).mean()
            
            ####### applying noise:
            if hasattr(self, 'noise') and self.noise_activated:
                image_rec = self.noise(image_rec, self.global_step, active_step = self.noise_active_step, p=0.99)
            
            #### extract the watermark
            pred = self.decoder(image_rec)
            
            ### calculate watermark extraction and recreation loss:
            
            secret_loss = self.bce_loss(pred, 0.5*(c+1))
            
            w_emb_loss = rec_loss*self.recon_weight + self.adversarial_loss_weight * loss_adversarial + \
                            secret_loss * self.message_absolute_loss_weight + self.perceptual_loss_weight * p_loss

            ## final loss is average over batch size:     
            d_loss = fake_preds - real_preds
            diff = (~torch.logical_xor(pred>0, 0.5*(c+1)>0)) # b k -> b k
            bit_acc = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
            bit_acc = torch.mean(bit_acc)

            ##### loss dict update
            loss_dict['bit_acc'] = bit_acc
            loss_dict['psnr'] = psnr
            loss_dict['emb_loss'] = w_emb_loss
            loss_dict['emb_rec_loss'] = rec_loss
            loss_dict['emb_secret_loss'] = secret_loss
            loss_dict['emb_adversarial_loss'] = loss_adversarial
            loss_dict['dis_loss'] = d_loss
            loss_dict['dis_real_loss'] = real_preds#.mean()
            loss_dict['dis_fake_loss'] = fake_preds#.mean()
            
            loss_dict_val = {f"val/{key}": val for key, val in loss_dict.items() if key != 'img_lw'}
            self.log_dict(loss_dict_val, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        

        embedding_params = list(self.watermark_initial_0.parameters()) + \
                            list(self.watermark_initial_0_conv.parameters()) + \
                            list(self.watermark_initial.parameters()) + \
                            list(self.watermark_initial_conv.parameters()) + \
                            list(self.watermark[0].parameters()) + \
                            list(self.watermark_conv[0].parameters()) + \
                            list(self.watermark[1].parameters()) + \
                            list(self.watermark_conv[1].parameters()) + \
                            list(self.watermark[2].parameters()) + \
                            list(self.watermark_conv[2].parameters()) + \
                            list(self.watermark[3].parameters()) + \
                            list(self.watermark_conv[3].parameters()) + list(self.decoder.parameters())
        discriminator_params = list(self.discriminator.parameters())
        

        embedding_optimizer = torch.optim.AdamW(embedding_params, lr=self.learning_rate)
        discriminator_optimizer = torch.optim.AdamW(discriminator_params, lr=self.learning_rate)

        embedding_lr_scheduler = CosineAnnealingLR(embedding_optimizer, T_max = self.epoch_num)
        discriminator_lr_scheduler = CosineAnnealingLR(discriminator_optimizer, T_max = self.epoch_num)

        if self.dis_update_freq == 0:
            return [embedding_optimizer, discriminator_optimizer]
        elif self.dis_update_freq > 0:
            return [
                {
                    "optimizer": embedding_optimizer,
                    "frequency": 1,
                    # "lr_scheduler": embedding_lr_scheduler,
                },
                {
                    "optimizer": discriminator_optimizer,
                    "frequency": self.dis_update_freq,
                    # "lr_scheduler": discriminator_lr_scheduler,
                },
            ]

    
    def log_images(self, batch, fixed_input=False, **kwargs):
        with torch.no_grad():
            log = dict()
            if fixed_input and self.fixed_img is not None:
                x, c, img, img_recon = self.fixed_x, self.fixed_control, self.fixed_img, self.fixed_input_recon
            else:
                x, c, img, img_recon = self.get_input(batch)
            

            x, image_out = self(x, img_recon, c)
                

            image_out = torch.clamp(image_out, min=-1., max=1.)

            if hasattr(self, 'noise') and self.noise_activated:
                img_noise = self.noise(image_out, self.global_step, p=1.0)
                log['noised'] = img_noise
            log['input'] = img
            log['output'] = image_out
            log['recon'] = img_recon
            return log
    
    ### image to feature and feature to image functions:
    def decode_first_stage(self, z):
        z = 1./self.scale_factor * z
        image_rec = self.ae.decode(z)
        return image_rec

    def decode_first_stage_watermark(self, z):
        z = 1./self.scale_factor * z
        delta_I = self.ae_watermark_decoder(z)
        delta_I = self.tanh_activation(delta_I)
        return delta_I
    
    def encode_first_stage(self, image):
        encoder_posterior = self.ae.encode(image)
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    ### loss functions:
    def compute_recon_loss(self, inputs, reconstructions):
        if self.recon_type == 'rgb':
            # rec_loss = torch.abs(inputs - reconstructions).mean(dim=[1,2,3])
            # rec_loss = torch.mean((inputs - reconstructions)**2, dim=[1,2,3])
            rec_loss = torch.mean((inputs - reconstructions)**2)
        elif self.recon_type == 'yuv':
             
            reconstructions_yuv = self.rgb_to_yuv((reconstructions + 1) / 2)
            inputs_yuv = self.rgb_to_yuv((inputs + 1) / 2)
            yuv_loss = torch.mean((reconstructions_yuv - inputs_yuv)**2, dim=[2,3])
            rec_loss = torch.mean(torch.mm(yuv_loss, self.yuv_scales))
            
        elif self.recon_type == 'watson_vgg':
            rec_loss = self.loss_w_vgg((1+inputs)/2.0, (1+reconstructions)/2.0) / reconstructions.shape[0]   
        else:
            raise ValueError(f"Unknown recon type {self.recon_type}")
        return rec_loss
    
    
    def calculate_psnr(self, image_rec, img_rec_gt):
        ## calculate psnr:
        with torch.no_grad():
            delta = 255 * torch.clamp((image_rec+1.0) / 2.0 - (img_rec_gt+1.0) / 2.0, 0 , 1)
            delta = delta.reshape(-1, image_rec.shape[-3], image_rec.shape[-2], image_rec.shape[-1]) # BxCxHxW
            psnr = 20*np.log10(255) - 10*torch.log10(torch.mean(delta**2, dim=(1,2,3)))
            psnr = psnr.mean()
        return psnr
    
    def rgb_to_yuv(self, image):    
        r = image[..., 0, :, :]
        g = image[..., 1, :, :]
        b = image[..., 2, :, :]

        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.147 * r - 0.289 * g + 0.436 * b
        v = 0.615 * r - 0.515 * g - 0.100 * b

        return torch.stack([y, u, v], -3)
    
    def nonlinearity(self,x):
        # swish
        return x*torch.sigmoid(x)


class Discriminator1(nn.Module):
    """
    Discriminator network to differentiate between watermarked and non-watermarked images
    """
    def __init__(self):
        super(Discriminator1, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )

        self.head = nn.Linear(128, 1)

    def copy_encoder_weight(self, ae_model):
        return None
    
    def forward(self, image):
        x = self.layers(image) # (B,C,H,W) --> (B,C,1,1)
        x.squeeze_(3).squeeze_(2)
        x = self.head(x)
        return x


