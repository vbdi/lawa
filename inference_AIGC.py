"""
This code is based on the text-to-image script provided by Stable Diffusion repository.
"""
import os, sys, torch, glob 
import argparse
from pathlib import Path
import numpy as np
from torchvision import transforms
import argparse
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from tools.eval_metrics import compute_psnr, compute_ssim, compute_mse, compute_lpips, compute_sifid
import lpips
from tools.sifid import SIFID
import torch
import pytorch_lightning as pl
import pandas as pd
import cv2
import utils_img

from torchvision.transforms import functional
import torch.nn.functional as nnf

## Stable Diffusion requirements:
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from augly.image import functional as aug_functional

from torch.autograd.variable import Variable
import torch.nn.functional as F
from utils_img import no_ssl_verification

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def generate_attacks(img, attacks):
    """ Code from SSL watermark"""
    """ Generate a list of attacked images from a PIL image. """
    attacked_imgs = []
    for attack in attacks:
        attack = attack.copy()
        attack_name = attack.pop('attack')
        attacked_imgs.append(attacks_dict[attack_name](img, **attack))
    return attacked_imgs

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

os.environ['CURL_CA_BUNDLE'] = ''
def main(args):

    ### Loading Stable Diffuion model
    ## SD txt2img function:
    if args.laion400m:
        print("Falling back to LAION 400M model...")
        args.config_sd = "stable-diffusion/configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        args.ckpt = "stable-diffusion/models/ldm/text2img-large/model.ckpt"
        args.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(args.seed)

    config_sd = OmegaConf.load(f"{args.config_sd}")
    with no_ssl_verification():
        model_sd = load_model_from_config(config_sd, f"{args.ckpt}")

    model_sd = model_sd.to(device)

    if args.dpm_solver:
        sampler = DPMSolverSampler(model_sd)
    elif args.plms:
        sampler = PLMSSampler(model_sd)
    else:
        sampler = DDIMSampler(model_sd)

    os.makedirs(args.outdir, exist_ok=True)
    outpath = args.outdir

    batch_size = args.n_samples
    n_rows = args.n_rows if args.n_rows > 0 else batch_size
    if not args.from_file:
        prompt = args.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {args.from_file}")
        with open(args.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if args.fixed_code:
        start_code = torch.randn([args.n_samples, args.C, args.H // args.f, args.W // args.f], device=device)

    precision_scope = autocast if args.precision=="autocast" else nullcontext

    ### Load the pre-trained modified decoder model
    config = OmegaConf.load(args.config).model
    message_len = config.params.decoder_config.params.message_len
    if int(args.message_len) != message_len:
        raise Exception(f"Provided message_len argument does not match the message length in the config file!")
    model = instantiate_from_config(config)
    # print(model.decoder)
    state_dict = torch.load(args.weight, map_location=torch.device('cpu'))
    if 'global_step' in state_dict:
        print(f'Global step: {state_dict["global_step"]}, epoch: {state_dict["epoch"]}')

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    misses, ignores = model.load_state_dict(state_dict, strict=False)
    print(f'Missed keys: {misses}\nIgnore keys: {ignores}')
    
    # device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")    
    model = model.cuda(device)
    
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    ## load the message:
    if args.message == '':
        test_message_num = 1
        manual_seeds = [112]
        torch.manual_seed(manual_seeds[0]) ## set the manual seed for reproducibility
        message = torch.zeros((test_message_num, message_len), dtype=torch.float).random_(0,2).cuda(device)
        torch.seed()
        message = 2. * message - 1.
    else:
        if len(args.message) != message_len:
            raise Exception(f"Provided message should be {message_len} bits!")
        message = [int(j) for j in args.message]
        message = torch.from_numpy(np.array(message)).cuda(device)
        message = message.unsqueeze_(0)
        message = 2. * message - 1.

    message = message.repeat(args.n_samples,1)

    ############ inference
    lpips_alex = lpips.LPIPS(net='alex').cuda(device)
    sifid_model = SIFID()

    results = pd.DataFrame(columns=["image_name","MSE","PSNR","SSIM","LPIPS","SIFID","Bit Acc"])
    
    with torch.no_grad():
        ######## using SD sampler to generate z
        with precision_scope("cuda"):
            with model_sd.ema_scope():
                tic = time.time()
                for n in trange(args.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if args.scale != 1.0:
                            uc = model_sd.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model_sd.get_learned_conditioning(prompts)
                        shape = [args.C, args.H // args.f, args.W // args.f]
                        samples_ddim, _ = sampler.sample(S=args.ddim_steps,
                                                         conditioning=c,
                                                         batch_size=args.n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=args.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=args.ddim_eta,
                                                         x_T=start_code)
        
                        
                        ## generate the image with the original decoder:
                        x_samples_ddim = model_sd.decode_first_stage(samples_ddim)
                        
                        ## generate the image with modified decoder:
                        samples_ddim = model.ae.post_quant_conv(1./model.scale_factor * samples_ddim)
                        _ , W_x_samples_ddim = model(samples_ddim, x_samples_ddim, message)

                        # map to between 0 and 1
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        w_images = torch.clamp((W_x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                        for i, x_sample in enumerate(x_samples_ddim):
                            X_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            img = Image.fromarray(X_sample.astype(np.uint8))
                            img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                            
                            w_img = 255. * rearrange(w_images[i].cpu().numpy(), 'c h w -> h w c')
                            w_img = Image.fromarray(w_img.astype(np.uint8))
                            w_img.save(os.path.join(sample_path, f"{base_count:05}_watermarked.png"))

                            ## difference
                            diff_img = np.abs(np.asarray(img).astype(int) - np.asarray(w_img).astype(int)) *10
                            diff_img = Image.fromarray(diff_img.astype(np.uint8))
                            diff_img.save(os.path.join(sample_path, f"{base_count:05}_diff.png"))

                            # quality metrics
                            MSE = compute_mse(np.array(img)[None,...], np.array(w_img)[None,...])
                            PSNR = compute_psnr(np.array(img)[None,...], np.array(w_img)[None,...])
                            SSIM = compute_ssim(np.array(img)[None,...], np.array(w_img)[None,...])
                            print(f'MSE: {MSE}')
                            print(f'PSNR: {PSNR}')
                            print(f'SSIM: {SSIM}')
                            
                            org_norm = torch.from_numpy(np.array(img)[None,...]/127.5-1.).permute(0,3,1,2).float().cuda()
                            w_norm = torch.from_numpy(np.array(w_img)[None,...]/127.5-1.).permute(0,3,1,2).float().cuda()
                            LPIPS = compute_lpips(org_norm, w_norm, lpips_alex)
                            SIFIC = compute_sifid(org_norm, w_norm, sifid_model)
                            print(f'LPIPS: {LPIPS}')
                            print(f'SIFID: {SIFIC}')

                            # decode
                            print('Extracting message...')
                            message_extracted = model.decoder(W_x_samples_ddim)
                                
                            
                            diff = (~torch.logical_xor(message_extracted[i]>0, 0.5*(message+1)>0)) # b k -> b k
                            bit_acc = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
                            bit_acc = torch.mean(bit_acc).cpu().numpy()
                            print(f'Bit acc: {bit_acc}')

                            results.loc[len(results)] = [f"{base_count:05}.png", MSE[0], PSNR[0], SSIM[0], LPIPS, SIFIC[0], bit_acc]

                            # apply attacks:
                            attacked_w_imgs = generate_attacks(w_img, attacks)
                            temp_result = [f"{base_count:05}.png"]
                            for j, attacked_img in enumerate(attacked_w_imgs):
                                # decode
                                attacked_img = torch.from_numpy(np.asarray(attacked_img)).cuda(device)
                                attacked_img = (attacked_img / 255.) * 2. - 1.0 
                                attacked_img = torch.permute(torch.unsqueeze(attacked_img,0), (0,3,1,2))
                                
                                # extracted message:
                                message_extracted = model.decoder(attacked_img) 
                                diff = (~torch.logical_xor(message_extracted>0, 0.5*(message+1)>0)) # b k -> b k
                                bit_acc = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
                                bit_acc = torch.mean(bit_acc).cpu().numpy()
                                temp_result.append(bit_acc)
                            
                            attack_results.loc[len(attack_results)] = temp_result
                            
                            
                            base_count += 1

    results.to_csv("results/SD14_LaWa/test_results_quality.csv")
    attack_results.to_csv("results/SD14_LaWa/test_results_attacks.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--config", default='configs/SD14_LaWa_inference.yaml', help="Path to config file of the modified decoder")
    parser.add_argument('-w', "--weight", default='weights/LaWa/last.ckpt', help="Path to saved checkpoint of the modified decoder")
    
    parser.add_argument(
        "--message", default='', help="waternark message"
    )
    parser.add_argument(
        "--message_len", default=48, help="Length of watermark message"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="results/SD14_LaWa/txt2img-samples"
    )   
    parser.add_argument(
        "--ckpt",
        type=str,
        default="weights/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--config_sd",
        type=str,
        default="stable-diffusion/configs/stable-diffusion/v1-inference_wm.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--from_file",
        type=str,
        help="if specified, load prompts from this file",
    )
    
    
    ###########################################################################
    ## SD arguments:
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="A white plate of food on a dining table",
        help="the prompt to render"
    )
    
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    
    ## for attack:
    attacks_dict = {
        "none": lambda x : x,
        "rotation": functional.rotate,
        "grayscale": functional.rgb_to_grayscale,
        "contrast": functional.adjust_contrast,
        "brightness": functional.adjust_brightness,
        "hue": functional.adjust_hue,
        "hflip": functional.hflip,
        "vflip": functional.vflip,
        "blur": functional.gaussian_blur, # sigma = ksize*0.15 + 0.35  - ksize = (sigma-0.35)/0.15
        "jpeg": aug_functional.encoding_quality,
        "resize": utils_img.resize,
        "center_crop": utils_img.center_crop,
        "comb": utils_img.comb,
    }

    attacks = [{'attack': 'none'}] \
        + [{'attack': 'rotation', 'angle': jj} for jj in range(0,25,5)] \
        + [{'attack': 'center_crop', 'scale': 0.1*jj} for jj in range(1,11)] \
        + [{'attack': 'resize', 'scale': 0.1*jj} for jj in range(7,11)] \
        + [{'attack': 'blur', 'kernel_size': 1+2*jj} for jj in range(1,10)] \
        + [{'attack': 'jpeg', 'quality': 10*jj} for jj in range(1,11)] \
        + [{'attack': 'contrast', 'contrast_factor': 0.5*jj} for jj in range(1,5)] \
        + [{'attack': 'brightness', 'brightness_factor': 0.5*jj} for jj in range(1,5)] \
        + [{'attack': 'hue', 'hue_factor': -0.5 + 0.25*jj} for jj in range(1,5)] \
        + [{'attack': 'hue', 'hue_factor': 0.2}] \
        + [{'attack': 'comb'}]

    attack_columns =['img_name']
    for jj in range(len(attacks)):
        attack = attacks[jj].copy()
        # change params name before logging to harmonize df between attacks
        attack_name = attack.pop('attack')
        param_names = ['param%i'%kk for kk in range(len(attack.keys()))]
        attack_params = dict(zip(param_names,list(attack.values())))
        if len(param_names) >=1:
            column_name = attack_name + "_" + list(attack.keys())[0] + str(attack_params['param0'])
        else:
            column_name = attack_name
        print(column_name)
        attack_columns.append(column_name)

    attack_results = pd.DataFrame(columns = attack_columns)
    args = parser.parse_args()
    main(args)