
import os
import numpy as np
import torch
from torchvision import  transforms
from torchvision.transforms import functional
from augly.image import functional as aug_functional
import warnings
import contextlib
import requests
from urllib3.exceptions import InsecureRequestWarning
import ssl 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def random_crop(x, scale):
    crop_size = int(np.sqrt(scale) * x.size[0])
    crop_transform = transforms.RandomCrop(crop_size)
    x = crop_transform(x)
    return x

def center_crop(x, scale):
    scale = np.sqrt(scale)
    new_edges_size = [int(s*scale) for s in x.size][::-1]
    return functional.center_crop(x, new_edges_size)

def resize(x, scale):
    scale = np.sqrt(scale)
    new_edges_size = [int(s*scale) for s in x.size][::-1]
    return functional.resize(x, new_edges_size)

def comb(x):
    scale = np.sqrt(0.4)
    new_edges_size = [int(s*scale) for s in x.size][::-1]
    x = functional.center_crop(x, new_edges_size)
    x = functional.adjust_brightness(x,1.5)
    x = aug_functional.encoding_quality(x,quality=80)
    return x


ssl._create_default_https_context = ssl._create_unverified_context
old_merge_environment_settings = requests.Session.merge_environment_settings
@contextlib.contextmanager
def no_ssl_verification():
    opened_adapters = set()
    def merge_environment_settings(self, url, proxies, stream, verify, cert):

        opened_adapters.add(self.get_adapter(url))

        settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
        settings['verify'] = False

        return settings

    requests.Session.merge_environment_settings = merge_environment_settings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', InsecureRequestWarning)
            yield
    finally:
        requests.Session.merge_environment_settings = old_merge_environment_settings

        for adapter in opened_adapters:
            try:
                adapter.close()
            except:
                pass
