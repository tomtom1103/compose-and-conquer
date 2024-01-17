import os
import cv2
import torch
from torchvision.transforms import transforms
from transformers import logging
import numpy as np
from PIL import Image

STOPWORDS=[
    "naked", "nude", "sex", "porn", "adult", "erotic", "explicit", "xxx", "nsfw", 
    "breast", "boobs", "genital", "vagina", "penis", "ass", "butt", "bdsm", 
    "fetish", "kinky", "nudity", "violence", "gore", "blood", "killing", "murder", 
    "rape", "abuse", "suicide", "death", "prostitute", "escort", "stripper", "drug",
    "cocaine", "marijuana", "weed", "meth", "lsd", "heroin", "ecstasy", "acid", "girl",
    "teen", "woman", "boy", "milf", "female", "seductress", "kiss", "gloryhole", "testicle",
    "boob", "women", "cleavage", "sexy", "女", "giantess", "fuck", "fuuck", "fuucking", "waifu",
    "risqué", "spread", "spreaded", "spreading", "dildo", "erection", "vibrator", "naturist"
]

def print_all_children(module, prefix="", depth=3, current_depth=0, ignore=[]):
    if current_depth == depth:
        return

    for name, child in module.named_children():
        if not name.isdigit():
            print(prefix + name)
            if name in ignore:
                print(prefix + "  PASS")
            else:
                print_all_children(child, prefix=prefix + "  ", depth=depth, current_depth=current_depth+1, ignore=ignore)

def count_params(model):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{count:,}')

def get_folder_size(folder_path):
    total_size = 0
    for path, dirs, files in os.walk(folder_path):
        for f in files:
            fp = os.path.join(path, f)
            total_size += os.path.getsize(fp)

    total_size = total_size / (1024 ** 3)
    return total_size

def batch_to_device(batch, device):
    for key, value in batch.items():
        if torch.is_tensor(value):
            batch[key] = batch[key].to(device)
    return batch

def disable_verbosity():
    logging.set_verbosity_error()
    # print('logging improved.')
    return

def add_gridlines(image_array):
    """
    input: np.array of 1,3,h,w
    """
    full = Image.fromarray(image_array)

    h, w = full.size
    if h != w:
        raise NotImplementedError
    
    to_tensor = transforms.ToTensor()
    embedding_grid = torch.zeros([3, h, w])
    thickness = 10
    
    # Add horizontal lines
    embedding_grid[:, h // 3 - thickness // 2 : w // 3 + thickness // 2, :] = 1
    embedding_grid[:, 2 * h // 3 - thickness // 2 : 2 * w // 3 + thickness // 2, :] = 1
    
    # Add vertical lines
    embedding_grid[:, :, h // 3 - thickness // 2 : w // 3 + thickness // 2] = 1
    embedding_grid[:, :, 2 * h // 3 - thickness // 2 : 2 * w // 3 + thickness // 2] = 1
    
    # Convert tensor grid to a PIL image
    grid_img = transforms.ToPILImage()(embedding_grid)
    
    # Paste the grid onto the original image
    full.paste(grid_img, (0, 0), grid_img.convert('L'))

    full_array = np.array(full)

    return full_array



def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img
