import os
import sys
if './' not in sys.path:
	sys.path.append('./')
import time
import torch
import einops
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from torchvision.utils import save_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from utils.utils import batch_to_device, disable_verbosity
from utils.primitives import GetDepthMap, GetCLIPImageEmbeddings
disable_verbosity()

class CustomDataset(Dataset):
    def __init__(self, base_dir, placeholder, transform=None):
        self.image_dir = os.path.join(base_dir, placeholder)
        self.mask_dir = os.path.join(base_dir, f'{placeholder}_mask')
        self.bg_dir = os.path.join(base_dir, f'{placeholder}_background')
        self.fg_dir = os.path.join(base_dir, f'{placeholder}_foreground')
        self.transform = transform
        captions_file = os.path.join(base_dir, f'{placeholder}_captions.txt')

        data = []
        with open(captions_file, 'r') as file:
            lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line:
                try:
                    uid, caption = line.split(":")
                    data.append([uid, caption])
                except ValueError:
                    pass
        self.captions_df = pd.DataFrame(data, columns=["uid", "caption"])

    def __len__(self):
        return len(self.captions_df)

    def __getitem__(self, idx):
        uid, caption = self.captions_df.iloc[idx]
        image_path = os.path.join(self.image_dir, f"{uid}.jpg")
        fg_path = os.path.join(self.fg_dir, f"{uid}.jpg")
        bg_path = os.path.join(self.bg_dir, f"{uid}.jpg")
        image = Image.open(image_path).convert("RGB")
        fg = Image.open(fg_path).convert("RGB")
        bg = Image.open(bg_path).convert("RGB")
        
        image = np.array(self.transform(image))
        fg = np.array(self.transform(fg))
        bg = np.array(self.transform(bg))

        return {"jpg": image, "uid": uid, "fg":fg, "bg":bg}


def get_primitives(args):
    base_dir = args.base_dir
    placeholder = args.placeholder

    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize(512, transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(512),
    ])

    dataset = CustomDataset(
        base_dir=base_dir,
        placeholder=placeholder,
        transform=transform
    )

    batch_size=1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    bg_depth_dir = os.path.join(base_dir, f"{placeholder}_background_depthmaps")
    bg_emb_dir = os.path.join(base_dir, f"{placeholder}_background_embeddings")
    fg_depth_dir = os.path.join(base_dir, f"{placeholder}_foreground_depthmaps")
    fg_emb_dir = os.path.join(base_dir, f"{placeholder}_foreground_embeddings")

    os.makedirs(bg_depth_dir, exist_ok=True)
    os.makedirs(bg_emb_dir, exist_ok=True)
    os.makedirs(fg_depth_dir, exist_ok=True)
    os.makedirs(fg_emb_dir, exist_ok=True)

    get_depth = GetDepthMap(device=device)
    get_emb = GetCLIPImageEmbeddings(device=device)

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch = batch_to_device(batch, device)
        base_images = batch['jpg']
        bg_images = batch['bg']
        fg_images = batch['fg']
        uids = batch['uid']

        bg_embs = get_emb(bg_images)
        fg_embs = get_emb(base_images)
        bg_images = einops.rearrange(bg_images.to(torch.float)/255.0, 'b h w c -> b c h w')
        fg_images = einops.rearrange(fg_images.to(torch.float)/255.0, 'b h w c -> b c h w')

        bg_depths = get_depth(bg_images)
        fg_depths = get_depth(fg_images)

        for j in range(batch_size):
            uid = uids[j]
            bg_depth = bg_depths[j]
            bg_emb = bg_embs[j].to("cpu")
            fg_depth = fg_depths[j]
            fg_emb = fg_embs[j].to("cpu")

            bg_depth_path = os.path.join(bg_depth_dir, f"{uid}.jpg")
            bg_emb_path = os.path.join(bg_emb_dir, f"{uid}.pt")
            fg_depth_path = os.path.join(fg_depth_dir, f"{uid}.jpg")
            fg_emb_path = os.path.join(fg_emb_dir, f"{uid}.pt")

            save_image(bg_depth, bg_depth_path)
            torch.save(bg_emb, bg_emb_path)
            save_image(fg_depth, fg_depth_path)
            torch.save(fg_emb, fg_emb_path)

    end = time.time()
    total_time = end - start
    print(f"Total execution time: {total_time/60/60:.2f} hours.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get primitives")
    parser.add_argument("--base_dir", type=str, default="/workspace/pickapic", help="base directory containing bg/fg images, masks, and captions")
    parser.add_argument("--placeholder", type=str, default="val", help="either train or val")
    args = parser.parse_args()
    get_primitives(args)