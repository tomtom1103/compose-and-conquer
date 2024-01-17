import os
import sys
if './' not in sys.path:
	sys.path.append('./')
import einops
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
from ldm.util import instantiate_from_config

class MergedDataset(Dataset):
    def __init__(self, dataset_configs):
        self.datasets = [instantiate_from_config(config) for config in dataset_configs]
        self.lengths = [len(dataset) for dataset in self.datasets]
        self.offsets = [0] + np.cumsum(self.lengths).tolist()

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        for i, dataset in enumerate(self.datasets):
            if idx < self.lengths[i]:
                return dataset[idx]
            idx -= self.lengths[i]
        raise IndexError("Index out of range")
    
class CnCDataset_Local(Dataset):
    def __init__(self,
                 base_dir,
                 placeholder,
                 drop_txt_prob,
                 drop_cond_prob):
        
        self.image_dir = os.path.join(base_dir, placeholder)
        self.bg_depth_dir = os.path.join(base_dir, f"{placeholder}_background_depthmaps")
        self.fg_depth_dir = os.path.join(base_dir, f"{placeholder}_foreground_depthmaps")
        self.drop_txt_prob = drop_txt_prob
        self.drop_cond_prob = drop_cond_prob
        captions_file = os.path.join(base_dir, f"{placeholder}_captions.txt")
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
        self.transform_normalize = transforms.Compose([
            transforms.Resize(512, transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.transform = transforms.Compose([ #HACK
            transforms.Resize(512, transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.captions_df)
    
    def __getitem__(self, idx):
        uid, caption = self.captions_df.iloc[idx]
        image = Image.open(os.path.join(self.image_dir, f"{uid}.jpg"))
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = self.transform_normalize(image)

        bg_depth = Image.open(os.path.join(self.bg_depth_dir, f"{uid}.jpg"))
        bg_depth = self.transform(bg_depth)
        
        fg_depth = Image.open(os.path.join(self.fg_depth_dir, f"{uid}.jpg"))
        fg_depth = self.transform(fg_depth)
        
        if random.random() < self.drop_txt_prob:
            caption = ''

        if random.random() < self.drop_cond_prob:
            bg_depth = torch.zeros_like(bg_depth)

        if random.random() < self.drop_cond_prob:
            fg_depth = torch.zeros_like(fg_depth)

        image = einops.rearrange(image, 'c h w -> h w c')
        bg_depth = einops.rearrange(bg_depth, 'c h w -> h w c')
        fg_depth = einops.rearrange(fg_depth, 'c h w -> h w c')

        return {"jpg": image, "txt": caption, "bg_depth": bg_depth, "fg_depth": fg_depth,}

class CnCDataset_Global(Dataset):
    def __init__(self,
                 base_dir,
                 placeholder,
                 drop_txt_prob,
                 drop_cond_prob):
        
        self.image_dir = os.path.join(base_dir, placeholder)
        self.bg_emb_dir = os.path.join(base_dir, f"{placeholder}_background_embeddings")
        self.fg_emb_dir = os.path.join(base_dir, f"{placeholder}_foreground_embeddings")
        self.mask_dir = os.path.join(base_dir, f"{placeholder}_mask")

        self.drop_txt_prob = drop_txt_prob
        self.drop_cond_prob = drop_cond_prob
        captions_file = os.path.join(base_dir, f"{placeholder}_captions.txt")
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
        self.transform_normalize = transforms.Compose([
            transforms.Resize(512, transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.captions_df)
    
    def __getitem__(self, idx):
        uid, caption = self.captions_df.iloc[idx]
        image = Image.open(os.path.join(self.image_dir, f"{uid}.jpg"))
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = self.transform_normalize(image)
        
        bg_emb = torch.load(os.path.join(self.bg_emb_dir, f"{uid}.pt"))
        fg_emb = torch.load(os.path.join(self.fg_emb_dir, f"{uid}.pt"))
        mask = Image.open(os.path.join(self.mask_dir, f"{uid}.png"))
        mask = self.to_tensor(mask)

        if random.random() < self.drop_txt_prob:
            caption = ''

        if random.random() < self.drop_cond_prob:
            bg_emb = torch.zeros_like(bg_emb)

        if random.random() < self.drop_cond_prob:
            fg_emb = torch.zeros_like(fg_emb)

        bg_emb = bg_emb.squeeze()
        fg_emb = fg_emb.squeeze()

        image = einops.rearrange(image, 'c h w -> h w c')

        return {"jpg": image, "txt": caption, "bg_emb": bg_emb, "fg_emb": fg_emb, "mask": mask}

if __name__ == "__main__":
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from torchvision.transforms import transforms
    import einops

    config = OmegaConf.load("configs/global_fuser_v1.yaml")
    dataset = instantiate_from_config(config['data'])

    dataloader = DataLoader(dataset, num_workers=0, batch_size=2, pin_memory=True, shuffle=False)
    batch = next(iter(dataloader))
    img = batch['jpg']

    for batch in tqdm(dataloader):
        print(batch['txt'])
        pass

