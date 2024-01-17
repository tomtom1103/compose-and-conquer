import os
import sys
if './' not in sys.path:
	sys.path.append('./')
import time
import torch
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from utils.utils import get_folder_size

class CocoStuffDataset(Dataset):
    def __init__(self, base_dir, placeholder, transform=None):
        self.image_dir = os.path.join(base_dir, placeholder)
        self.seg_dir = os.path.join(os.path.dirname(base_dir), "annotations", placeholder)
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
        seg_path = os.path.join(self.seg_dir, f"{uid}.png")
        image = Image.open(image_path).convert("RGB")
        seg = Image.open(seg_path)
        if self.transform is not None:
            image = self.transform(image)
            seg = self.transform(seg)

        return {"jpg": image, "txt": caption, "uid": uid, "seg": seg}

def cocostuff_get_mask(args):
    base_dir = args.base_dir
    placeholder = args.placeholder

    start = time.time()
    transform = transforms.Compose([
        transforms.Resize(512, transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
    ])
    dataset = CocoStuffDataset(
        base_dir=base_dir,
        placeholder=placeholder,
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    base_dir = "./"
    mask_dir = os.path.join(base_dir, f"{placeholder}_mask")
    os.makedirs(mask_dir, exist_ok=True)

    to_pil = transforms.ToPILImage()

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        uids = batch['uid']
        masks = batch['seg']
        for mask, uid in zip(masks, uids):
            mask = np.array(to_pil(mask))
            mask = np.where(mask==0, 90, mask) # 사람이 0이라 일단 90으로 보냄
            mask = np.where(mask < 91, mask, 0) # things 카테고리 빼고 0으로 
            mask = np.where((mask > 0) & (mask < 91), 255, mask) # things 카테고리 255로
            mask = np.expand_dims(mask, axis=-1)
            binary_mask = mask == 255
            undilated_mask = binary_mask.astype(np.uint8) * 255
            save_mask = to_pil(undilated_mask)
            mask_path = os.path.join(mask_dir, f"{uid}.png")
            save_mask.save(mask_path)

    end = time.time()
    total_time = end - start
    print(f"Total execution time: {total_time/60/60:.2f} hours.")
    print(f"Size of the folder '{mask_dir}': {get_folder_size(mask_dir):.2f} GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get cocostuff masks")
    parser.add_argument("--base_dir", type=str, default="/workspace/cocostuff/dataset/images", help="base directory containing images and captions file")
    parser.add_argument("--placeholder", type=str, default="val2017", help="either train or val")
    args = parser.parse_args()
    cocostuff_get_mask(args)