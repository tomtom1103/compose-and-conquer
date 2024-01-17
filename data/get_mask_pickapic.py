import os
import sys
if './' not in sys.path:
	sys.path.append('./')
import time
import torch
import argparse
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from utils.primitives import GetSODMask
from utils.utils import get_folder_size

class PickapicDataset(Dataset):
    def __init__(self, base_dir, placeholder, transform=None):
        self.image_dir = os.path.join(base_dir, placeholder)
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
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return {"jpg": image, "txt": caption, "uid": uid}

def pickapic_get_mask(args):
    base_dir = args.base_dir
    placeholder = args.placeholder

    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize(512, transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
    ])
    dataset = PickapicDataset(
        base_dir=base_dir,
        placeholder=placeholder,
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    get_mask = GetSODMask(device=device)

    mask_dir = os.path.join(base_dir, f"{placeholder}_mask")
    os.makedirs(mask_dir, exist_ok=True)

    to_pil = transforms.ToPILImage()

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        images = batch['jpg'].to(device)
        uids = batch['uid']
        masks = get_mask(images)
        for mask, uid in zip(masks, uids):
            mask = to_pil(mask)
            mask_path = os.path.join(mask_dir, f"{uid}.png")
            mask.save(mask_path)

    end = time.time()
    total_time = end - start
    print(f"Total execution time: {total_time/60/60:.2f} hours.")
    print(f"Size of the folder '{mask_dir}': {get_folder_size(mask_dir):.2f} GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get pickapic masks")
    parser.add_argument("--base_dir", type=str, default="/workspace/pickapic", help="base directory containing images and captions file")
    parser.add_argument("--placeholder", type=str, default="val", help="either train or val")
    args = parser.parse_args()
    pickapic_get_mask(args)