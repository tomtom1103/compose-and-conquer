import os
import sys
if './' not in sys.path:
	sys.path.append('./')
import time
import torch
import argparse
import numpy as np
from glob import glob
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from diffusers import StableDiffusionInpaintPipeline
from scipy.ndimage import binary_dilation

class CustomDataset(Dataset):
    def __init__(self, base_dir, placeholder, transform=None):
        self.image_dir = os.path.join(base_dir, placeholder)
        self.mask_dir = os.path.join(base_dir, f'{placeholder}_mask')
        self.transform = transform
        self.uid_list = list(glob(os.path.join(self.image_dir, '*.jpg'), recursive=True))

    def __len__(self):
        return len(self.uid_list)

    def __getitem__(self, idx):
        uid = os.path.basename(self.uid_list[idx][:-4])
        image_path = os.path.join(self.image_dir, f"{uid}.jpg")
        mask_path = os.path.join(self.mask_dir, f"{uid}.png")
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        if self.transform is not None:
            image = self.transform(image).convert("RGB")
            mask = self.transform(mask)

        return {"jpg": image, "mask": mask, "uid": uid}


def get_background(args):
    base_dir = args.base_dir
    placeholder = args.placeholder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize(512, transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
    ])
    dataset = CustomDataset(
        base_dir=base_dir,
        placeholder=placeholder,
        transform=transform
    )
    batch_size = 1
    assert batch_size == 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    to_pil = transforms.ToPILImage()
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float32, safety_checker=None)
    
    pipe = pipe.to(device)
    prompt = "empty scenery, ultra detailed, no people"
    neg_prompt = "low resolution, worst quality, low quality"

    background_dir = os.path.join(base_dir, f"{placeholder}_background")
    os.makedirs(background_dir, exist_ok=True)

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        image = batch['jpg'][0]
        uid = batch['uid'][0]
        mask = to_pil(batch['mask'][0])

        mask = np.array(mask)
        mask = np.expand_dims(mask, axis=-1)
        binary_mask = mask == 255
        dilated_mask = binary_dilation(binary_mask, structure=np.ones((50,50,1)))
        dilated_mask = dilated_mask.astype(np.uint8) * 255

        pil_image = to_pil(image)
        pil_mask = to_pil(dilated_mask)

        out = pipe(prompt=prompt,
           negative_prompt=neg_prompt,
           image=pil_image,
           mask_image=pil_mask,
           strength=1,
           guidance_scale=10,
           num_inference_steps=25,
           ).images[0]
        
        background_path = os.path.join(background_dir, f"{uid}.jpg")
        out.save(background_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get pickapic background")
    parser.add_argument("--base_dir", type=str, default="/workspace/pickapic", help="base directory containing images and masks")
    parser.add_argument("--placeholder", type=str, default="val", help="either train or val")
    args = parser.parse_args()
    get_background(args)