import os
import sys
if './' not in sys.path:
	sys.path.append('./')
import io
import time
import argparse
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from torchvision.transforms import transforms
from utils.utils import STOPWORDS, get_folder_size

def parse_pickapic(args):
    start = time.time()
    resize = transforms.Compose([
            transforms.Resize(512, transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(512),
        ])

    save_dir = args.image_save_dir
    parquet_dir = args.parquet_dir
    caption_file = args.caption_file
    os.makedirs(save_dir, exist_ok=True)

    parquet_files = [f for f in os.listdir(parquet_dir) if f.endswith('.parquet')]

    stopwords = STOPWORDS
    saved_uids = set()
    skipped = 0

    with open(caption_file, 'w') as file:
        for file_num, parquet_file in tqdm(enumerate(parquet_files), total=len(parquet_files)):
            current_skipped = 0
            df = pd.read_parquet(os.path.join(parquet_dir, parquet_file))
            
            for i, row in tqdm(df.iterrows(), total=df.shape[0]):
                caption = row['caption'].replace(':', '')
                if any(word in caption.lower().split() for word in stopwords) or len(caption.split())<3:
                    current_skipped += 1
                    skipped += 1
                    continue

                if row['label_0'] > row['label_1']:
                    img_data = row['jpg_0']
                    uid = row['image_0_uid']
                elif row['label_0'] < row['label_1']:
                    img_data = row['jpg_1']
                    uid = row['image_1_uid']
                else:  # tied case
                    img_data = row['jpg_1']
                    uid = row['image_1_uid']

                if uid not in saved_uids:
                    saved_uids.add(uid)
                    save_path = os.path.join(save_dir, f'{uid}.jpg')
                    image = Image.open(io.BytesIO(img_data))
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    image = resize(image)
                    image.save(save_path)
                    file.write(f'{uid}:{caption}\n')
                else:
                    current_skipped += 1

            print(f"{file_num}: Number of samples skipped: {current_skipped} out of {len(df)}: {current_skipped/len(df) * 100:.2f}%")

    end = time.time()
    total_time = end-start
    print(f"Total execution time: {total_time/60/60:.2f} hours.")
    print(f"Number of total samples skipped: {skipped}")
    print(f"Size of the folder '{save_dir}': {get_folder_size(save_dir):.2f} GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse Pick-a-Pick")
    parser.add_argument("--parquet_dir", type=str, default="/workspace/pickapic/val_parquet", help="directory containing multiple pickapic parquet files")
    parser.add_argument("--image_save_dir", type=str, default="/workspace/pickapic/val", help="where to save filtered pickapic .jpg images")
    parser.add_argument("--caption_file", type=str, default="/workspace/pickapic/val_captions.txt", help="filename of where to save filtered captions")
    args = parser.parse_args()
    parse_pickapic(args)