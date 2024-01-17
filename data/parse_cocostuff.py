import os
import json
import argparse

def parse_cocostuff(args):
    json_dir = args.json_dir
    image_dir = args.image_dir
    caption_file = args.caption_file

    with open(json_dir) as file:
        data = json.load(file)

    annotations = data['annotations']
    processed_images = set()

    with open(caption_file, 'w') as out:
        first_caption = True
        for annotation in annotations:
            image_name = f"{str(annotation['image_id']).zfill(12)}"
            image_file = f"{image_name}.jpg"

            if os.path.exists(os.path.join(image_dir, image_file)) and image_name not in processed_images:
                caption = annotation['caption'].replace('\n', ' ').replace('\r', ' ')

                if not first_caption:
                    out.write('\n')
                else:
                    first_caption = False

                out.write(f"{image_name}:{caption}")
                processed_images.add(image_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse COCO-Stuff")
    parser.add_argument("--json_dir", type=str, default="/workspace/annotations/captions_val2017.json", help="json file of MS COCO captions")
    parser.add_argument("--image_dir", type=str, default="/workspace/cocostuff/dataset/images", help="COCO-Stuff image dir")
    parser.add_argument("--caption_file", type=str, default="/workspace/cocostuff/dataset/images/val2017_captions.txt", help="filename of where to save filtered captions")
    args = parser.parse_args()
    parse_cocostuff(args)