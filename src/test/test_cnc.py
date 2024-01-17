import sys
if './' not in sys.path:
	sys.path.append('./')
import random
import torch
import cv2
import einops
import gradio as gr
import numpy as np

from torchvision.transforms import transforms
from pytorch_lightning import seed_everything
from models.util import create_model, load_state_dict
from models.ddim_hacked import DDIMSampler
from utils.utils import resize_image, HWC3, disable_verbosity, add_gridlines
from utils.primitives import GetDepthMap, GetSODMask, GetCLIPImageEmbeddings

import warnings
warnings.filterwarnings("ignore")

disable_verbosity()
DEFAULT_NEGATIVE_PROMPT = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model('./configs/cnc_v1.yaml').cpu()
model.load_state_dict(load_state_dict('./trained_weights/cnc_v1.ckpt'))
model = model.to(device)
ddim_sampler = DDIMSampler(model)

get_depth = GetDepthMap(device=device)
get_mask = GetSODMask(device=device)
get_emb = GetCLIPImageEmbeddings(device=device)

to_pil = transforms.ToPILImage()

def process(fg_image,
            bg_image,
            fg_emb_image,
            bg_emb_image,
            depth_version,
            fg_depth_control,
            bg_depth_control,
            fg_global_control,
            bg_global_control,
            hw,
            batch_size,
            prompt,
            neg_prompt,
            ddim_steps,
            cfg_scale,
            seed,
            ):
    print("Running...")
    if seed == -1:
        seed = random.randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)

    seed_everything(seed)

    to_tensor = transforms.ToTensor()
    H, W, C = hw, hw, 3

    if fg_image is not None:
        fg_image = cv2.resize(fg_image, (W,H))
        mask_anchor = fg_image
    else:
        fg_image = np.zeros((1, 3, hw, hw))
        mask_anchor = None
        
    if bg_image is not None:
        bg_image = cv2.resize(bg_image, (W,H))
    else:
        bg_image = np.zeros((1, 3, hw, hw))

    if fg_emb_image is not None:
        fg_emb_image = cv2.resize(fg_emb_image, (W,H))
        fg_emb = get_emb(fg_emb_image)
        if mask_anchor is None:
            mask_anchor = fg_emb_image
    else:
        fg_emb_image = np.zeros((1, 3, hw, hw))
        fg_emb = torch.zeros(1,768).to(device)
        if mask_anchor is None:
            mask_anchor = fg_emb_image
        
    if bg_emb_image is not None:
        bg_emb_image = cv2.resize(bg_emb_image, (W,H))
        bg_emb = get_emb(bg_emb_image)
    else:
        bg_emb_image = np.zeros((1, 3, hw, hw))
        bg_emb = torch.zeros(1,768).to(device)

    mask_anchor_tensor = to_tensor(mask_anchor).unsqueeze(0).to(device)
    fg_mask = get_mask(mask_anchor_tensor, resize=(hw, hw))
    fg_mask_3ch = fg_mask.expand(-1, 3, -1, -1)

    if fg_image.sum().item() == 0:
        fg_depth_v1 = torch.zeros(1, 3, hw, hw).to(device)
        fg_depth_v2 = torch.zeros(1, 3, hw, hw).to(device)
        fg_depth_v3 = torch.zeros(1, 3, hw, hw).to(device)
    else:
        fg_image_tensor = to_tensor(fg_image).unsqueeze(0).to(device)
        fg_depth_v1 = get_depth(fg_image_tensor)
        fg_depth_v2 = get_depth(fg_image_tensor * fg_mask_3ch)
        fg_depth_v3 = fg_depth_v1 * fg_mask_3ch

    if bg_image.sum().item() == 0:
        bg_depth = torch.zeros(1, 3, hw, hw).to(device)
    else:
        bg_image_tensor = to_tensor(bg_image).unsqueeze(0).to(device)
        bg_depth = get_depth(bg_image_tensor)

    if depth_version=="v1":
        fg_depth = fg_depth_v1.clone()
    elif depth_version=="v2":
        fg_depth = fg_depth_v2.clone()
    elif depth_version=="v3":
        fg_depth = fg_depth_v3.clone()

    fg_depth = torch.clamp(fg_depth * fg_depth_control, min=0.0, max=1.0)
    bg_depth = torch.clamp(bg_depth * bg_depth_control, min=0.0, max=1.0)
    
    full_prompt = prompt + ' ,masterpiece, ultra detailed, high resolution'

    c_cross = model.get_learned_conditioning(full_prompt)
    uc_cross = model.get_learned_conditioning(neg_prompt)
    cond = {"bg_depth": [bg_depth.repeat(batch_size, 1, 1, 1)],
            "fg_depth": [fg_depth.repeat(batch_size, 1, 1, 1)],
            "bg_emb": [bg_emb.repeat(batch_size, 1)],
            "fg_emb": [fg_emb.repeat(batch_size, 1)],
            "c_crossattn": [c_cross.repeat(batch_size, 1, 1)],
            "mask": [fg_mask.repeat(batch_size, 1, 1, 1)]}

    un_cond = {"bg_depth": [bg_depth.repeat(batch_size, 1, 1, 1)],
            "fg_depth": [fg_depth.repeat(batch_size, 1, 1, 1)],
            "bg_emb": [bg_emb.repeat(batch_size, 1)],
            "fg_emb": [fg_emb.repeat(batch_size, 1)],
            "c_crossattn": [uc_cross.repeat(batch_size, 1, 1)],
            "mask": [fg_mask.repeat(batch_size, 1, 1, 1)]}
    
    shape = (4, H // 8, W // 8)

    samples, _ = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False,
                                                 unconditional_guidance_scale=cfg_scale,
                                                 unconditional_conditioning=un_cond,
                                                 global_strength=[fg_global_control, bg_global_control])


    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    results = [x_samples[i] for i in range(batch_size)]

    fg_depth_array = (einops.rearrange(fg_depth, 'b c h w -> b h w c') * 255.0).cpu().numpy().clip(0, 255).astype(np.uint8)[0]
    bg_depth_array = (einops.rearrange(bg_depth, 'b c h w -> b h w c') * 255.0).cpu().numpy().clip(0, 255).astype(np.uint8)[0]
    fg_mask_array = (einops.rearrange(fg_mask_3ch, 'b c h w -> b h w c') * 255.0).cpu().numpy().clip(0, 255).astype(np.uint8)[0]

    if fg_emb_image.sum().item() != 0:
        fg_emb_array = add_gridlines(fg_emb_image)
        bg_emb_array = add_gridlines(bg_emb_image)
    else:
        fg_emb_array = np.zeros((768,768,3))
        bg_emb_array = np.zeros((768,768,3))

    grid = np.concatenate(results,axis=1)
    
    conditions = [fg_depth_array, bg_depth_array, fg_emb_array, bg_emb_array, fg_mask_array, grid]

    return [results, conditions]

with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("## Compose and Conquer Demo")

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                fg_image = gr.Image(source='upload', type="numpy", label='Foreground Image', value="./samples/demo/247.jpg")
                bg_image = gr.Image(source='upload', type="numpy", label='Background Image', value="./samples/demo/239.jpg")
            with gr.Row():
                fg_emb_image = gr.Image(source='upload', type="numpy", label='Foreground Embedding Image', value="./samples/demo/104.jpg")
                bg_emb_image = gr.Image(source='upload', type="numpy", label='Background Embedding Image', value="./samples/demo/324.jpg")

            depth_version = gr.Radio(["v1", "v2", "v3"], label="Foreground Depth Version", value="v2", interactive=True)
            fg_depth_control = gr.Slider(label="Foreground Depth Controller", minimum=0, maximum=2, value=1, step=0.1)
            bg_depth_control = gr.Slider(label="Background Depth Controller", minimum=0, maximum=2, value=1, step=0.1)
            fg_global_control = gr.Slider(label="Foreground Global Control", minimum=0, maximum=2, value=1, step=0.1)
            bg_global_control = gr.Slider(label="Background Global Control", minimum=0, maximum=2, value=1, step=0.1)
            hw = gr.Slider(label="Image size", minimum=512, maximum=1024, value=768, step=256)
            batch_size = gr.Slider(label="Batch size", minimum=1, maximum=12, value=1, step=1)
            prompt = gr.Textbox(label="Prompt", value='a robot astronaut in front of a cyberpunk train')
            neg_prompt = gr.Textbox(label="Negative Prompt", value=DEFAULT_NEGATIVE_PROMPT)

            ddim_steps = gr.Slider(label="Sampling steps", minimum=1, maximum=100, value=50, step=1)
            cfg_scale = gr.Slider(label="CFG Scale", minimum=1, maximum=20, value=7.5, step=0.5)
            seed = gr.Slider(label="Seed", minimum=-1, maximum=65536, value=-1, step=1)

            run_button = gr.Button(label="Run")

        with gr.Column(scale=1):
            image_gallery = gr.Gallery(label='Output', preview=True)
            cond_gallery = gr.Gallery(label='Output', preview=True)

    inputs = [fg_image,
              bg_image,
              fg_emb_image,
              bg_emb_image,
              depth_version,
              fg_depth_control,
              bg_depth_control,
              fg_global_control,
              bg_global_control,
              hw,
              batch_size,
              prompt,
              neg_prompt,
              ddim_steps,
              cfg_scale,
              seed,
            ]

    run_button.click(fn=process, inputs=inputs, outputs=[image_gallery, cond_gallery])

demo.queue().launch(server_name='127.0.0.1')