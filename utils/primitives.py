import sys
if './' not in sys.path:
	sys.path.append('./')
import random
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from annotator.u2net.models import U2NET
from typing import Tuple, Union

class GetDepthMap(object):

    '''
    when called, input image should be a normalized torch tensor of shape [B, 3, H, W]
    '''
    
    def __init__(self, device, model_type=None):
        self.model_type = "DPT_Large" if model_type == None else model_type
        self.device = device
        self.midas_model = torch.hub.load("intel-isl/MiDaS", self.model_type).to(self.device)
        self.midas_model = self.midas_model.eval()
        for param in self.midas_model.parameters():
            param.requires_grad = False

    def __call__(self, image):
        with torch.no_grad():
            depth = self.midas_model(image)
            depth = depth.unsqueeze(1).expand(-1, 3, -1, -1).clone()
            depth -= torch.min(depth)
            depth /= torch.max(depth)

            torch.clamp(depth, 0, 1)

        return depth
    
class GetCLIPImageEmbeddings(object):

    '''
    when called, input should be either a single PIL Image, an unnormalized np.uint8 array,
    or an unnormalized torch.uint8 tensor, shapes of [b,h,w,c] or [h,w,c].
    '''

    def __init__(self, device, version=None):
        self.version = "openai/clip-vit-large-patch14" if version == None else version
        self.model = CLIPVisionModelWithProjection.from_pretrained(self.version)
        self.processor = AutoProcessor.from_pretrained(self.version)
        self.device = device
        self.model = self.model.to(self.device)
        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def __call__(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        image_embedding = outputs.image_embeds
        return image_embedding
    
class GetSODMask(object):

    '''
    when called, input image should be a normalized torch tensor of shape [B, 3, 320, 320]
    set threshold to None for non binary values
    '''

    def __init__(self, ckpt_path=None, device=None,):
        self.ckpt_path = 'annotator/u2net/weights/u2net.pth' if ckpt_path==None else ckpt_path
        self.device = torch.device('cuda') if device==None else device
        self.ckpt = torch.load(self.ckpt_path, map_location=self.device)

        self.u2net = U2NET(3,1).to(self.device)
        self.u2net.load_state_dict(self.ckpt)
        self.u2net.eval()

        self.transform = transforms.Compose([
            transforms.Resize(320, transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])

    def __call__(self, input_image, threshold:float=0.5, resize:Union[int, Tuple[int, int]]=512):
        if input_image.shape[-1] != 320:
            input_image = F.interpolate(input_image, size=(320,320), mode='bilinear', recompute_scale_factor=False, align_corners=False)
        with torch.no_grad():
            mask,d2,d3,d4,d5,d6,d7= self.u2net(input_image)
        del d2,d3,d4,d5,d6,d7

        if isinstance(threshold, float):
            mask = torch.clamp(mask,0,1)
            ones = torch.ones_like(mask)
            zeros = torch.zeros_like(mask)
            mask = torch.where(mask>threshold, ones, zeros)
        
        if isinstance(resize, int):
            mask = transforms.Resize((resize, resize), transforms.InterpolationMode.BILINEAR)(mask)
        elif isinstance(resize, tuple) and len(resize) == 2:
            mask = transforms.Resize(resize, transforms.InterpolationMode.BILINEAR)(mask)
        else:
            raise AssertionError("Check the resize params.")
            
        return mask

class GetColorPalette(object): #NOTE: unused for final ver.
    def __init__(self, downscale_factor=64):
        self.downscale_factor = downscale_factor

    def __call__(self, image):
        h, w = image.shape[2:]
        color = F.interpolate(image, size=(h // self.downscale_factor, w // self.downscale_factor), mode='bilinear', align_corners=False)
        color = F.interpolate(color, size=(h, w), mode='nearest')
        return color
    
class GetGrayscale(object): #NOTE: unused for final ver.
    def __init__(self):
        weights_list = [
            torch.tensor([0.299, 0.587, 0.114]),  # standard grayscale
            torch.tensor([0.333, 0.333, 0.333]),  # equal weights
            torch.tensor([0.114, 0.587, 0.299]),  # inverted standard
            torch.tensor([0.5, 0.25, 0.25]),  # heavy red, lighter green and blue
            torch.tensor([0.25, 0.25, 0.5])  # lighter red and green, heavy blue
        ]
        self.weights_list = weights_list

    def __call__(self, image):
        weights = random.choice(self.weights_list)
        weights = weights[None, :, None, None]
        grayscale = (image * weights).sum(dim=1, keepdim=True)
        grayscale = torch.clamp(grayscale, 0, 1)

        return grayscale
