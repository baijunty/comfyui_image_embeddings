import json
import torch
import numpy as np
from PIL import Image, ImageOps
import requests
import folder_paths
import os
from io import BytesIO
import hashlib


class ImageLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_path_or_url": ("STRING", {"default": "", "multiline": False}),
            }
        }

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    
    def load_image(self, image_path_or_url):
        # 检查是否为URL
        if image_path_or_url.startswith(('http://', 'https://')):
            return self.load_image_from_url(image_path_or_url)
        else:
            return self.load_image_from_path(image_path_or_url)

    def load_image_from_url(self, url):
        try:
            response = requests.get(url, timeout=90)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
        except Exception as e:
            print(f"Error loading image from URL {url}: {e}")
            # 返回一个空图像
            img = Image.new("RGB", (512, 512), color='black')
        
        return self.process_image(img)

    def load_image_from_path(self, image_path):
        # 检查是否为相对路径或绝对路径
        if not os.path.isabs(image_path):
            # 如果不是绝对路径，尝试在ComfyUI的input目录中查找
            input_dir = folder_paths.get_input_directory()
            image_path = os.path.join(input_dir, image_path)
        
        # 验证路径是否存在
        if not os.path.exists(image_path):
            print(f"Invalid image path: {image_path}")
            # 返回一个空图像
            img = Image.new("RGB", (512, 512), color='black')
        else:
            img = Image.open(image_path)
        
        return self.process_image(img)

    def process_image(self, img):
        # 处理图像，参考ComfyUI的LoadImage节点
        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in Image.Spin(img) if hasattr(Image, 'Spin') else [img]:
            # 如果是多帧图像，只处理第一帧
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            elif i.mode == 'P' and 'transparency' in i.info:
                mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

    @classmethod
    def IS_CHANGED(s, image_path_or_url):
        # 如果是URL，每次都重新加载，因为内容可能已更改
        return image_path_or_url

class VisionOutputEmbedding2JSON:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vision_output": ("CLIP_VISION_OUTPUT",),
            }
        }
    CATEGORY = "utils"
    RETURN_TYPES = ('STRING',)
    FUNCTION = "output_embedding_to_json"
    def output_embedding_to_json(self, vision_output):
        # 将tensor转换为numpy数组后再进行JSON编码
        image_embeds = vision_output['image_embeds']
        if torch.is_tensor(image_embeds):
            # 如果是PyTorch张量，转换为numpy数组
            image_embeds = image_embeds.squeeze().detach().cpu().numpy().tolist()
        elif isinstance(image_embeds, np.ndarray):
            # 如果已经是numpy数组，转换为列表
            image_embeds = image_embeds.tolist()
        return (json.dumps(image_embeds),)
    
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ImageLoader": ImageLoader,
    "OutputEmbedding": VisionOutputEmbedding2JSON
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageLoader": "Image Loader (Path or URL)",
    "OutputEmbedding": "Output Embedding to JSON"
}