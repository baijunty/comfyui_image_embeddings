import json
import torch
import numpy as np
from PIL import Image, ImageOps
import requests
import folder_paths
import os
from io import BytesIO
from torchvision import transforms


class ImageLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_path_or_url": ("STRING", {"default": "", "multiline": False}),
            }
        }

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", "MASK","STRING",)
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
        
        return self.process_image(img,url.split('/')[-1])

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
            return self.process_image(img,'')
        
        # 检查是否为目录
        if os.path.isdir(image_path):
            # 获取目录中的所有图片文件
            image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
            image_files = []
            for file in os.listdir(image_path):
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in image_extensions:
                    image_files.append(os.path.join(image_path, file))
            
            if not image_files:
                print(f"No image files found in directory: {image_path}")
                img = Image.new("RGB", (512, 512), color='black')
                return self.process_image(img,'')
            
            # 按文件名排序，保证顺序一致
            image_files.sort()
            
            # 加载所有图片
            all_images = []
            all_masks = []
            all_names = []
            for img_path in image_files:
                try:
                    img = Image.open(img_path)
                    processed_img, processed_mask,_ = self.process_image(img,'')
                    all_images.append(processed_img)
                    all_masks.append(processed_mask)
                    all_names.append(os.path.basename(img_path))
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
            
            if not all_images:
                print(f"No valid images could be loaded from directory: {image_path}")
                img = Image.new("RGB", (512, 512), color='black')
                return self.process_image(img,'')
            
            return (all_images, all_masks,all_names)
        else:
            # 原来的单个文件处理逻辑
            img = Image.open(image_path)
            return self.process_image(img,os.path.basename(image_path))

    def process_image(self, img,name):
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
        return (output_image, output_mask,name)

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
                "name": ("STRING",), },
        }
    CATEGORY = "utils"
    RETURN_TYPES = ('STRING',)
    FUNCTION = "output_embedding_to_json"
    def output_embedding_to_json(self, vision_output,name):
        # 将tensor转换为numpy数组后再进行JSON编码
        image_embeds = vision_output['image_embeds']
        if torch.is_tensor(image_embeds):
            # 如果是PyTorch张量，转换为numpy数组
            image_embeds = image_embeds.squeeze().detach().cpu().numpy().tolist()
        elif isinstance(image_embeds, np.ndarray):
            # 如果已经是numpy数组，转换为列表
            image_embeds = image_embeds.tolist()
        return (json.dumps({name: image_embeds}),)
def hex_to_signed(hex_str, bits):
    """
    将十六进制字符串转换为有符号整数。

    Args:
        hex_str (str): 十六进制字符串。
        bits (int): 位数，用于确定有符号整数的范围。

    Returns:
        int: 转换后的有符号整数值。

    Note:
        如果输入的十六进制值超出指定位数的有符号整数范围，可能会返回错误的结果。
    """
    unsigned_val = int(hex_str, 16)
    mask = (1 << (bits - 1))
    if unsigned_val & mask:
        return unsigned_val - (1 << bits)
    else:
        return unsigned_val
grayscale = transforms.Grayscale(num_output_channels=1)
resize = transforms.Resize((8, 8))
class ImageHashNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"images": ("IMAGE", ),
                "names": ("STRING",), },
        }
    CATEGORY = "utils"
    RETURN_TYPES = ('STRING',)
    FUNCTION = "image_hash"
    def image_hash(self, images,names):
        # 计算图像的哈希值
        hashes = {}
        from imagehash import ImageHash
        file_names=[]
        if isinstance(names, list):
            file_names = names
        else:
            file_names = [names]
        for i,image in enumerate(images):
            # ComfyUI的IMAGE张量通常是[Batch, Height, Width, Channels]格式
            # PyTorch transforms期望的是[Channels, Height, Width]格式
            
            # 确保我们处理的是[Height, Width, Channels]格式
            if len(image.shape) == 4:  # 如果是[B, H, W, C]格式，取第一个
                image = image.squeeze(0)  # 移除批次维度，得到[H, W, C]
            
            # 现在image应该是[H, W, C]格式
            if len(image.shape) == 3 and image.shape[-1] in [1, 3]:  # [Height, Width, Channels]
                tensor = image.permute(2, 0, 1)  # 转换为 [Channels, Height, Width]
            else:
                tensor = image
            tensor = resize(tensor)
            tensor = grayscale(tensor)
            mean = tensor.mean()
            binary_hash = (tensor > mean).to(torch.uint8)
            pixels = binary_hash.squeeze(0).numpy()
            mean = pixels.mean()
            diff = pixels > mean
            hash=hex_to_signed(str(ImageHash(diff)), 64)
            hashes.update({file_names[i]: hash})
        #list to str
        return (json.dumps(hashes),)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ImageLoader": ImageLoader,
    "OutputEmbedding": VisionOutputEmbedding2JSON,
    "ImageHash": ImageHashNode
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageLoader": "Image Loader (Path or URL)",
    "OutputEmbedding": "Output Embedding to JSON",
    "ImageHash": "Image Hash"
}