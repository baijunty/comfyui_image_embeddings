import json
import torch
import numpy as np
from PIL import Image, ImageOps
import requests
import folder_paths
import os
from io import BytesIO
from torchvision import transforms

def process_image(img,name):
    output_images = []
    output_masks = []
    w, h = None, None
    excluded_formats = ['MPO']
    for i in Image.Spin(img) if hasattr(Image, 'Spin') else [img]:
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

class CustomImageLoader:
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
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return process_image(img,url.split('/')[-1])

    def load_image_from_path(self, image_path):
        # 检查是否为相对路径或绝对路径
        if not os.path.isabs(image_path):
            # 如果不是绝对路径，尝试在ComfyUI的input目录中查找
            input_dir = folder_paths.get_input_directory()
            image_path = os.path.join(input_dir, image_path)
        
        # 检查是否为目录
        if os.path.isdir(image_path):
            # 获取目录中的所有图片文件
            image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
            image_files = []
            for file in os.listdir(image_path):
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in image_extensions:
                    image_files.append(os.path.join(image_path, file))
            
            # 按文件名排序，保证顺序一致
            image_files.sort()
            
            # 加载所有图片
            all_images = []
            all_masks = []
            all_names = []
            for img_path in image_files:
                img = Image.open(img_path)
                processed_img, processed_mask,_ = process_image(img,'')
                all_images.append(processed_img)
                all_masks.append(processed_mask)
                all_names.append(os.path.basename(img_path))
            
            return (all_images, all_masks,all_names)
        else:
            # 原来的单个文件处理逻辑
            img = Image.open(image_path)
            return process_image(img,os.path.basename(image_path))


    @classmethod
    def IS_CHANGED(s, image_path_or_url):
        return float("NaN") if image_path_or_url.startswith(('http://', 'https://')) else image_path_or_url


class Image2Base64:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "names": ("STRING",), },
        }
    CATEGORY = "utils"
    RETURN_TYPES = ('STRING',)
    FUNCTION = "image_to_base64"
    def image_to_base64(self, images,names):
        """
        将输入的图像张量转换为base64编码，并按名称:编码内容的格式输出JSON字符串
        """
        
        # 创建结果字典
        result_dict = {}
        file_names=[]
        import base64
        if isinstance(names, list):
            file_names = names
        else:
            file_names = [names]
        # 遍历图像批次
        for index, image in enumerate(images):
            
            # 转换图像张量为PIL图像
            n = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(n, 0, 255).astype(np.uint8))
            
            # 将PIL图像转换为base64
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            # 添加到结果字典
            result_dict[file_names[index]] = img_base64
        
        # 将结果字典转换为JSON字符串
        json_string = json.dumps(result_dict)
        
        return (json_string,)

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

class Resize2DivisibleImage:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "name": ("STRING",),
                "target": ("INT", {"default": 32, "min": 2, "max": 2048, "step": 1}),
                "max_size": ("INT", {"default": 1024, "min": 8, "max": 8192, "step": 1}),  # 0 表示不限制
            },
        }
    CATEGORY = "utils"
    RETURN_TYPES = ('IMAGE','STRING',)
    FUNCTION = "divisible_resize_image"

    def resize_image(self, image,target, max_size=0):
        """
        将图像的宽高调整到能被target整除的尺寸（保持宽高比例）
        
        Args:
            image: ComfyUI的IMAGE张量 [Batch, Height, Width, Channels]
            target: 目标整除数
        
        Returns:
            调整后的图像
        """
        # 获取原始尺寸 [B, H, W, C]
        b, h, w, c = image.shape

        # 计算保持宽高比例的缩放因子，若提供 max_size 则限制最大尺寸
        scale = 1.0
        if max_size > 0:
            scale = min(max_size / h, max_size / w, 1.0)

        # 根据缩放因子计算新的尺寸，并取能被 target 整除的最大尺寸（向下取整）
        new_h = int((h * scale) // target * target)
        new_w = int((w * scale) // target * target)

        # 防止尺寸为 0
        if new_h == 0:
            new_h = target
        if new_w == 0:
            new_w = target

        # 如果尺寸未变化，直接返回原图
        if new_h == h and new_w == w:
            return image

        # 将 [B, H, W, C] 转换为 [B, C, H, W] 进行卷积操作
        image_permuted = image.permute(0, 3, 1, 2)

        import torch.nn.functional as F
        resized = F.interpolate(
            image_permuted,
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=True
        )

        # 转换回 [B, H, W, C]
        resized = resized.permute(0, 2, 3, 1).squeeze(0)
        return resized

    def divisible_resize_image(self, image, name, target, max_size=0):
        """
        将图像的宽高调整到能被target整除的尺寸
        
        Args:
            image: ComfyUI的IMAGE张量 [Batch, Height, Width, Channels]
            name: 图像名称
            target: 目标整除数
        
        Returns:
            调整后的图像和名称
        """
        images = []
        if isinstance(image, list):
            images = image
        else:
            images = [image]
        resized = []
        for img in images:
            resized.append(self.resize_image(img, target, max_size))
        return (resized, name)

class Base64ImageLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base64_string": ("STRING",),
                "name": ("STRING",), },
        }
    CATEGORY = "utils"
    RETURN_TYPES =  ("IMAGE", "MASK","STRING",)
    FUNCTION = "load_base64_image"
    def load_base64_image(self, base64_string,name):
        """
        将base64编码的图像数据转换为PIL图像对象。

        Args:
            base64_string (str): base64编码的图像数据。

        Returns:
            PIL.Image.Image: 转换后的PIL图像对象。
        """
        import base64
        from io import BytesIO
        # 将base64编码的字符串解码为字节流
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_bytes))
        return process_image(image,name)
    @classmethod
    def IS_CHANGED(s,base64_string,name):
        return float("NaN")

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "CustomImageLoader": CustomImageLoader,
    "OutputEmbedding": VisionOutputEmbedding2JSON,
    "ImageHash": ImageHashNode,
    "Image2Base64": Image2Base64,
    "Resize2DivisibleImage": Resize2DivisibleImage,
    "Base64ImageLoader": Base64ImageLoader,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "CustomImageLoader": "Image Loader (Path or URL)",
    "OutputEmbedding": "Output Embedding to JSON",
    "ImageHash": "Image Hash",
    "Image2Base64": "Image to Base64",
    "Base64ImageLoader": "Base64 Image Loader",
    "Resize2DivisibleImage": "Resize to visible Image",
}
