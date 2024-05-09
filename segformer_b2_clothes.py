import os
import numpy as np
from urllib.request import urlopen
import torchvision.transforms as transforms  
import folder_paths
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image,ImageOps, ImageFilter
import torch.nn as nn
import torch

comfy_path = os.path.dirname(folder_paths.__file__)
custom_nodes_path = os.path.join(comfy_path, "custom_nodes")


# 指定本地分割模型文件夹的路径
model_folder_path = os.path.join(custom_nodes_path,"Comfyui_segformer_b2_clothes","checkpoints","segformer_b2_clothes")

processor = SegformerImageProcessor.from_pretrained(model_folder_path)
model = AutoModelForSemanticSegmentation.from_pretrained(model_folder_path)

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# 切割服装
def get_segmentation(tensor_image):
    cloth = tensor2pil(tensor_image)
    # 预处理和预测
    inputs = processor(images=cloth, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits.cpu()
    upsampled_logits = nn.functional.interpolate(logits, size=cloth.size[::-1], mode="bilinear", align_corners=False)
    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
    return pred_seg,cloth


def create_black_image():
    image = Image.fromarray(np.zeros((512, 512), dtype=np.uint8))
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# 根据标签生成遮罩
def get_mask(pred_seg, labels_to_keep):
    mask = np.isin(pred_seg, labels_to_keep).astype(np.uint8)
    # 创建agnostic-mask图像
    mask_image = Image.fromarray(mask * 255)
    mask_image = mask_image.convert("RGB")
    mask_image = pil2tensor(mask_image)
    return mask_image

class segformer_b2_clothes:
   
    def __init__(self):
        pass
    
    # Labels: 0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses", 4: "Upper-clothes", 5: "Skirt", 6: "Pants", 7: "Dress", 8: "Belt", 9: "Left-shoe", 10: "Right-shoe", 11: "Face", 12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm", 16: "Bag", 17: "Scarf"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {     
                 "image":("IMAGE", {"default": "","multiline": False}),

                 "Face": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                 "Hat": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                 "Hair": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                 "Upper_clothes": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                 "Skirt": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                 "Pants": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                 "Dress": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                 "Belt": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                 "shoe": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                 "leg": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                 "arm": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                 "Bag": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                 "Scarf": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                },
                "optional":
                {
                "mask_selection_A": ("LIST",),
                "mask_selection_B": ("LIST",),
                "mask_selection_C": ("LIST",),
                "mask_selection_D": ("LIST",),
                "mask_selection_E": ("LIST",),
                "mask_selection_F": ("LIST",),
                "mask_selection_G": ("LIST",),
                "mask_selection_H": ("LIST",),
                "mask_selection_I": ("LIST",),
                }
        }

    RETURN_TYPES = ("IMAGE","IMAGE","IMAGE","IMAGE","IMAGE","IMAGE","IMAGE","IMAGE","IMAGE","IMAGE",)
    RETURN_NAMES = ("mask_image","mask_A","mask_B","mask_C","mask_D","mask_E","mask_F","mask_G","mask_H","mask_I")
    #RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("mask_image",)
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "CXH"

    def sample(self,image,Face,Hat,Hair,Upper_clothes,Skirt,Pants,Dress,Belt,shoe,leg,arm,Bag,Scarf,mask_selection_A=[0],mask_selection_B=[0],mask_selection_C=[0],mask_selection_D=[0],mask_selection_E=[0],mask_selection_F=[0],mask_selection_G=[0],mask_selection_H=[0],mask_selection_I=[0]):
        mask_images = []
        mask_image = None
        for item in image:
            # seg切割结果，衣服pil
            pred_seg,cloth = get_segmentation(item)
            labels_to_keep = [0]
            # if background :
            #     labels_to_keep.append(0)
            if not Hat:
                labels_to_keep.append(1)
            if not Hair:
                labels_to_keep.append(2)
            if not Upper_clothes:
                labels_to_keep.append(4)
            if not Skirt:
                labels_to_keep.append(5)
            if not Pants:
                labels_to_keep.append(6)
            if not Dress:
                labels_to_keep.append(7)
            if not Belt:
                labels_to_keep.append(8)
            if not shoe:
                labels_to_keep.append(9)
                labels_to_keep.append(10)
            if not Face:
                labels_to_keep.append(11)
            if not leg:
                labels_to_keep.append(12)
                labels_to_keep.append(13)
            if not arm:
                labels_to_keep.append(14) 
                labels_to_keep.append(15) 
            if not Bag:
                labels_to_keep.append(16)
            if not Scarf:
                labels_to_keep.append(17)



            for mask_selection in [mask_selection_A, mask_selection_B, mask_selection_C,
                                   mask_selection_D,
                                   mask_selection_E, mask_selection_F, mask_selection_G, mask_selection_H,
                                   mask_selection_I]:
                if mask_selection != [0]:
                    mask_image = get_mask(pred_seg, mask_selection)
                else:
                    mask_image = create_black_image()
                mask_images.append(mask_image)

            mask_image = get_mask(pred_seg, labels_to_keep)
        mask_A, mask_B, mask_C, mask_D, mask_E, mask_F, mask_G, mask_H, mask_I = tuple(mask_images)

        return (mask_image, mask_A, mask_B, mask_C, mask_D, mask_E, mask_F, mask_G, mask_H, mask_I,)
    # mask_selection是指定的labels_to_keep合集，完善代码。当labels_to_keep为非空列表时，输出对应的结果。如果为空，则输出512 * 512像素的黑色图片。


class segformer_b2_mask_selection:
    def __init__(self):
        pass
    # Labels: 0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses", 4: "Upper-clothes", 5: "Skirt", 6: "Pants", 7: "Dress", 8: "Belt", 9: "Left-shoe", 10: "Right-shoe", 11: "Face", 12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm", 16: "Bag", 17: "Scarf"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
            {
                "Face": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "Hat": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "Hair": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "Upper_clothes": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "Skirt": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "Pants": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "Dress": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "Belt": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "shoe": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "leg": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "arm": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "Bag": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "Scarf": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("mask_selection",)
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "CXH"

    def sample(self, Face, Hat, Hair, Upper_clothes, Skirt, Pants, Dress, Belt, shoe, leg, arm, Bag, Scarf,):

        labels_to_keep = [0]
        # if background :
        #     labels_to_keep.append(0)
        if not Hat:
            labels_to_keep.append(1)
        if not Hair:
            labels_to_keep.append(2)
        if not Upper_clothes:
            labels_to_keep.append(4)
        if not Skirt:
            labels_to_keep.append(5)
        if not Pants:
            labels_to_keep.append(6)
        if not Dress:
            labels_to_keep.append(7)
        if not Belt:
            labels_to_keep.append(8)
        if not shoe:
            labels_to_keep.append(9)
            labels_to_keep.append(10)
        if not Face:
            labels_to_keep.append(11)
        if not leg:
            labels_to_keep.append(12)
            labels_to_keep.append(13)
        if not arm:
            labels_to_keep.append(14)
            labels_to_keep.append(15)
        if not Bag:
            labels_to_keep.append(16)
        if not Scarf:
            labels_to_keep.append(17)
        print(labels_to_keep)
        return (labels_to_keep,)
    # mask_selection是指定的labels_to_keep合集，完善代码。当labels_to_keep为非空列表时，输出对应的结果。如果为空，则输出512 * 512像素的黑色图片。