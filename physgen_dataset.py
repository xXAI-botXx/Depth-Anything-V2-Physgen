"""
PhysGen Dataset

See:
- https://huggingface.co/datasets/mspitzna/physicsgen
- https://arxiv.org/abs/2503.05333
- https://github.com/physicsgen/physicsgen
"""
import os
from PIL import Image

from datasets import load_dataset

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
# import torchvision.transforms as transforms
from torchvision import transforms

# def resize_to_divisible_by_14(image: np.ndarray) -> np.ndarray:
#     """
#     Resize an image to the next smaller width and height that is divisible by 14.
    
#     Parameters:
#         image (np.ndarray): Input image (H x W x C or H x W).
        
#     Returns:
#         np.ndarray: Resized image.
#     """
#     original_channels, original_height, original_width = image.shape[:]

#     new_width = original_width - (original_width % 14)
#     new_height = original_height - (original_height % 14)

#     resized_image = cv2.resize(image, (original_channels, new_width, new_height), interpolation=cv2.INTER_AREA)
#     return resized_image

def resize_tensor_to_divisible_by_14(tensor: torch.Tensor) -> torch.Tensor:
    """
    Resize a tensor to the next smaller (H, W) divisible by 14.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (C, H, W) or (B, C, H, W)
    
    Returns:
        torch.Tensor: Resized tensor
    """
    if tensor.dim() == 3:
        c, h, w = tensor.shape
        new_h = h - (h % 14)
        new_w = w - (w % 14)
        return F.interpolate(tensor.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
    
    elif tensor.dim() == 4:
        b, c, h, w = tensor.shape
        new_h = h - (h % 14)
        new_w = w - (w % 14)
        return F.interpolate(tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
    
    else:
        raise ValueError("Tensor must be 3D (C, H, W) or 4D (B, C, H, W)")


class PhysGenDataset(Dataset):

    def __init__(self, variation="sound_baseline", mode="train"):
        """
        Loads PhysGen Dataset.

        Parameters:
        - variation : str
            Chooses the used dataset variant: sound_baseline, sound_reflection, sound_diffraction, sound_combined.
        - mode : str
            Can be "train", "test", "eval".
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        # get data
        self.dataset = load_dataset("mspitzna/physicsgen", name=variation, trust_remote_code=True)
        self.dataset = self.dataset[mode]
        # self.dataset = dataset

        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Converts [0,255] PIL image to [0,1] FloatTensor
        ])
        print(f"PhysGen ({variation}) Dataset for {mode} got created")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        # print(sample)
        # print(sample.keys())
        input_img = sample["osm"]  # PIL Image
        target_img = sample["soundmap"]  # PIL Image

        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        # Fix real image size 512x512 > 256x256
        input_img = F.interpolate(input_img.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)
        input_img = input_img.squeeze(0)
        # target_img = target_img.unsqueeze(0)

        # change size
        input_img = resize_tensor_to_divisible_by_14(input_img)
        target_img = resize_tensor_to_divisible_by_14(target_img)

        # add fake rgb
        if input_img.shape[0] == 1:  # shape (B, 1, H, W)
            input_img = input_img.repeat(3, 1, 1)  # make it (B, 3, H, W)

        # add fake rgb
        # if target_img.shape[0] == 1:  # shape (B, 1, H, W)
        #     target_img = target_img.repeat(3, 1, 1)  # make it (B, 3, H, W)

        # print(input_img.shape)
        # print(target_img.shape)

        return input_img, target_img, idx





