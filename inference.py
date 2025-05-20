import argparse
import os
import shutil
import re

import numpy as np
import cv2

from physgen_dataset import PhysGenDataset
from depth_anything_v2.dpt import DepthAnythingV2

import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils



def get_newest_model(model_name:str, path="./checkpoints"):
    # found_models = []
    newest_model = None
    epoch_of_newest_model = None
    for model in os.listdir(path):
        if model_name in model:    # and end with pth?
            # found_models += [model]

            if not newest_model:
                newest_model = model
                epoch_of_newest_model = int(re.findall(r'\d+', string=model)[-1])
            else:
                epoch = int(re.findall(r'\d+', string=model)[-1])
                if epoch > epoch_of_newest_model:
                    newest_model = model
                    epoch_of_newest_model = epoch

    # newest_model = sorted(found_models, key=lambda x:int(x.split(".")[0].split("epoch")[-1]))[-1]
    print(f"found newest model: {newest_model}")
    return newest_model

def normalize_depth(depth):
    # Normalize depth to [0, 1] range
    depth_min = depth.min()
    depth_max = depth.max()
    return (depth - depth_min) / (depth_max - depth_min + 1e-8)

def inference_forward(input_img, model, device):
    input_img = input_img.to(device)
    pred = model(input_img)
    
    # Normalize
    pred = normalize_depth(pred).cpu()

    # Combine Patches
    pred = vutils.make_grid(pred, normalize=True)

    # Get RGB -> Gray
    # Weighting after Luma-Formel: 0.299*R + 0.587*G + 0.114*B
    pred = 0.299 * pred[0] + 0.587 * pred[1] + 0.114 * pred[2]
    pred = pred.unsqueeze(-1)  # Shape: (252, 252, 1)
    # pred = pred[2, :, :].unsqueeze(-1)    # -> just one Channel

    # To Numpy
    pred = pred.detach().numpy()

    # Value Upscaling
    pred = pred * 256

    # Invert
    # pred = np.abs(pred-256)

    return pred

def inference_method(input_img, model, size):
    input_img = input_img.permute((0, 2, 3, 1))
    input_img = input_img.squeeze(0)

    pred = model.infer_image(input_img.detach().numpy(), size)
    
    pred = (pred - pred.min()) / (pred.max() - pred.min()) * 255.0
    pred = pred.astype(np.uint8)

    return pred

def inference(variation, model_name, encoder):
    exact_model_name = get_newest_model(model_name, path="./checkpoints")
    model_path = f"./checkpoints/{exact_model_name}"
    output_path = f"./eval/{model_name}"

    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    # load data
    dataset = PhysGenDataset(mode="test", variation=variation)
    data_len = len(dataset)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # load model
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    phys_anything = DepthAnythingV2(**model_configs[encoder])
    phys_anything.load_state_dict(torch.load(model_path, map_location='cpu'))
    phys_anything = phys_anything.to(DEVICE).eval()

    for i, data in enumerate(dataloader):
        print(f'Progress {i+1}/{data_len}')
        
        input_img, target_img, idx = data
        idx = idx[0].item() if isinstance(idx, torch.Tensor) else idx

        # infer_img = inference_method(input_img, phys_anything, target_img.shape[2])
        forward_img = inference_forward(input_img, phys_anything, DEVICE)
        
        print(f"Prediction shape [infer_image]: {infer_img.shape}")
        print(f"Prediction shape [forward]: {forward_img.shape}")
        
        # phys = np.repeat(phys[..., np.newaxis], 3, axis=-1)
        
        os.makedirs(output_path, exist_ok=True)
        save_img = os.path.join(output_path, f"{model_name}_{idx}.png")
        cv2.imwrite(save_img, forward_img)
        print(f"    -> saved pred at {save_img}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference arguments")
    parser.add_argument("--variation", help="Chooses the used dataset variant: sound_baseline, sound_reflection, sound_diffraction, sound_combined.")
    parser.add_argument("--model_name", help="Name of the model (without .pth) in the ./checkpoints folder.")
    parser.add_argument("--encoder", default="vitb", choices=["vits", "vitb", "vitl", "vitg"])
    args = parser.parse_args()

    inference(variation=args.variation, model_name=args.model_name, encoder=args.encoder)






