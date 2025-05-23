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

def inference_forward(input_img, model, device, scale_to_256=False):
    input_img = input_img.to(device)
    pred = model(input_img)
    
    # Normalize
    # print(f"Min: {pred.min()}, Max: {pred.max()}")
    pred = torch.clamp(pred, max=1.0)
    # pred = normalize_depth(pred).cpu()

    # Combine Patches
    pred = vutils.make_grid(pred, normalize=False)

    # Get RGB -> Gray
    # Weighting after Luma-Formel: 0.299*R + 0.587*G + 0.114*B
    pred = 0.299 * pred[0] + 0.587 * pred[1] + 0.114 * pred[2]
    pred = pred.unsqueeze(-1)  # Shape: (252, 252, 1)
    # pred = pred[2, :, :].unsqueeze(-1)    # -> just one Channel

    # To Numpy
    pred = pred.cpu().detach().numpy()

    # Value Upscaling
    if scale_to_256:
        pred = pred * 255

    # Invert
    # pred = np.abs(pred-255)

    return pred

def inference_method(input_img, model, size):
    input_img = input_img.permute((0, 2, 3, 1))
    input_img = input_img.squeeze(0)

    pred = model.infer_image(input_img.detach().numpy(), size)
    
    pred = (pred - pred.min()) / (pred.max() - pred.min()) * 255.0
    pred = pred.astype(np.uint8)

    return pred

def inference(variation, model_name, encoder, save_only_result=False):
    exact_model_name = get_newest_model(model_name, path="./checkpoints")
    model_path = f"./checkpoints/{exact_model_name}"
    
    # Create and clear output paths
    output_path = f"./eval/{model_name}"
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    output_pred_path = f"{output_path}/pred"
    if os.path.exists(output_pred_path):
        shutil.rmtree(output_pred_path)
    os.makedirs(output_pred_path, exist_ok=True)

    output_real_path = f"{output_path}/real"
    if os.path.exists(output_real_path):
        shutil.rmtree(output_real_path)
    os.makedirs(output_real_path, exist_ok=True)

    output_osm_path = f"{output_path}/osm"
    if os.path.exists(output_osm_path):
        shutil.rmtree(output_osm_path)
    os.makedirs(output_osm_path, exist_ok=True)

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
        
        # print(f"Prediction shape [infer_image]: {infer_img.shape}")
        # print(f"Prediction shape [forward]: {forward_img.shape}")
        # print(f"Prediction shape [osm]: {input_img.shape}")
        # print(f"Prediction shape [target]: {target_img.shape}")
        
        # print(f"OSM Info:\n    -> shape: {input_img.shape}\n    -> min: {input_img.min()}, max: {input_img.max()}")

        # phys = np.repeat(phys[..., np.newaxis], 3, axis=-1)

        # Transform to Numpy
        pred_img = forward_img.squeeze(2)
        if not (0 <= pred_img.min() <= 255 and 0 <= pred_img.max() <=255):
            raise ValueError(f"Prediction has values out of 0-256 range => min:{pred_img.min()}, max:{pred_img.max()}")
        if pred_img.max() <= 1.0:
            pred_img *= 255
        pred_img = pred_img.astype(np.uint8)

        real_img = target_img.squeeze(0).cpu().squeeze(0).detach().numpy()
        if not (0 <= real_img.min() <= 255 and 0 <= real_img.max() <=255):
            raise ValueError(f"Real target has values out of 0-256 range => min:{real_img.min()}, max:{real_img.max()}")
        if real_img.max() <= 1.0:
            real_img *= 255
        real_img = real_img.astype(np.uint8)

        if len(input_img.shape) == 4:
            osm_img = input_img[0, 0].cpu().detach().numpy()
        else:
            osm_img = input_img[0].cpu().detach().numpy()
        if not (0 <= osm_img.min() <= 255 and 0 <= osm_img.max() <=255):
            raise ValueError(f"Real target has values out of 0-256 range => min:{osm_img.min()}, max:{osm_img.max()}")
        if osm_img.max() <= 1.0:
            osm_img *= 255
        osm_img = osm_img.astype(np.uint8)

        # print(f"OSM Info:\n    -> shape: {osm_img.shape}\n    -> min: {osm_img.min()}, max: {osm_img.max()}")
        
        # Save Results
        file_name = f"{model_name}_{idx}.png"
        if save_only_result:
            save_img = os.path.join(output_path, file_name)
        else:
            save_img = os.path.join(output_pred_path, file_name)
        cv2.imwrite(save_img, pred_img)
        print(f"    -> saved pred at {save_img}")

        if not save_only_result:
            # save real image
            save_img = os.path.join(output_real_path, file_name)
            cv2.imwrite(save_img, real_img)   # works? right format?
            print(f"    -> saved real at {save_img}")

            # save osm image
            save_img = os.path.join(output_osm_path, file_name)
            cv2.imwrite(save_img, osm_img)   # works? right format?
            print(f"    -> saved osm at {save_img}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference arguments")
    parser.add_argument("--variation", help="Chooses the used dataset variant: sound_baseline, sound_reflection, sound_diffraction, sound_combined.")
    parser.add_argument("--model_name", help="Name of the model (without .pth) in the ./checkpoints folder.")
    parser.add_argument("--encoder", default="vitb", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--save_only_result", action='store_true')
    args = parser.parse_args()

    inference(variation=args.variation, 
              model_name=args.model_name, 
              encoder=args.encoder, 
              save_only_result=args.save_only_result
             )






