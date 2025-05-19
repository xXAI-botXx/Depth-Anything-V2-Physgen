import argparse
import os

import numpy as np
import cv2

from datasets import load_dataset
from data.physgen_dataset import PhysGenDataset

import torch





def inference(variation, model_path, encoder):
    model_path = f"./checkpoints/{model_name}.pth"
    output_path = f"./eval/{model_name}"

    # load data
    dataset = PhysGenDataset(mode="test")
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
        
        input_img, target_img, idx = cv2.imread(filename)
        
        phys = phys_anything.infer_image(input_img.detach().numpy(), target_img.shape[1])
        
        phys = (phys - phys.min()) / (phys.max() - phys.min()) * 255.0
        phys = phys.astype(np.uint8)
        
        phys = np.repeat(phys[..., np.newaxis], 3, axis=-1)
        
        os.makedirs(output_path, exists_ok=True)
        save_img = os.path.join(output_path, f"{model_name}_{idx}.png")
        cv2.imwrite(, phys)
        print(f"    -> saved pred at {save_img}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference arguments")
    parser.add_argument("--variation", help="Chooses the used dataset variant: sound_baseline, sound_reflection, sound_diffraction, sound_combined.")
    parser.add_argument("--model_name", help="Name of the model (without .pth) in the ./checkpoints folder.")
    parser.add_argument("--encoder", default="vitb", choices=["vits", "vitb", "vitl", "vitg"])
    args = parser.parse_args()

    inference(variation=args.variation, model_name=args.model_name, encoder=args.encoder)






