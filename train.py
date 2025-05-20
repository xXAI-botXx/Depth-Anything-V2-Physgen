import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from physgen_dataset import PhysGenDataset
from depth_anything_v2.dpt import DepthAnythingV2
from inference import inference_forward

import numpy as np
from tqdm import tqdm
import wandb
import torchvision.utils as vutils



def train(variation, model_name, encoder, batch_size, epochs, lr):
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Prepare dataset
    train_dataset = PhysGenDataset(mode='train', variation=variation)
    val_dataset = PhysGenDataset(mode='validation', variation=variation)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Model configuration
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    model = DepthAnythingV2(**model_configs[encoder]).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.L1Loss()

    # Initialize Weights & Biases
    wandb.init(project="Master-PhysGen", name=model_name, config={
        "encoder": encoder,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "variation": variation
    })
    wandb.watch(model, log="all")

    best_model = None
    best_loss = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_img, target_depth, _ = batch
            input_img, target_depth = input_img.to(device), target_depth.to(device)

            

            optimizer.zero_grad()
            pred_depth = model(input_img)
            # print(pred_depth.shape)
            # print(target_depth.shape)
            loss = criterion(pred_depth, target_depth) * 1000
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        wandb.log({"train_loss": avg_train_loss, "epoch": epoch + 1})

        # Validation
        model.eval()
        val_loss = 0.0
        val_img_log = None
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                input_img, target_depth, _ = batch
                input_img, target_depth = input_img.to(device), target_depth.to(device)
                pred_depth = model(input_img)
                loss = criterion(pred_depth, target_depth) * 1000
                val_loss += loss.item()

                if i == 0:
                    # Log first batch images
                    val_img_log = inference_forward(input_img, model, device)


        avg_val_loss = val_loss / len(val_loader)
        wandb.log({
            "val_loss": avg_val_loss,
            "epoch": epoch + 1,
            "sample_depth_map": wandb.Image(val_img_log) if val_img_log is not None else None
        })

        # Save model
        if not best_model:
            best_model = f"./checkpoints/{args.model_name}_epoch{epoch+1}.pth"
            best_loss = avg_val_loss
        elif best_loss > avg_val_loss:
            os.remove(best_model)

            best_model = f"./checkpoints/{args.model_name}_epoch{epoch+1}.pth"
            best_loss = avg_val_loss
        else:
            continue
        
        os.makedirs("./checkpoints", exist_ok=True)
        save_path = best_model
        torch.save(model.state_dict(), save_path)
        print(f"Saved model at {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Depth Anything with wandb")
    parser.add_argument("--variation", help="Dataset variant: sound_baseline, sound_reflection, sound_diffraction, sound_combined")
    parser.add_argument("--model_name", help="Name for saving the model checkpoint")
    parser.add_argument("--encoder", default="vitb", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train(args.variation, args.model_name, args.encoder, args.batch_size, args.epochs, args.lr)


