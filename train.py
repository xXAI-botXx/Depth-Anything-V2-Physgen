import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from physgen_dataset import PhysGenDataset
from depth_anything_v2.dpt import DepthAnythingV2
from inference import inference_forward

import numpy as np
from tqdm import tqdm
import wandb
import torchvision.utils as vutils

import kornia  # for SSIM and gradients

class CombinedLoss(nn.Module):
    def __init__(self, 
                 silog_lambda=0.5, 
                 weight_silog=1.0, 
                 weight_grad=1.0, 
                 weight_ssim=1.0,
                 weight_l1=1.0):
        super().__init__()
        self.silog_lambda = silog_lambda
        self.weight_silog = weight_silog
        self.weight_grad = weight_grad
        self.weight_ssim = weight_ssim
        self.weight_l1 = weight_l1

        self.init_weight_silog = self.weight_silog
        self.init_weight_grad = self.weight_grad
        self.init_weight_ssim = self.weight_ssim
        self.init_weight_l1 = self.weight_l1

        self.avg_loss_silog = 0
        self.avg_loss_grad = 0
        self.avg_loss_ssim = 0
        self.avg_loss_l1 = 0
        self.steps = 0

        # Instantiate SSIMLoss module
        self.ssim_module = kornia.losses.SSIMLoss(window_size=11, reduction='mean')

    def silog_loss(self, pred, target):
        eps = 1e-6
        pred = torch.clamp(pred, min=eps)
        target = torch.clamp(target, min=eps)
        
        diff_log = torch.log(target) - torch.log(pred)
        loss = torch.sqrt(torch.mean(diff_log ** 2) -
                          self.silog_lambda * torch.mean(diff_log) ** 2)
        return loss

    def gradient_l1_loss(self, pred, target):
        # Create Channel Dimension
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if target.ndim == 3:
            target = target.unsqueeze(1)

        # Gradient in x-direction (horizontal -> dim=3)
        pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]

        # Gradient in y-direction (vertical -> dim=2)
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]

        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)

        return loss_x + loss_y

    def ssim_loss(self, pred, target):
        # SSIM returns similarity, so we subtract from 1
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if target.ndim == 3:
            target = target.unsqueeze(1)

        return self.ssim_module(pred, target)

    def l1_loss(self, pred, target):
        loss = torch.abs(target - pred)
        return loss.mean()

    def forward(self, pred, target):
        loss_silog = self.silog_loss(pred, target)
        loss_grad = self.gradient_l1_loss(pred, target)
        loss_ssim = self.ssim_loss(pred, target)
        loss_l1 = self.l1_loss(pred, target)

        self.avg_loss_silog += loss_silog
        self.avg_loss_grad += loss_grad
        self.avg_loss_ssim += loss_ssim
        self.avg_loss_l1 += loss_l1
        self.steps += 1

        total_loss = (
            self.weight_silog * loss_silog +
            self.weight_grad * loss_grad +
            self.weight_ssim * loss_ssim +
            self.weight_l1 * loss_l1
        )
        return total_loss

    def step(self, epoch):
        self.avg_loss_silog = 0
        self.avg_loss_grad = 0
        self.avg_loss_ssim = 0
        self.avg_loss_l1 = 0
        self.steps = 0
        
        # if 5 < epoch < 50:
        #     new_adjustment = min(1.0*((epoch-5)/10), 1.0)
        #     self.weight_silog = self.init_weight_silog+new_adjustment
        #     self.weight_grad = self.init_weight_grad+new_adjustment
        #     self.weight_ssim = self.init_weight_ssim+new_adjustment
        #     self.weight_l1 = self.init_weight_l1+new_adjustment
        # elif epoch >= 50:
        #     new_adjustment = min(10.0*((epoch-50)/100), 10.0)
        #     self.weight_silog = self.init_weight_silog+new_adjustment
        #     self.weight_grad = self.init_weight_grad+new_adjustment
        #     self.weight_ssim = self.init_weight_ssim+new_adjustment
        #     self.weight_l1 = self.init_weight_l1+new_adjustment

    def get_avg_losses(self):
        return (self.avg_loss_silog/self.steps,
                self.avg_loss_grad/self.steps,
                self.avg_loss_ssim/self.steps,
                self.avg_loss_l1/self.steps)

# class SiLogLoss(nn.Module):
#     def __init__(self, lambd=0.5):
#         super().__init__()
#         self.lambd = lambd

#     def forward(self, pred, target):
#         diff_log = torch.log(target) - torch.log(pred)
#         loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
#                           self.lambd * torch.pow(diff_log.mean(), 2))

#         return loss

def train(variation, model_name, encoder, batch_size, epochs, lr):
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Prepare dataset
    train_dataset = PhysGenDataset(mode='train', variation=variation)
    val_dataset = PhysGenDataset(mode='validation', variation=variation)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    total_iters = epochs * len(train_loader)    # loop steps / optimizer steps, not every single image steps

    # Model configuration
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    model = DepthAnythingV2(**model_configs[encoder]).to(device)
    for param in model.parameters():
        param.requires_grad = True


    # model.depth_head.parameters()
    # lambda_loss = 0.5
    # criterion_1 = SiLogLoss(lambd=lambda_loss)
    criterion = CombinedLoss(silog_lambda=0.5, 
                             weight_silog=0.5, 
                             weight_grad=2.0, 
                             weight_ssim=1.0,
                             weight_l1=50.0)

    # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    start_lr_1 = 1e-8
    goal_lr_1 = lr*0.001
    start_lr_2 = lr*0.001
    goal_lr_2 = lr # * 10.0
    optimizer = optim.AdamW([
                             {'params': [param for name, param in model.named_parameters() if 'pretrained' in name], 'lr': start_lr_1},
                             {'params': [param for name, param in model.named_parameters() if 'pretrained' not in name], 'lr': start_lr_2}
                            ], 
                            lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    warm_up_iters = int(total_iters*0.05)
    warm_up_blend_1 = np.linspace(start_lr_1, goal_lr_1, warm_up_iters)
    warm_up_blend_2 = np.linspace(start_lr_2, goal_lr_2, warm_up_iters)

    # create schedular
    def lr_lambda(epoch):
        perc_goal_lr = 0.05  # x% of the original lr
        start_epoch = 5
        duration = 5.0
        if epoch < start_epoch:
            return 1.0
        elif start_epoch <= epoch < (start_epoch+duration):
            return 1.0 - (1.0-perc_goal_lr) * ((epoch - start_epoch) / duration)  # linear runter auf 0.1
        else:
            return 1.0 - (1.0-0.0001) * min(((epoch - (start_epoch+duration)) / epochs), 1.0)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    

    # Initialize Weights & Biases
    wandb.init(project="Master-PhysGen", name=model_name, config={
        "encoder": encoder,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "variation": variation
    })
    wandb.watch(model, log="all")

    last_model = None
    cur_iter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_img, target_depth, _ = batch
            target_depth = target_depth.squeeze(1)
            input_img, target_depth = input_img.to(device), target_depth.to(device)

            optimizer.zero_grad()
            pred_depth = model(input_img)
            # print(f"Prediction:\n    - Shape: {pred_depth.shape}\n    - Min: {pred_depth.min()}\n    - Max: {pred_depth.max()}")
            # print(f"Target:\n    - Shape: {target_depth.shape}\n    - Min: {target_depth.min()}\n    - Max: {target_depth.max()}")
            loss = criterion(pred_depth, target_depth) # criterion_1(pred_depth, target_depth)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if cur_iter < warm_up_iters:
                optimizer.param_groups[0]['lr'] = warm_up_blend_1[cur_iter]
                optimizer.param_groups[1]['lr'] = warm_up_blend_2[cur_iter]
            
            cur_iter += 1

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
                loss = criterion(pred_depth, target_depth) # criterion_1(pred_depth, target_depth) 
                val_loss += loss.item()

                if i == 0:
                    # Log first batch images
                    val_img_log = inference_forward(input_img, model, device, scale_to_256=True)


        avg_val_loss = val_loss / len(val_loader)
        loss_silog, loss_grad, loss_ssim, loss_l1 = criterion.get_avg_losses()
        # weight_silog, weight_grad, weight_ssim = criterion.last_weights
        wandb.log({
            "val_loss": avg_val_loss,
            "epoch": epoch + 1,
            "lr encoder": optimizer.param_groups[0]['lr'], # scheduler.get_last_lr()[0],
            "lr decoder": optimizer.param_groups[1]['lr'],
            "loss silog": loss_silog, 
            "loss grad": loss_grad, 
            "loss ssim": loss_ssim,
            "loss L1": loss_l1,
            "weight loss silog": criterion.weight_silog, 
            "weight loss grad": criterion.weight_grad,
            "weight loss ssim": criterion.weight_ssim,
            "weight loss L1": criterion.weight_l1,
            "sample_depth_map": wandb.Image(val_img_log) if val_img_log is not None else None
        })
        criterion.step(epoch)

        # Save model
        # if not best_model:
        #     best_model = f"./checkpoints/{args.model_name}_epoch{epoch+1}.pth"
        #     best_loss = avg_val_loss
        # elif best_loss > avg_val_loss:
        #     os.remove(best_model)

        #     best_model = f"./checkpoints/{args.model_name}_epoch{epoch+1}.pth"
        #     best_loss = avg_val_loss
        # else:
        #     continue

        if last_model:
            os.remove(last_model)

        last_model = f"./checkpoints/{args.model_name}_epoch{epoch+1}.pth"

        # Update Loss Weighting
        # if 0 <= epoch <= 10:    
        #     lambda_l1 = min(1.0, (epoch - 5) / 5)
        # elif epoch > 10:
        #     lambda_l1 = min(50.0, ((epoch - 10) / 150) * 50.0)
        # else:
        #     lambda_l1 = 0.0
        
        os.makedirs("./checkpoints", exist_ok=True)
        save_path = last_model
        torch.save(model.state_dict(), save_path)
        print(f"Saved model at {save_path}")

        # Update learn rate
        if cur_iter >= warm_up_iters:
            scheduler.step()

            # freeze encoder after warm up
            for name, param in model.named_parameters():
                if 'pretrained' in name:
                    param.requires_grad = False


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


