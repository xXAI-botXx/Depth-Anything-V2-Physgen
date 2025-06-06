import argparse
import os
import sys
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import transforms

from physgen_dataset import PhysGenDataset
from depth_anything_v2.dpt import DepthAnythingV2
from cfo_model import ComplexFocusOnly
from inference import inference_forward

import numpy as np
from tqdm import tqdm
import wandb
import torchvision.utils as vutils

import kornia  # for SSIM and gradients

class PerceptualLoss(nn.Module):
    def __init__(self, layers=('relu1_2', 'relu2_2', 'relu3_3'), device='cuda'):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features.to(device).eval()
        self.device = device

        self.layer_map = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 15,
            'relu4_3': 22,
            'relu5_3': 29,
        }
        self.selected_layers = layers
        self.vgg_slices = nn.Sequential(*list(vgg.children())[:self.layer_map[layers[-1]] + 1])
        for param in self.vgg_slices.parameters():
            param.requires_grad = False

        self.criterion = nn.MSELoss()

        self.normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])

    def _preprocess(self, img):
        # Expect img to be (B, 1, H, W), grayscale in [0,1]
        if len(img.shape) == 2:
            img = img.unsqueeze(0).unsqueeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(1)
        if img.shape[1] == 1:
            img = img.repeat(1, 3, 1, 1)  # Convert to 3 channels
        img = torch.stack([self.normalization(i) for i in img])
        return img

    def forward(self, pred, target):
        pred = self._preprocess(pred).to(self.device)
        target = self._preprocess(target).to(self.device)

        loss = 0.0
        x = pred
        y = target
        for i, layer in enumerate(self.vgg_slices):
            x = layer(x)
            y = layer(y)
            if i in [self.layer_map[l] for l in self.selected_layers]:
                loss += self.criterion(x, y)
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, 
                 silog_lambda=0.5, 
                 weight_silog=0.5, 
                 weight_grad=10.0, 
                 weight_ssim=5.0,
                 weight_edge_aware=10.0,
                 weight_l1=1.0,
                 weight_vgg=1.0):
        super().__init__()
        self.silog_lambda = silog_lambda
        self.weight_silog = weight_silog
        self.weight_grad = weight_grad
        self.weight_ssim = weight_ssim
        self.weight_edge_aware = weight_edge_aware
        self.weight_l1 = weight_l1
        self.weight_vgg = weight_vgg

        self.init_weight_silog = self.weight_silog
        self.init_weight_grad = self.weight_grad
        self.init_weight_ssim = self.weight_ssim
        self.init_weight_edge_aware = self.weight_edge_aware
        self.init_weight_l1 = self.weight_l1
        self.init_weight_vgg = self.weight_vgg

        self.avg_loss_silog = 0
        self.avg_loss_grad = 0
        self.avg_loss_ssim = 0
        self.avg_loss_l1 = 0
        self.avg_loss_vgg = 0
        self.avg_loss_edge_aware = 0
        self.steps = 0

        # Instantiate SSIMLoss module
        self.ssim_module = kornia.losses.SSIMLoss(window_size=11, reduction='mean')
        # self.ssim_module = kornia.losses.MS_SSIMLoss(reduction='mean')

        self.vgg_loss = PerceptualLoss(layers=('relu1_2', 'relu2_2', 'relu3_3'), device='cuda')


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

        # self.ssim_module = self.ssim_module.to(pred.device)
        return self.ssim_module(pred, target)

    def edge_aware_loss(self, pred, target):
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if target.ndim == 3:
            target = target.unsqueeze(1)

        pred_grad_x = pred[:, :, :, :-1] - pred[:, :, :, 1:]
        pred_grad_y = pred[:, :, :-1, :] - pred[:, :, 1:, :]

        target_grad_x = torch.mean(torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:]), 1, keepdim=True)
        target_grad_y = torch.mean(torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :]), 1, keepdim=True)

        pred_grad_x *= torch.exp(-target_grad_x)
        pred_grad_y *= torch.exp(-target_grad_y)

        # return (pred_grad_y.abs().mean() + target_grad_y.abs().mean())
        return (pred_grad_x.abs().mean() + pred_grad_y.abs().mean())

    def l1_loss(self, pred, target):
        loss = torch.abs(target - pred)
        return loss.mean()

    def forward(self, pred, target):
        loss_silog = self.silog_loss(pred, target)
        loss_grad = self.gradient_l1_loss(pred, target)
        loss_ssim = self.ssim_loss(pred, target)
        loss_l1 = self.l1_loss(pred, target)
        loss_edge_aware = self.edge_aware_loss(pred, target)
        loss_vgg = self.vgg_loss(pred, target)

        self.avg_loss_silog += loss_silog
        self.avg_loss_grad += loss_grad
        self.avg_loss_ssim += loss_ssim
        self.avg_loss_l1 += loss_l1
        self.avg_loss_edge_aware += loss_edge_aware
        self.avg_loss_vgg
        self.steps += 1

        total_loss = (
            self.weight_silog * loss_silog +
            self.weight_grad * loss_grad +
            self.weight_ssim * loss_ssim +
            self.weight_edge_aware * loss_edge_aware +
            self.weight_l1 * loss_l1 +
            self.weight_vgg * loss_vgg
        )
        return total_loss

    def step(self, epoch):
        self.avg_loss_silog = 0
        self.avg_loss_grad = 0
        self.avg_loss_ssim = 0
        self.avg_loss_l1 = 0
        self.avg_loss_edge_aware = 0
        self.avg_loss_vgg = 0
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
                self.avg_loss_l1/self.steps,
                self.avg_loss_edge_aware/self.steps,
                self.avg_loss_vgg/self.steps)

    def get_dict(self, data_idx):
        loss_silog, loss_grad, loss_ssim, loss_l1, loss_edge_aware, loss_vgg = self.get_avg_losses()
        return {
                f"{data_idx}_loss silog": loss_silog, 
                f"{data_idx}_loss grad": loss_grad, 
                f"{data_idx}_loss ssim": loss_ssim,
                f"{data_idx}_loss L1": loss_l1,
                f"{data_idx}_loss edge aware": loss_edge_aware,
                f"{data_idx}_loss vgg": loss_vgg,
                f"{data_idx}_weight loss silog": self.weight_silog, 
                f"{data_idx}_weight loss grad": self.weight_grad,
                f"{data_idx}_weight loss ssim": self.weight_ssim,
                f"{data_idx}_weight loss L1": self.weight_l1,
                f"{data_idx}_weight loss edge aware": self.weight_edge_aware,
                f"{data_idx}_weight vgg": self.weight_vgg
               }

# class LaplacianPyramidLoss(nn.Module):
#     def __init__(self, max_levels=3):
#         super().__init__()
#         self.max_levels = max_levels
#         self.gaussian_filter = self._build_gaussian_filter()

#     def _build_gaussian_filter(self):
#         # 5x5 Gaussian kernel
#         kernel = torch.tensor(
#             [[1., 4., 6., 4., 1.],
#              [4., 16., 24., 16., 4.],
#              [6., 24., 36., 24., 6.],
#              [4., 16., 24., 16., 4.],
#              [1., 4., 6., 4., 1.]]
#         )
#         kernel /= kernel.sum()
#         kernel = kernel.view(1, 1, 5, 5)
#         return kernel

#     def gaussian_blur(self, x):
#         C = x.shape[1]
#         kernel = self.gaussian_filter.to(x.device).repeat(C, 1, 1, 1)
#         x = F.pad(x, (2, 2, 2, 2), mode='reflect')
#         return F.conv2d(x, kernel, groups=C)

#     def laplacian_pyramid(self, x):
#         pyramid = []
#         current = x
#         for _ in range(self.max_levels):
#             blurred = self.gaussian_blur(current)
#             laplacian = current - blurred
#             pyramid.append(laplacian)
#             current = F.interpolate(blurred, scale_factor=0.5, mode='bilinear', align_corners=False)
#         pyramid.append(current)  # lowest-res residual
#         return pyramid

#     def forward(self, pred, target):
#         if pred.ndim == 3: pred = pred.unsqueeze(1)
#         if target.ndim == 3: target = target.unsqueeze(1)

#         pred_pyramid = self.laplacian_pyramid(pred)
#         target_pyramid = self.laplacian_pyramid(target)

#         loss = 0
#         for p, t in zip(pred_pyramid, target_pyramid):
#             loss += F.l1_loss(p, t)
#         return loss

class SmallDiffLoss(nn.Module):
    def __init__(self, 
                 exp_mse_loss_weight=1.0,
                 reciprocal_loss_weight=1.0,
                 ps_huber_loss_weight=1.0,
                 inv_w_mse_loss_weight=1.0,
                 log_cosh_loss_weight=1.0):
        super().__init__()
        self.exp_mse_loss_weight = exp_mse_loss_weight
        self.reciprocal_loss_weight = reciprocal_loss_weight
        self.ps_huber_loss_weight = ps_huber_loss_weight
        self.inv_w_mse_loss_weight = inv_w_mse_loss_weight
        self.log_cosh_loss_weight = log_cosh_loss_weight

        self.init_exp_mse_loss_weight = exp_mse_loss_weight
        self.init_reciprocal_loss_weight = reciprocal_loss_weight
        self.init_ps_huber_loss_weight = ps_huber_loss_weight
        self.init_inv_w_mse_loss_weight = inv_w_mse_loss_weight
        self.init_log_cosh_loss_weight = log_cosh_loss_weight

        self.avg_loss_exp_mse = 0
        self.avg_loss_reciprocal = 0
        self.avg_loss_ps_huber = 0
        self.avg_loss_inv_w_mse = 0
        self.avg_loss_log_cosh = 0
        # self.avg_loss_combined = 0
        self.steps = 0

        # self.combined_loss_model = CombinedLoss(silog_lambda=0.5, 
        #                                         weight_silog=1.0, 
        #                                         weight_grad=100.0, 
        #                                         weight_ssim=1.0,
        #                                         weight_edge_aware=100.0,
        #                                         weight_l1=10.0,
        #                                         weight_vgg=0.0)

    def soft_exp_mse_loss(self, pred, target):
        epsilon = 0.1
        error = (pred - target) ** 2
        return torch.mean(torch.exp(epsilon * error))

    def reciprocal_loss(self, pred, target):
        epsilon = 1e-6
        error = (pred - target) ** 2
        return torch.mean(1.0 / (error + epsilon))

    def ps_huber_loss(self, pred, target):
        delta = 0.1
        diff = pred - target
        return torch.mean(delta**2 * (torch.sqrt(1 + (diff / delta)**2) - 1))

    def inv_w_mse_loss(self, pred, target):
        epsilon = 1e-6
        error = (pred - target)
        weight = 1.0 / (torch.abs(error) + epsilon)
        return torch.mean(weight * (error ** 2))

    def log_cosh_loss(self, pred, target):
        diff = pred - target
        return torch.mean(torch.log(torch.cosh(diff + 1e-12)))

    def forward(self, pred, target):
        exp_mse_loss = self.soft_exp_mse_loss(pred, target)
        reciprocal_loss = self.reciprocal_loss(pred, target)
        ps_huber_loss = self.ps_huber_loss(pred, target)
        inv_w_mse_loss = self.inv_w_mse_loss(pred, target)
        log_cosh_loss = self.log_cosh_loss(pred, target)

        self.avg_loss_exp_mse += exp_mse_loss
        self.avg_loss_reciprocal += reciprocal_loss
        self.avg_loss_ps_huber += ps_huber_loss
        self.avg_loss_inv_w_mse += inv_w_mse_loss
        self.avg_loss_log_cosh += log_cosh_loss
        self.steps += 1

        total_loss = (
            self.exp_mse_loss_weight * exp_mse_loss +
            self.reciprocal_loss_weight * reciprocal_loss +
            self.ps_huber_loss_weight * ps_huber_loss +
            self.inv_w_mse_loss_weight * inv_w_mse_loss +
            self.log_cosh_loss_weight * log_cosh_loss
        )
        return total_loss

    def step(self, epoch):
        self.avg_loss_exp_mse = 0
        self.avg_loss_reciprocal = 0
        self.avg_loss_ps_huber = 0
        self.avg_loss_inv_w_mse = 0
        self.avg_loss_log_cosh = 0
        self.steps = 0

    def get_avg_losses(self):
        return (
                self.avg_loss_exp_mse/self.steps,
                self.avg_loss_reciprocal/self.steps,
                self.avg_loss_ps_huber/self.steps,
                self.avg_loss_inv_w_mse/self.steps,
                self.avg_loss_log_cosh/self.steps
               )
            
    def get_dict(self, data_idx):
        exp_mse_loss, reciprocal_loss, ps_huber_loss, inv_w_mse_loss, log_cosh_loss = self.get_avg_losses()
        return {
                f"{data_idx}_loss exp mse": exp_mse_loss, 
                f"{data_idx}_loss reciprocal": reciprocal_loss, 
                f"{data_idx}_loss ps huber": ps_huber_loss,
                f"{data_idx}_loss inv weight mse": inv_w_mse_loss,
                f"{data_idx}_loss log cosh loss": log_cosh_loss,
                f"{data_idx}_weight loss exp mse": self.exp_mse_loss_weight,
                f"{data_idx}_weight loss reciprocal": self.reciprocal_loss_weight,
                f"{data_idx}_weight loss ps huber": self.ps_huber_loss_weight,
                f"{data_idx}_weight loss inv weight mse": self.inv_w_mse_loss_weight,
                f"{data_idx}_weight loss log cosh": self.log_cosh_loss_weight
               }

class InbalancedHistLoss(nn.Module):
    def __init__(self, 
                 inverted_histogram_weighted_l1_loss_weight=1.0,
                 kl_divergence_histogram_loss_weight=0.1
                ):
        super().__init__()
        self.inverted_histogram_weighted_l1_loss_weight = inverted_histogram_weighted_l1_loss_weight
        self.kl_divergence_histogram_loss_weight = kl_divergence_histogram_loss_weight

        self.avg_loss_inverted_histogram_weighted_l1 = 0
        self.avg_loss_kl_divergence_histogram = 0
        self.steps = 0

    def inverted_histogram_weighted_l1_loss(self, pred, target):
        values, counts = torch.unique(target.flatten(), return_counts=True)
        all_counts = counts.sum().float()

        counts = torch.log(counts.float() + 1)
        all_counts = torch.log(all_counts + 1)

        weight_factor = 1.0
        weights = {values[idx].item(): torch.exp( ( (1-(counts[idx].item()/all_counts)) -0.2) *weight_factor) for idx in range(len(values))}
        # weights = {values[idx].item(): max(counts[idx].item()/all_counts, min_weight) for idx in range(len(values))}
        
        weights_map = torch.zeros_like(target, dtype=torch.float)
        for cur_value in values:
            cur_value = cur_value.item()
            weights_map[target == cur_value] = weights[cur_value]
        loss = weights_map * torch.abs(pred - target)
        return loss.mean()

    # def inverted_histogram_weighted_l1_loss(self, pred, target):
    #     values, counts = torch.unique(target, return_counts=True)
    #     all_counts = counts.sum().float()

    #     weights = 1.0 - (counts.float() / all_counts)

    #     # Build weights_map with broadcasting (vectorized mask)
    #     weights_map = torch.zeros_like(target)
    #     for i, val in enumerate(values):
    #         mask = target == val
    #         weights_map[mask] = weights[i]

    #     loss = weights_map * torch.abs(pred - target)
    #     return loss.mean()

    def kl_divergence_histogram_loss(self, pred, target, bins=255):
        pred_hist = torch.histc(pred, bins=bins, min=0, max=1)
        target_hist = torch.histc(target, bins=bins, min=0, max=1)

        # pred_hist = torch.log(pred_hist + 1)
        # target_hist = torch.log(target_hist + 1)

        pred_p = pred_hist / (pred_hist.sum() + 1e-8)
        target_p = target_hist / (target_hist.sum() + 1e-8)

        kl = (target_p * (torch.log(target_p + 1e-8) - torch.log(pred_p + 1e-8))).sum()
        return kl

    def forward(self, pred, target):
        inverted_histogram_weighted_l1_loss = self.inverted_histogram_weighted_l1_loss(pred, target)
        kl_divergence_histogram_loss = self.kl_divergence_histogram_loss(pred, target)

        self.avg_loss_inverted_histogram_weighted_l1 += inverted_histogram_weighted_l1_loss
        self.avg_loss_kl_divergence_histogram += kl_divergence_histogram_loss
        self.steps += 1

        total_loss = (
            self.inverted_histogram_weighted_l1_loss_weight * inverted_histogram_weighted_l1_loss +
            self.kl_divergence_histogram_loss_weight * kl_divergence_histogram_loss
        )
        return total_loss

    def step(self, epoch):
        self.avg_loss_inverted_histogram_weighted_l1 = 0
        self.avg_loss_kl_divergence_histogram = 0
        self.steps = 0

    def get_avg_losses(self):
        return (
                self.avg_loss_inverted_histogram_weighted_l1/self.steps,
                self.avg_loss_kl_divergence_histogram/self.steps
               )
            
    def get_dict(self, data_idx):
        inverted_histogram_weighted_l1_loss, kl_divergence_histogram_loss = self.get_avg_losses()
        return {
                f"{data_idx}_loss histogram weighted l1": inverted_histogram_weighted_l1_loss, 
                f"{data_idx}_loss kl divergence histogram": kl_divergence_histogram_loss, 
                f"{data_idx}_weight loss histogram weighted l1": self.inverted_histogram_weighted_l1_loss_weight,
                f"{data_idx}_weight loss kl divergence histogram": self.kl_divergence_histogram_loss_weight
               }

class FocalLoss(nn.Module):
    def __init__(self, 
                 focal_loss_weight=1.0
                ):
        super().__init__()
        self.focal_loss_weight = focal_loss_weight

        self.avg_loss_focal = 0
        self.steps = 0

    def focal_loss(self, pred, target):
        alpha = 1
        gamma = 2
        mse = F.mse_loss(pred, target, reduction='none')
        pt = torch.exp(-mse)
        focal_loss = alpha * (1-pt)**gamma * mse
        return focal_loss.mean()

    def forward(self, pred, target):
        focal_loss = self.focal_loss(pred, target)

        self.avg_loss_focal += focal_loss
        self.steps += 1

        total_loss = (
            self.focal_loss_weight * focal_loss
        )
        return total_loss

    def step(self, epoch):
        self.avg_loss_focal = 0
        self.steps = 0

    def get_avg_losses(self):
        return (
                self.avg_loss_focal/self.steps
               )
            
    def get_dict(self, data_idx):
        avg_loss_focal = self.get_avg_losses()
        return {
                f"{data_idx}_loss focal": avg_loss_focal, 
                f"{data_idx}_weight loss focal": self.focal_loss_weight
               }

class WeightedInbalancedDiffLoss(nn.Module):
    def __init__(self, 
                 l1_loss_weight=1.0
                ):
        super().__init__()
        self.l1_loss_weight = l1_loss_weight

        self.avg_loss_l1 = 0
        self.steps = 0

    def anti_collapse_loss(self, pred, target):
        diff = torch.abs(target - pred)
        
        # Basis-Gewichtung
        weights = torch.ones_like(target, dtype=torch.float)
        weights[target <= 0.1] = 0.01  # Minimale Gewichtung für Nullwerte
        weights[(target > 0.4) & (target < 0.65)] = 1.0
        
        # KRITISCH: Bestrafung wenn Vorhersage bei 0.5 liegt, aber Target nicht
        pred_near_half = (pred > 0.45) & (pred < 0.55)
        target_not_half = (target < 0.4) | (target > 0.65)
        collapse_penalty = pred_near_half & target_not_half
        
        # Massive Bestrafung für Kollaps
        weights[collapse_penalty] = 1000.0
        
        # Extra Belohnung für korrekte Nicht-0.5-Vorhersagen
        correct_rare = (target > 0.65) | (target < 0.4)
        weights[correct_rare] = 500.0
        
        return torch.mean(diff * weights)

    def forward(self, pred, target):
        l1_loss = self.anti_collapse_loss(pred, target)

        self.avg_loss_l1 += l1_loss
        self.steps += 1

        total_loss = (
            self.l1_loss_weight * l1_loss
        )
        return total_loss

    def step(self, epoch):
        self.avg_loss_l1 = 0
        self.steps = 0

    def get_avg_losses(self):
        return (
                self.avg_loss_l1/self.steps
               )
            
    def get_dict(self, data_idx):
        avg_loss_l1 = self.get_avg_losses()
        return {
                f"{data_idx}_loss weighted l1": avg_loss_l1, 
                f"{data_idx}_weight loss weighted l1": self.l1_loss_weight
               }

class InbalancedSmallDiffLoss(nn.Module):
    def __init__(self, 
                 inverted_histogram_weighted_l1_loss_weight=1.0,
                 kl_divergence_histogram_loss_weight=0.1
                ):
        super().__init__()
        self.inverted_histogram_weighted_l1_loss_weight = inverted_histogram_weighted_l1_loss_weight
        self.kl_divergence_histogram_loss_weight = kl_divergence_histogram_loss_weight

        self.avg_loss_inverted_histogram_weighted_l1 = 0
        self.avg_loss_kl_divergence_histogram = 0
        self.steps = 0

    def inverted_histogram_weighted_l1_loss(self, pred, target):
        values, counts = torch.unique(target.flatten(), return_counts=True)
        all_counts = counts.sum().float()

        counts = torch.log(counts.float() + 1)
        all_counts = torch.log(all_counts + 1)

        weight_factor = 100.0
        weights = {values[idx].item(): torch.exp( (1-(counts[idx].item()/all_counts)) *weight_factor) for idx in range(len(values))}
        # weights = {values[idx].item(): max(counts[idx].item()/all_counts, min_weight) for idx in range(len(values))}
        
        weights_map = torch.zeros_like(target, dtype=torch.float)
        for cur_value in values:
            cur_value = cur_value.item()
            weights_map[target == cur_value] = weights[cur_value]
        loss = weights_map * torch.abs(pred - target)
        return loss.mean()

    def kl_divergence_histogram_loss(self, pred, target, bins=255):
        pred_hist = torch.histc(pred, bins=bins, min=0, max=1)
        target_hist = torch.histc(target, bins=bins, min=0, max=1)

        # pred_hist = torch.log(pred_hist + 1)
        # target_hist = torch.log(target_hist + 1)

        pred_p = pred_hist / (pred_hist.sum() + 1e-8)
        target_p = target_hist / (target_hist.sum() + 1e-8)

        kl = (target_p * (torch.log(target_p + 1e-8) - torch.log(pred_p + 1e-8))).sum()
        return kl

    def forward(self, pred, target):
        inverted_histogram_weighted_l1_loss = self.inverted_histogram_weighted_l1_loss(pred, target)
        kl_divergence_histogram_loss = self.kl_divergence_histogram_loss(pred, target)

        self.avg_loss_inverted_histogram_weighted_l1 += inverted_histogram_weighted_l1_loss
        self.avg_loss_kl_divergence_histogram += kl_divergence_histogram_loss
        self.steps += 1

        total_loss = (
            self.inverted_histogram_weighted_l1_loss_weight * inverted_histogram_weighted_l1_loss +
            self.kl_divergence_histogram_loss_weight * kl_divergence_histogram_loss
        )
        return total_loss

    def step(self, epoch):
        self.avg_loss_inverted_histogram_weighted_l1 = 0
        self.avg_loss_kl_divergence_histogram = 0
        self.steps = 0

    def get_avg_losses(self):
        return (
                self.avg_loss_inverted_histogram_weighted_l1/self.steps,
                self.avg_loss_kl_divergence_histogram/self.steps
               )
            
    def get_dict(self, data_idx):
        inverted_histogram_weighted_l1_loss, kl_divergence_histogram_loss = self.get_avg_losses()
        return {
                f"{data_idx}_loss histogram weighted l1": inverted_histogram_weighted_l1_loss, 
                f"{data_idx}_loss kl divergence histogram": kl_divergence_histogram_loss, 
                f"{data_idx}_weight loss histogram weighted l1": self.inverted_histogram_weighted_l1_loss_weight,
                f"{data_idx}_weight loss kl divergence histogram": self.kl_divergence_histogram_loss_weight
               }

class ExtendedInbalancedSmallDiffLoss(nn.Module):
    def __init__(self,
                 inverted_histogram_weighted_l1_loss_weight=1.0,
                 kl_divergence_histogram_loss_weight=0.1,
                 variance_loss_weight=1.0,
                 range_loss_weight=1.0,
                 percentile_loss_weight=1.0,
                 focal_high_value_loss_weight=10.0,
                 gradient_loss_weight=0.5,
                 adaptive_threshold_loss_weight=2.0
                ):
        super().__init__()
        self.inverted_histogram_weighted_l1_loss_weight = inverted_histogram_weighted_l1_loss_weight
        self.kl_divergence_histogram_loss_weight = kl_divergence_histogram_loss_weight
        self.variance_loss_weight = variance_loss_weight
        self.range_loss_weight = range_loss_weight
        self.percentile_loss_weight = percentile_loss_weight
        self.focal_high_value_loss_weight = focal_high_value_loss_weight
        self.gradient_loss_weight = gradient_loss_weight
        self.adaptive_threshold_loss_weight = adaptive_threshold_loss_weight
        
        # Tracking variables
        self.avg_losses = {}
        self.steps = 0
    
    def inverted_histogram_weighted_l1_loss(self, pred, target):
        values, counts = torch.unique(target.flatten(), return_counts=True)
        all_counts = counts.sum().float()
        counts = torch.log(counts.float() + 1)
        all_counts = torch.log(all_counts + 1)
        weight_factor = 2.0
        weights = {values[idx].item(): torch.exp( ( (1-(counts[idx].item()/all_counts))) *weight_factor) for idx in range(len(values))}
       
        weights_map = torch.zeros_like(target, dtype=torch.float)
        for cur_value in values:
            cur_value = cur_value.item()
            weights_map[target == cur_value] = weights[cur_value]
        loss = weights_map * torch.abs(pred - target)
        return loss.mean()
    
    def kl_divergence_histogram_loss(self, pred, target, bins=255):
        pred_hist = torch.histc(pred, bins=bins, min=0, max=1)
        target_hist = torch.histc(target, bins=bins, min=0, max=1)
        pred_p = pred_hist / (pred_hist.sum() + 1e-8)
        target_p = target_hist / (target_hist.sum() + 1e-8)
        kl = (target_p * (torch.log(target_p + 1e-8) - torch.log(pred_p + 1e-8))).sum()
        return kl
    
    def variance_loss(self, pred, target):
        """Penalize difference in variance to encourage proper spread"""
        pred_var = torch.var(pred)
        target_var = torch.var(target)
        return F.mse_loss(pred_var, target_var)
    
    def range_loss(self, pred, target):
        """Penalize difference in min/max values to encourage full range"""
        pred_min, pred_max = torch.min(pred), torch.max(pred)
        target_min, target_max = torch.min(target), torch.max(target)
        
        min_loss = F.mse_loss(pred_min, target_min)
        max_loss = F.mse_loss(pred_max, target_max)
        
        return min_loss + max_loss
    
    def percentile_loss(self, pred, target, percentiles=[90, 95, 99]):
        """Focus on matching high percentiles (rare high values)"""
        loss = 0
        for p in percentiles:
            pred_p = torch.quantile(pred.flatten(), p/100.0)
            target_p = torch.quantile(target.flatten(), p/100.0)
            loss += F.mse_loss(pred_p, target_p) * (p/100.0)  # Higher weight for higher percentiles
        return loss / len(percentiles)
    
    def focal_high_value_loss(self, pred, target, threshold=0.1):
        """Focal loss specifically for high values (>threshold)"""
        high_mask = target > threshold
        if high_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        high_pred = pred[high_mask]
        high_target = target[high_mask]
        
        # Focal loss with gamma=2 for hard examples
        diff = torch.abs(high_pred - high_target)
        focal_weight = (1 - torch.exp(-diff)) ** 2  # Higher weight for larger errors
        
        return (focal_weight * diff).mean()
    
    def gradient_loss(self, pred, target):
        """Preserve spatial gradients for better detail preservation"""
        def compute_gradient(x):
            grad_x = x[:, :, 1:] - x[:, :, :-1]  # Horizontal gradient
            grad_y = x[:, 1:, :] - x[:, :-1, :]  # Vertical gradient
            return grad_x, grad_y
        
        pred_grad_x, pred_grad_y = compute_gradient(pred.unsqueeze(1) if pred.dim() == 2 else pred)
        target_grad_x, target_grad_y = compute_gradient(target.unsqueeze(1) if target.dim() == 2 else target)
        
        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        
        return loss_x + loss_y
    
    def adaptive_threshold_loss(self, pred, target):
        """Adaptive loss that changes weight based on target value magnitude"""
        # Create adaptive weights: higher weight for higher target values
        adaptive_weights = 1.0 + (target / (torch.max(target) + 1e-8)) * 10.0
        
        diff = torch.abs(pred - target)
        weighted_diff = adaptive_weights * diff
        
        return weighted_diff.mean()
    
    def forward(self, pred, target):
        # Compute all losses
        losses = {}
        
        losses['inverted_histogram_weighted_l1'] = self.inverted_histogram_weighted_l1_loss(pred, target)
        losses['kl_divergence_histogram'] = self.kl_divergence_histogram_loss(pred, target)
        losses['variance'] = self.variance_loss(pred, target)
        losses['range'] = self.range_loss(pred, target)
        losses['percentile'] = self.percentile_loss(pred, target)
        losses['focal_high_value'] = self.focal_high_value_loss(pred, target)
        losses['gradient'] = self.gradient_loss(pred, target)
        losses['adaptive_threshold'] = self.adaptive_threshold_loss(pred, target)
        
        # Update running averages
        for key, value in losses.items():
            if key not in self.avg_losses:
                self.avg_losses[key] = 0
            self.avg_losses[key] += value.item()
        
        self.steps += 1
        
        # Combine losses
        total_loss = (
            self.inverted_histogram_weighted_l1_loss_weight * losses['inverted_histogram_weighted_l1'] +
            self.kl_divergence_histogram_loss_weight * losses['kl_divergence_histogram'] +
            self.variance_loss_weight * losses['variance'] +
            self.range_loss_weight * losses['range'] +
            self.percentile_loss_weight * losses['percentile'] +
            self.focal_high_value_loss_weight * losses['focal_high_value'] +
            self.gradient_loss_weight * losses['gradient'] +
            self.adaptive_threshold_loss_weight * losses['adaptive_threshold']
        )
        
        return total_loss
    
    def step(self, epoch):
        self.avg_losses = {}
        self.steps = 0
    
    def get_avg_losses(self):
        if self.steps == 0:
            return {key: 0 for key in self.avg_losses.keys()}
        return {key: value / self.steps for key, value in self.avg_losses.items()}
    
    def get_dict(self, data_idx):
        avg_losses = self.get_avg_losses()
        result = {}
        
        # Add losses
        for key, value in avg_losses.items():
            result[f"{data_idx}_loss {key}"] = value
        
        # Add weights
        result.update({
            f"{data_idx}_weight loss histogram weighted l1": self.inverted_histogram_weighted_l1_loss_weight,
            f"{data_idx}_weight loss kl divergence histogram": self.kl_divergence_histogram_loss_weight,
            f"{data_idx}_weight loss variance": self.variance_loss_weight,
            f"{data_idx}_weight loss range": self.range_loss_weight,
            f"{data_idx}_weight loss percentile": self.percentile_loss_weight,
            f"{data_idx}_weight loss focal high value": self.focal_high_value_loss_weight,
            f"{data_idx}_weight loss gradient": self.gradient_loss_weight,
            f"{data_idx}_weight loss adaptive threshold": self.adaptive_threshold_loss_weight,
        })
        
        return result

class WeightedCombinedLoss(nn.Module):
    def __init__(self, 
                 silog_lambda=0.5, 
                 weight_silog=0.5, 
                 weight_grad=10.0, 
                 weight_ssim=5.0,
                 weight_edge_aware=10.0,
                 weight_l1=1.0,
                 weight_var=1.0,
                 weight_range=1.0):
        super().__init__()
        self.silog_lambda = silog_lambda
        self.weight_silog = weight_silog
        self.weight_grad = weight_grad
        self.weight_ssim = weight_ssim
        self.weight_edge_aware = weight_edge_aware
        self.weight_l1 = weight_l1
        self.weight_var = weight_var
        self.weight_range = weight_range

        self.avg_loss_silog = 0
        self.avg_loss_grad = 0
        self.avg_loss_ssim = 0
        self.avg_loss_l1 = 0
        self.avg_loss_edge_aware = 0
        self.avg_loss_var = 0
        self.avg_loss_range = 0
        self.steps = 0

        # Instantiate SSIMLoss module
        self.ssim_module = kornia.losses.SSIMLoss(window_size=11, reduction='mean')
        # self.ssim_module = kornia.losses.MS_SSIMLoss(reduction='mean')


    def silog_loss(self, pred, target, weight_map):
        eps = 1e-6
        pred = torch.clamp(pred, min=eps)
        target = torch.clamp(target, min=eps)
        
        diff_log = torch.log(target) - torch.log(pred)
        diff_log = diff_log * weight_map

        loss = torch.sqrt(torch.mean(diff_log ** 2) -
                          self.silog_lambda * torch.mean(diff_log) ** 2)
        return loss

    def gradient_l1_loss(self, pred, target, weight_map):
        # Create Channel Dimension
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if target.ndim == 3:
            target = target.unsqueeze(1)
        if weight_map.ndim == 3:
            weight_map = weight_map.unsqueeze(1)

        # Gradient in x-direction (horizontal -> dim=3)
        pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]

        # Gradient in y-direction (vertical -> dim=2)
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]

        weight_x = weight_map[:, :, :, 1:] * weight_map[:, :, :, :-1]
        weight_y = weight_map[:, :, 1:, :] * weight_map[:, :, :-1, :]

        loss_x = torch.mean(torch.abs(pred_grad_x - target_grad_x) * weight_x)
        loss_y = torch.mean(torch.abs(pred_grad_y - target_grad_y) * weight_y)
        
        # loss_x = F.l1_loss(pred_grad_x, target_grad_x) 
        # loss_y = F.l1_loss(pred_grad_y, target_grad_y)

        return loss_x + loss_y

    def ssim_loss(self, pred, target, weight_map):
        # SSIM returns similarity, so we subtract from 1
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if target.ndim == 3:
            target = target.unsqueeze(1)

        # self.ssim_module = self.ssim_module.to(pred.device)
        return self.ssim_module(pred, target)

    def edge_aware_loss(self, pred, target, weight_map):
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if target.ndim == 3:
            target = target.unsqueeze(1)
        if weight_map.ndim == 3:
            weight_map = weight_map.unsqueeze(1)

        pred_grad_x = pred[:, :, :, :-1] - pred[:, :, :, 1:]
        pred_grad_y = pred[:, :, :-1, :] - pred[:, :, 1:, :]

        target_grad_x = torch.mean(torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:]), 1, keepdim=True)
        target_grad_y = torch.mean(torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :]), 1, keepdim=True)

        weight_x = weight_map[:, :, :, 1:] * weight_map[:, :, :, :-1]
        weight_y = weight_map[:, :, 1:, :] * weight_map[:, :, :-1, :]

        pred_grad_x *= torch.exp(-target_grad_x* weight_x) 
        pred_grad_y *= torch.exp(-target_grad_y* weight_y)

        # return (pred_grad_y.abs().mean() + target_grad_y.abs().mean())
        return (pred_grad_x.abs().mean() + pred_grad_y.abs().mean())

    def l1_loss(self, pred, target, weight_map):
        loss = torch.abs(target - pred) * weight_map
        return loss.mean()

    def variance_loss(self, pred, target):
        pred_var = torch.var(pred)
        target_var = torch.var(target)
        return F.mse_loss(pred_var, target_var)
    
    def range_loss(self, pred, target):
        pred_min, pred_max = torch.min(pred), torch.max(pred)
        target_min, target_max = torch.min(target), torch.max(target)
        
        min_loss = F.mse_loss(pred_min, target_min)
        max_loss = F.mse_loss(pred_max, target_max)
        
        return min_loss + max_loss

    def forward(self, pred, target):
        weight_map = calc_weight_map(target)
        loss_silog = self.silog_loss(pred, target, weight_map)
        loss_grad = self.gradient_l1_loss(pred, target, weight_map)
        loss_ssim = self.ssim_loss(pred, target, weight_map)
        loss_l1 = self.l1_loss(pred, target, weight_map)
        loss_edge_aware = self.edge_aware_loss(pred, target, weight_map)
        loss_var = self.variance_loss(pred, target)
        loss_range = self.range_loss(pred, target)

        self.avg_loss_silog += loss_silog
        self.avg_loss_grad += loss_grad
        self.avg_loss_ssim += loss_ssim
        self.avg_loss_l1 += loss_l1
        self.avg_loss_edge_aware += loss_edge_aware
        self.avg_loss_var += loss_var
        self.avg_loss_range += loss_range
        self.steps += 1

        total_loss = (
            self.weight_silog * loss_silog +
            self.weight_grad * loss_grad +
            self.weight_ssim * loss_ssim +
            self.weight_edge_aware * loss_edge_aware +
            self.weight_l1 * loss_l1 +
            self.weight_var * loss_var +
            self.weight_range * loss_range
        )
        return total_loss

    def step(self, epoch):
        self.avg_loss_silog = 0
        self.avg_loss_grad = 0
        self.avg_loss_ssim = 0
        self.avg_loss_l1 = 0
        self.avg_loss_edge_aware = 0
        self.avg_loss_var = 0
        self.avg_loss_range = 0
        self.steps = 0

    def get_avg_losses(self):
        return (self.avg_loss_silog/self.steps,
                self.avg_loss_grad/self.steps,
                self.avg_loss_ssim/self.steps,
                self.avg_loss_l1/self.steps,
                self.avg_loss_edge_aware/self.steps,
                self.avg_loss_var/self.steps,
                self.avg_loss_range/self.steps
               )

    def get_dict(self, data_idx):
        loss_silog, loss_grad, loss_ssim, loss_l1, loss_edge_aware, loss_var, loss_range = self.get_avg_losses()
        return {
                f"{data_idx}_loss silog": loss_silog, 
                f"{data_idx}_loss grad": loss_grad, 
                f"{data_idx}_loss ssim": loss_ssim,
                f"{data_idx}_loss L1": loss_l1,
                f"{data_idx}_loss edge aware": loss_edge_aware,
                f"{data_idx}_loss var": loss_var,
                f"{data_idx}_loss range": loss_range,
                f"{data_idx}_weight loss silog": self.weight_silog, 
                f"{data_idx}_weight loss grad": self.weight_grad,
                f"{data_idx}_weight loss ssim": self.weight_ssim,
                f"{data_idx}_weight loss L1": self.weight_l1,
                f"{data_idx}_weight loss edge aware": self.weight_edge_aware,
                f"{data_idx}_weight loss var": self.weight_var,
                f"{data_idx}_weight loss range": self.weight_range
               }

def calc_weight_map(target):
    values, counts = torch.unique(target.flatten(), return_counts=True)
    all_counts = counts.sum().float()
    
    # weight_factor = 2.0
    # weights = {values[idx].item(): max(torch.exp( ( (1-(counts[idx].item()/all_counts))) *weight_factor), 0.0001) for idx in range(len(values))}
    
    weights = {values[idx].item(): 255.0/counts[idx].item() for idx in range(len(values))}

    # print(f"Weights:")
    # for cur_value, cur_counts in list(sorted(weights.items(), key=lambda x:x[0])):
    #     print('    - '+str(round(cur_value, 4))+': '+str(cur_counts.item()))

    weights_map = torch.zeros_like(target, dtype=torch.float)
    for cur_value in values:
        cur_value = cur_value.item()
        weights_map[target == cur_value] = weights[cur_value]

    return weights_map

def analyze_target_distribution(target):
    print("=== TARGET ANALYSIS ===")
    print(f"Min: {target.min().item()}")
    print(f"Max: {target.max().item()}")
    print(f"Mean: {target.mean().item()}")
    print(f"Median: {target.median().item()}")
    
    # Histogramm der Werte
    unique_vals, counts = torch.unique(target, return_counts=True)
    for val, count in zip(unique_vals[:10], counts[:10]):  # Top 10
        print(f"Value {val.item():.3f}: {count.item()} times")
    
    # Bereiche analysieren
    ranges = [
        (0.0, 0.1, "0.0-0.1"),
        (0.1, 0.4, "0.1-0.4"), 
        (0.4, 0.7, "0.4-0.7"),
        (0.7, 1.0, "0.7-1.0")
    ]
    
    for min_val, max_val, name in ranges:
        mask = (target >= min_val) & (target < max_val)
        count = mask.sum().item()
        print(f"Range {name}: {count} values")



def train(variation, input_type, output_type, model_name, model_type, encoder, batch_size, epochs, lr):
    settings = variation, input_type, output_type, model_name, model_type, encoder, batch_size, epochs, lr
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Prepare dataset
    if model_type in ["complex_focus_only", "complex_focus_only_pix2pix"]:
        train_dataset_base = PhysGenDataset(mode='train', variation="sound_baseline", input_type="osm", output_type="standard")
        val_dataset_base = PhysGenDataset(mode='validation', variation="sound_baseline", input_type="osm", output_type="standard")
        train_loader_base = DataLoader(train_dataset_base, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader_base = DataLoader(val_dataset_base, batch_size=batch_size, shuffle=False, num_workers=2)

        train_dataset_complex = PhysGenDataset(mode='train', variation=variation, input_type="osm", output_type="complex_only")
        val_dataset_complex = PhysGenDataset(mode='validation', variation=variation, input_type="osm", output_type="complex_only")
        train_loader_complex = DataLoader(train_dataset_complex, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader_complex = DataLoader(val_dataset_complex, batch_size=batch_size, shuffle=False, num_workers=2)

        train_dataset_fusion = PhysGenDataset(mode='train', variation=variation, input_type="osm", output_type="standard")
        val_dataset_fusion = PhysGenDataset(mode='validation', variation=variation, input_type="osm", output_type="standard")
        train_loader_fusion = DataLoader(train_dataset_fusion, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader_fusion = DataLoader(val_dataset_fusion, batch_size=batch_size, shuffle=False, num_workers=2)
        datasets = [(train_loader_base, val_loader_base), (train_loader_complex, val_loader_complex), (train_loader_fusion, val_loader_fusion)]
    else:
        train_dataset = PhysGenDataset(mode='train', variation=variation, input_type=input_type, output_type=output_type)
        val_dataset = PhysGenDataset(mode='validation', variation=variation, input_type=input_type, output_type=output_type)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        datasets = [(train_loader, val_loader)]

    if model_type in ["complex_focus_only", "complex_focus_only_pix2pix"]:
        total_iters = epochs * sum(len(loader) for loader, _ in datasets)
    else:
        total_iters = epochs * len(datasets[0][0])    # loop steps / optimizer steps, not every single image steps

    # Model configuration
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    if model_type == "complex_focus_only":
        model = ComplexFocusOnly(encoder).to(device)
    else:
        model = DepthAnythingV2(**model_configs[encoder]).to(device)
    for param in model.parameters():
        param.requires_grad = True

    # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    if model_type == "complex_focus_only":
        # Base
        base_start_lr_1 = 1e-8
        base_goal_lr_1 = lr*0.01
        base_start_lr_2 = lr*1.0
        base_goal_lr_2 = lr # * 10.0
        base_optimizer = optim.AdamW([
                                {'params': [param for name, param in model.phys_anything_baseline.named_parameters() if 'pretrained' in name], 'lr': base_start_lr_1},
                                {'params': [param for name, param in model.phys_anything_baseline.named_parameters() if 'pretrained' not in name], 'lr': base_start_lr_2}
                                ], 
                                lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
        base_warm_up_iters = int(total_iters*0.05)
        base_warm_up_blend_1 = np.linspace(base_start_lr_1, base_goal_lr_1, base_warm_up_iters)
        base_warm_up_blend_2 = np.linspace(base_start_lr_2, base_goal_lr_2, base_warm_up_iters)
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(base_optimizer, T_max=epochs, eta_min=1e-6)

        # Complex
        complex_start_lr_1 = 1e-8
        complex_goal_lr_1 = lr*0.001
        complex_start_lr_2 = lr*0.01 # lr*1000.0
        complex_goal_lr_2 = lr*10.0 # lr*1000.0
        complex_optimizer = optim.AdamW([
                                {'params': [param for name, param in model.phys_anything_complex.named_parameters() if 'pretrained' in name], 'lr': complex_start_lr_1},
                                {'params': [param for name, param in model.phys_anything_complex.named_parameters() if 'pretrained' not in name], 'lr':complex_start_lr_2}
                                ], 
                                lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
        complex_warm_up_iters = int(total_iters*0.01)
        complex_warm_up_blend_1 = np.linspace(complex_start_lr_1, complex_goal_lr_1, complex_warm_up_iters)
        complex_warm_up_blend_2 = np.linspace(complex_start_lr_2, complex_goal_lr_2, complex_warm_up_iters)
        complex_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(complex_optimizer, T_max=epochs, eta_min=1e-6)

        # Fusion
        fusion_start_lr = 1e-8
        fusion_goal_lr = lr*0.001
        fusion_optimizer = optim.AdamW(model.fusion_head.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
        fusion_warm_up_iters = int(total_iters*0.05)
        fusion_warm_up_blend = np.linspace(fusion_start_lr, fusion_goal_lr, fusion_warm_up_iters)
        fusion_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fusion_optimizer, T_max=epochs, eta_min=1e-6)

        optimizer = [base_optimizer, complex_optimizer, fusion_optimizer]
        warm_up_blend = [(base_warm_up_blend_1, base_warm_up_blend_2), (complex_warm_up_blend_1, complex_warm_up_blend_2), fusion_warm_up_blend]
        all_warm_up_iters = [base_warm_up_iters, complex_warm_up_iters, fusion_warm_up_iters]

        criterion = [CombinedLoss(silog_lambda=0.5, 
                                  weight_silog=10.0, 
                                  weight_grad=1000.0, 
                                  weight_ssim=10.0,
                                  weight_edge_aware=1000.0,
                                  weight_l1=100.0,
                                  weight_vgg=100.0),
                     WeightedCombinedLoss(silog_lambda=0.5, 
                                            weight_silog=0.5, 
                                            weight_grad=10.0, 
                                            weight_ssim=5.0,
                                            weight_edge_aware=10.0,
                                            weight_l1=100.0,
                                            weight_var=100.0,
                                            weight_range=1000.0),
                     CombinedLoss(silog_lambda=0.5, 
                                  weight_silog=10.0, 
                                  weight_grad=1000.0, 
                                  weight_ssim=10.0,
                                  weight_edge_aware=1000.0,
                                  weight_l1=100.0,
                                  weight_vgg=100.0)]
    elif model_type == "complex_focus_only_pix2pix":
        pass
    else:
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

        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        # criterion = [CombinedLoss(silog_lambda=0.5, 
        #                           weight_silog=0.5, 
        #                           weight_grad=10.0, 
        #                           weight_ssim=5.0,
        #                           weight_edge_aware=10.0,
        #                           weight_l1=1.0,
        #                           weight_vgg=1.0)]
        criterion = [WeightedCombinedLoss(silog_lambda=0.5, 
                                            weight_silog=0.5, 
                                            weight_grad=10.0, 
                                            weight_ssim=5.0,
                                            weight_edge_aware=10.0,
                                            weight_l1=100.0,
                                            weight_var=100.0,
                                            weight_range=1000.0)]
    

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
    global_cur_iter = [0]*len(criterion)

    # Analyze errors
    # torch.autograd.set_detect_anomaly(True)

    for epoch in range(epochs):
        # # FIXME
        # if epoch == 3:
        #     break
        for data_idx, (train_loader, val_loader) in enumerate(datasets):
            # if data_idx != 1:
            #     continue

            # Start Learning Fusion head after 6 epochs
            if data_idx == 2 and epoch <= 5:# epochs*0.8:
                continue
            # elif data_idx in [0, 1] and epoch > epochs*0.8:
            #     continue

            if model_type in ["complex_focus_only", "complex_focus_only_pix2pix"]:
                warm_up_iters = all_warm_up_iters[data_idx]

            model.train()

            if model_type in ["complex_focus_only", "complex_focus_only_pix2pix"]:
                model.switch_train(data_idx)

            running_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                input_img, target_depth, _ = batch
                target_depth = target_depth.squeeze(1)
                input_img, target_depth = input_img.to(device), target_depth.to(device)

                # analyze_target_distribution(target_depth)

                if model_type in ["complex_focus_only", "complex_focus_only_pix2pix"]:
                    optimizer[data_idx].zero_grad()  # set gradients to 0
                else:
                    optimizer.zero_grad()  # set gradients to 0
                if model_type in ["complex_focus_only", "complex_focus_only_pix2pix"]:
                    # print("Input min/max/nan:", input_img.min().item(), input_img.max().item(), torch.isnan(input_img).any())
                    pred_depth = model.forward_part(input_img, data_idx)
                    # print("Output min/max/nan:", pred_depth.min().item(), pred_depth.max().item(), torch.isnan(pred_depth).any())
                else:
                    pred_depth = model(input_img)
                loss = criterion[data_idx](pred_depth, target_depth) # criterion_1(pred_depth, target_depth)
                # print(f"Loss ({data_idx}):", loss.item())
                # print("Loss value:", loss.item(), "Is NaN:", torch.isnan(loss).item(), "Is Inf:", torch.isinf(loss).item())
                loss.backward()  # calc gradients

                if model_type in ["complex_focus_only", "complex_focus_only_pix2pix"]:
                    if global_cur_iter[data_idx] < warm_up_iters:
                        print("\n------------------------")
                        # print(f"Target:\n    - min = {target_depth.min().item()}\n    - max = {target_depth.max().item()}\n    - mean = {target_depth.mean().item()}\n    - var = {target_depth.var().item()}\n    - nan = {torch.isnan(target_depth).any()}")
                        print(f"Prediction Output:\n    - min = {pred_depth.min().item()}\n    - max = {pred_depth.max().item()}\n    - mean = {pred_depth.mean().item()}\n    - var = {pred_depth.var().item()}\n    - nan = {torch.isnan(pred_depth).any()}")
                        # print(criterion[data_idx].get_dict(data_idx))
                        model.get_gradient_insight(data_idx)
                        print("------------------------\n")

                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # limit gradient
                    optimizer[data_idx].step()  # optimize weights with gradients
                else:
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # limit gradient
                    optimizer.step()  # optimize weights with gradients

                running_loss += loss.item()

                if global_cur_iter[data_idx] < warm_up_iters:
                    if model_type in ["complex_focus_only", "complex_focus_only_pix2pix"]:
                        if data_idx < 2:
                            warm_up_blend_1, warm_up_blend_2 = warm_up_blend[data_idx]
                            optimizer[data_idx].param_groups[0]['lr'] = warm_up_blend_1[global_cur_iter[data_idx]]
                            optimizer[data_idx].param_groups[1]['lr'] = warm_up_blend_2[global_cur_iter[data_idx]]
                        else:
                            warm_up_blend_1 = warm_up_blend[data_idx]
                            optimizer[data_idx].param_groups[0]['lr'] = warm_up_blend_1[global_cur_iter[data_idx]]
                    else:
                        optimizer.param_groups[0]['lr'] = warm_up_blend_1[global_cur_iter[data_idx]]
                        optimizer.param_groups[1]['lr'] = warm_up_blend_2[global_cur_iter[data_idx]]
                
                global_cur_iter[data_idx] += 1

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
                    if model_type in ["complex_focus_only", "complex_focus_only_pix2pix"]:
                        pred_depth = model.forward_part(input_img, data_idx)
                    else:
                        pred_depth = model(input_img)
                    loss = criterion[data_idx](pred_depth, target_depth) # criterion_1(pred_depth, target_depth) 
                    val_loss += loss.item()

                    if i == 0:
                        print(f"Output mean (model-part {data_idx}): {pred_depth.mean().item()}")
                        print(f"Output min (model-part {data_idx}): {pred_depth.min().item()}")
                        print(f"Output max (model-part {data_idx}): {pred_depth.max().item()}")

                        # Log first batch images
                        if model_type in ["complex_focus_only", "complex_focus_only_pix2pix"]:
                            # clip = data_idx == 1
                            prenorm_255 = data_idx == 1
                            val_img_log = inference_forward(input_img, lambda x:model.forward_part(x, data_idx), device, scale_to_256=True, clip=True, prenorm_255=prenorm_255)
                        else:
                            val_img_log = inference_forward(input_img, model, device, scale_to_256=True)

                        # if (val_img_log == 0).all():
                        #     raise Exception("Prediction is completely black")
                        # if np.isnan(val_img_log).any():
                        #     raise Exception("Prediction contains NaN values")


            avg_val_loss = val_loss / len(val_loader)
            if model_type in ["complex_focus_only", "complex_focus_only_pix2pix"]:
                cur_optimizer = optimizer[data_idx]
            else:
                cur_optimizer = optimizer

            if model_type in ["complex_focus_only", "complex_focus_only_pix2pix"] and data_idx < 2:
                log_dict = {
                                f"{data_idx}_val_loss": avg_val_loss,
                                f"{data_idx}_epoch": epoch + 1,
                                f"{data_idx}_lr encoder": cur_optimizer.param_groups[0]['lr'], # scheduler.get_last_lr()[0],
                                f"{data_idx}_lr decoder": cur_optimizer.param_groups[1]['lr'],
                                f"{data_idx}_sample_depth_map": wandb.Image(val_img_log) if val_img_log is not None else None
                            }
            else:
                log_dict = {
                                f"{data_idx}_val_loss": avg_val_loss,
                                f"{data_idx}_epoch": epoch + 1,
                                f"{data_idx}_lr": cur_optimizer.param_groups[0]['lr'], # scheduler.get_last_lr()[0],
                                f"{data_idx}_sample_depth_map": wandb.Image(val_img_log) if val_img_log is not None else None
                            }
            log_dict.update(criterion[data_idx].get_dict(data_idx))
            wandb.log(log_dict)

            if data_idx == len(datasets)-1:
                criterion[data_idx].step(epoch)

                # Save model
                if last_model:
                    os.remove(last_model)

                last_model = f"./checkpoints/{model_name}_epoch{epoch+1}.pth"

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
                if global_cur_iter[data_idx] >= warm_up_iters:
                    if model_type in ["complex_focus_only", "complex_focus_only_pix2pix"]:
                        base_scheduler.step()
                        complex_scheduler.step()
                        fusion_scheduler.step()
                    else:
                        scheduler.step()

                    # freeze encoder after warm up
                    for name, param in model.named_parameters():
                        if 'pretrained' in name:
                            param.requires_grad = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Depth Anything with wandb")
    parser.add_argument("--variation", help="Dataset variant: sound_baseline, sound_reflection, sound_diffraction, sound_combined")
    parser.add_argument("--input_type", default="osm", help="Defines the used Input -> 'osm', 'base_simulation'")
    parser.add_argument("--output_type", default="standard", help="Defines the Output -> 'standard', 'complex_only'")
    parser.add_argument("--model_name", help="Name for saving the model checkpoint")
    parser.add_argument("--model_type", default="depth_any", help="Type of model -> 'depth_any', 'complex_focus_only', 'complex_focus_only_pix2pix'")
    parser.add_argument("--encoder", default="vitb", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train(args.variation, args.input_type, args.output_type, args.model_name, args.model_type, args.encoder, args.batch_size, args.epochs, args.lr)


