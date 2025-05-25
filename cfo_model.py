
import torch
import torch.nn as nn

from depth_anything_v2.dpt import DepthAnythingV2

class FusionHead(nn.Module):
    def __init__(self, input_channels, hidden_size=64):
        super(FusionHead, self).__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(input_channels, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, 1, kernel_size=1) 
        )

    def forward(self, x):
        return self.fusion(x)

class ComplexFocusOnly(nn.Module):
    def __init__(self, encoder):
        super().__init__()

        model_configs = {
                        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
                        }
        
        self.phys_anything_baseline = DepthAnythingV2(**model_configs[encoder])
        # if self.baseline_model_path:
        #     self.phys_anything_baseline.load_state_dict(torch.load(self.baseline_model_path, map_location='cpu'))

        self.phys_anything_complex = DepthAnythingV2(**model_configs[encoder])
        # if self.complex_model_path:
        #     self.phys_anything_complex.load_state_dict(torch.load(self.complex_model_path, map_location='cpu'))

        self.fusion_head = FusionHead(input_channels=2)

    def forward(self, x):
        # shape: [B, 3, H, W]
        # print(f"input_img.shape = {x.shape}")
        base_x = self.phys_anything_baseline(x).unsqueeze(1)  # shape: [B, 1, H, W]
        # print(f"base_x.shape = {base_x.shape}")
        complex_x = self.phys_anything_complex(x).unsqueeze(1)  # shape: [B, 1, H, W]
        # print(f"complex_x.shape = {complex_x.shape}")

        combined = torch.cat([base_x, complex_x], dim=1)  # shape: [B, 2, H, W]
        # print(f"combined.shape = {combined.shape}")
        pred = self.fusion_head(combined)
        if len(pred.shape) == 4:
            pred = pred.squeeze(1)
        # pred = base_x - complex_x

        return pred

    def forward_baseline(self, x):
        return self.phys_anything_baseline(x)

    def forward_complex(self, x):
        return self.phys_anything_complex(x)

    def train_baseline(self):
        for param in self.phys_anything_baseline.parameters():
            param.requires_grad = True

        for param in self.phys_anything_complex.parameters():
            param.requires_grad = False

        for param in self.fusion_head.parameters():
            param.requires_grad = False

    def train_complex(self):
        for param in self.phys_anything_baseline.parameters():
            param.requires_grad = False

        for param in self.phys_anything_complex.parameters():
            param.requires_grad = True

        for param in self.fusion_head.parameters():
            param.requires_grad = False

    def train_fusion_head(self):
        for param in self.phys_anything_baseline.parameters():
            param.requires_grad = False

        for param in self.phys_anything_complex.parameters():
            param.requires_grad = False

        for param in self.fusion_head.parameters():
            param.requires_grad = True

    def switch_train(self, idx):
        """
        Switches to another part to train.
        0 = Base-Part
        1 = Complex-Part
        2 = Fusion-Part
        """
        if idx == 0:
            self.train_baseline()
        elif idx == 1:
            self.train_complex()
        elif idx == 2:
            self.train_fusion_head()

    def forward_part(self, x, idx):
        """
        Forwards only one of the parts.
        0 = Base-Part
        1 = Complex-Part
        2 = Fusion-Part -> Uses also Base+Complex parts
        """
        if idx == 0:
            pred = self.forward_baseline(x)
        elif idx == 1:
            pred = self.forward_complex(x)
        elif idx == 2:
            pred = self.forward(x)

        return pred




