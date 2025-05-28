
import torch
import torch.nn as nn

from depth_anything_v2.dpt import DepthAnythingV2

def init_decoder_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class FusionHead(nn.Module):
    def __init__(self, input_channels, hidden_size=64):
        super(FusionHead, self).__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(input_channels, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, 1, kernel_size=1) 
        )

    def forward(self, x):
        return torch.sigmoid(self.fusion(x))

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
        # self.phys_anything_baseline.depth_head.apply(init_decoder_weights)
        # if self.baseline_model_path:
        #     self.phys_anything_baseline.load_state_dict(torch.load(self.baseline_model_path, map_location='cpu'))

        self.phys_anything_complex = DepthAnythingV2(**model_configs[encoder])
        # self.phys_anything_complex.depth_head.apply(init_decoder_weights)
        # if self.complex_model_path:
        #     self.phys_anything_complex.load_state_dict(torch.load(self.complex_model_path, map_location='cpu'))

        self.fusion_head = FusionHead(input_channels=2)

    def forward(self, x):
        # shape: [B, 3, H, W]
        # print(f"input_img.shape = {x.shape}")
        base_x = self.phys_anything_baseline(x).unsqueeze(1)  # shape: [B, 1, H, W]
        # print(f"base_x.shape = {base_x.shape}")
        complex_x = self.phys_anything_complex(x).unsqueeze(1)  # shape: [B, 1, H, W]
        # complex_x = (complex_x - 0.0) / (255.0 - 0.0)
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

    def get_gradient_insight(self, idx):
        """
        Gives some values about the gradient. Call this function directly after loss.backward().

        Idx:
        0 = Base-Part
        1 = Complex-Part
        2 = Fusion-Part -> Uses also Base+Complex parts
        """
        if idx == 0:
            model_name = "Baseline"
            model = self.phys_anything_baseline
        elif idx == 1:
            model_name = "Complex"
            model = self.phys_anything_complex
        else:
            model_name = "Fusion Head"
            model = self.fusion_head

        gradient_mean = 0
        gradient_min = 0
        gradient_max = 0
        requires_grad = 0
        nan_or_inf = 0
        requires_grad_but_no_forward_pass = 0
        counter = 0
        all_values = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradient_mean += param.grad.mean().item()
                gradient_min += param.grad.min().item()
                gradient_max += param.grad.max().item()
                counter += 1
            else:
                nan_or_inf += 1
            requires_grad += int(param.requires_grad)
            all_values += 1

            if param.requires_grad and param.grad is None:
                requires_grad_but_no_forward_pass += 1
        print(f"Gradient Insight ({model_name}):\
                                \n    - grad mean = {gradient_mean/counter:0.12f}\
                                \n    - min = {gradient_min/counter:0.12f}\
                                \n    - max = {gradient_max/counter:0.12f}\
                                \n    - requires_grad = {int((requires_grad/all_values)*100)}% ({requires_grad})\
                                \n    - nans/infs = {int((nan_or_inf/all_values)*100)}% ({nan_or_inf})\
                                \n    - requires grad but no forward pass = {int((requires_grad_but_no_forward_pass/all_values)*100)}% ({requires_grad_but_no_forward_pass})")




