
import torch
import torch.nn as nn

class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)

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
        pred = self.fusion(x)
        pred = torch.clamp(pred, 0.0, 1.0)
        # pred = torch.sigmoid(pred)
        return pred

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super().__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

class UNetGenerator(nn.Module):
    """
    UNet Generator from official Pix2Pix Repo.

    Construct a Unet generator
    Parameters:
        input_nc (int)  -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                            image of size 128x128 will become of size 1x1 # at the bottleneck
        ngf (int)       -- the number of filters in the last conv layer
        norm_layer      -- normalization layer

    We construct the U-Net from the innermost layer to the outermost layer.
    It is a recursive process.
    """
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class ComplexFocusOnlyPix2Pix(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        
        self.pix2pix_baseline = UnetGenerator(1, 1, 8, 64, norm_layer=nn.BatchNorm2d, use_dropout=True)
        self.discr_baseline = PixelDiscriminator(1, ndf=64, norm_layer=nn.BatchNorm2d)

        self.pix2pix_complex = UnetGenerator(1, 1, 8, 64, norm_layer=nn.BatchNorm2d, use_dropout=True)
        self.discr_complex = PixelDiscriminator(1, ndf=64, norm_layer=nn.BatchNorm2d)

        self.fusion_head = FusionHead(input_channels=2)

    def forward(self, x):
        # shape: [B, 3, H, W]
        # print(f"input_img.shape = {x.shape}")
        base_x = self.pix2pix_baseline(x).unsqueeze(1)  # shape: [B, 1, H, W]
        # print(f"base_x.shape = {base_x.shape}")
        complex_x = self.pix2pix_complex(x).unsqueeze(1)  # shape: [B, 1, H, W]
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
        return self.pix2pix_baseline(x)

    def forward_complex(self, x):
        return self.pix2pix_complex(x)

    def train_baseline(self):
        for param in self.pix2pix_baseline.parameters():
            param.requires_grad = True

        for param in self.pix2pix_complex.parameters():
            param.requires_grad = False

        for param in self.fusion_head.parameters():
            param.requires_grad = False

    def train_complex(self):
        for param in self.pix2pix_baseline.parameters():
            param.requires_grad = False

        for param in self.pix2pix_complex.parameters():
            param.requires_grad = True

        for param in self.fusion_head.parameters():
            param.requires_grad = False

    def train_fusion_head(self):
        for param in self.pix2pix_baseline.parameters():
            param.requires_grad = False

        for param in self.pix2pix_complex.parameters():
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
            model = self.pix2pix_baseline
        elif idx == 1:
            model_name = "Complex"
            model = self.pix2pix_complex
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

    def loss(self, input_, pred_, target_, data_idx):
        if data_idx == 0:
            pass
        elif data_idx == 1:
            pass
        else:
            pass


