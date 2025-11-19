import torch
import torch.nn as nn
import torch.nn.functional as F

# Bulk of the UNet model for image colorization
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # First conv
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Second conv
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    



class UNet_Colorizer(nn.Module):
    def __init__(self, n_channels_in=2, n_channels_out=3):
        super(UNet_Colorizer, self).__init__()
        
        # =======================================================
        # == 1. CRITICAL MODIFICATION: Input Layer ==
        # This layer accepts your 2-channel concatenated tensor
        # (B&W Image, Attention Map)
        # =======================================================
        self.inc = DoubleConv(n_channels_in, 64)

        # --- Encoder (Down-sampling) ---
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        # --- Decoder (Up-sampling) ---
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)  # Takes 512 from up + 512 from skip

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)   # Takes 256 from up + 256 from skip

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)   # Takes 128 from up + 128 from skip
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)    # Takes 64 from up + 64 from skip

        # =======================================================
        # == 2. CRITICAL MODIFICATION: Output Layer ==
        # This final 1x1 conv outputs your 3 color channels
        # =======================================================
        self.outc = nn.Conv2d(64, n_channels_out, kernel_size=1)
        
        # Final activation (Tanh is common for colorization)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x is your [Batch_Size, 2, H, W] tensor
        
        # --- Encoder ---
        x1 = self.inc(x)     # -> 64 channels. This is your first skip connection
        x2 = self.down1(x1)  # -> 128 channels
        x3 = self.down2(x2)  # -> 256 channels
        x4 = self.down3(x3)  # -> 512 channels
        x5 = self.down4(x4)  # -> 1024 channels (Bottleneck)

        # --- Decoder (with Skip Connections) ---
        x = self.up1(x5)                      # Upsample
        x = torch.cat([x, x4], dim=1)         # Concatenate skip connection (x4)
        x = self.conv1(x)                     # DoubleConv

        x = self.up2(x)                       # Upsample
        x = torch.cat([x, x3], dim=1)         # Concatenate skip connection (x3)
        x = self.conv2(x)                     # DoubleConv

        x = self.up3(x)                       # Upsample
        x = torch.cat([x, x2], dim=1)         # Concatenate skip connection (x2)
        x = self.conv3(x)                     # DoubleConv
        
        x = self.up4(x)                       # Upsample
        x = torch.cat([x, x1], dim=1)         # Concatenate skip connection (x1)
        x = self.conv4(x)                     # DoubleConv

        # --- Output ---
        logits = self.outc(x)                 # Final 1x1 conv to get 3 channels
        output = self.tanh(logits)            # Apply Tanh
        
        return output
