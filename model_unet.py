import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet3D, self).__init__()
        
        # Encoder (Using Strided Convolutions instead of MaxPool to avoid dimension bugs)
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(32, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(32, affine=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.down1 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1, bias=False) # 338 -> 169
        
        self.enc2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(64, affine=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.down2 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1, bias=False) # 169 -> 85 (or safe downsample)
        
        self.enc3 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(128, affine=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.down3 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(256, affine=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # Decoder
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(128, affine=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(64, affine=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(32, affine=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.out_conv = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        d_1 = self.down1(e1)
        
        e2 = self.enc2(d_1)
        d_2 = self.down2(e2)
        
        e3 = self.enc3(d_2)
        d_3 = self.down3(e3)
        
        b = self.bottleneck(d_3)
        
        # Decoder with dynamic size matching
        u3 = self.up3(b)
        if u3.shape[2:] != e3.shape[2:]:
            u3 = F.interpolate(u3, size=e3.shape[2:], mode='trilinear', align_corners=False)
        u3 = torch.cat([u3, e3], dim=1)
        de3 = self.dec3(u3)
        
        u2 = self.up2(de3)
        if u2.shape[2:] != e2.shape[2:]:
            u2 = F.interpolate(u2, size=e2.shape[2:], mode='trilinear', align_corners=False)
        u2 = torch.cat([u2, e2], dim=1)
        de2 = self.dec2(u2)
        
        u1 = self.up1(de2)
        if u1.shape[2:] != e1.shape[2:]:
            u1 = F.interpolate(u1, size=e1.shape[2:], mode='trilinear', align_corners=False)
        u1 = torch.cat([u1, e1], dim=1)
        de1 = self.dec1(u1)
        
        return self.out_conv(de1)
