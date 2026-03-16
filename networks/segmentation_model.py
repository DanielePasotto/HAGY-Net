import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from .harmonization_network import HarmNet

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
             g1 = TF.resize(g1, size=x1.shape[2:])
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class GatedFusionBlock(nn.Module):
    def __init__(self, channels):
        super(GatedFusionBlock, self).__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels * 2, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        if x1.shape != x2.shape:
             x2 = TF.resize(x2, size=x1.shape[2:])
             
        cat = torch.cat((x1, x2), dim=1)
        alpha = self.gate_conv(cat)
        
        return (x1 * alpha) + (x2 * (1 - alpha))

class AsymmetricGatedYNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(AsymmetricGatedYNet, self).__init__()
        
        self.encoder1_layers = nn.ModuleList()
        self.encoder2_layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        curr = in_channels
        for f in features:
            self.encoder1_layers.append(DoubleConv(curr, f))
            self.encoder2_layers.append(DoubleConv(curr, f))
            curr = f

        self.bottleneck_fusion = GatedFusionBlock(features[-1])
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        self.ups = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.fusion_blocks = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        
        reversed_features = features[::-1]
        bottleneck_out = features[-1] * 2 
        
        for f in reversed_features:
            self.ups.append(nn.ConvTranspose2d(bottleneck_out, f, 2, 2))
            self.attentions.append(AttentionBlock(F_g=f, F_l=f, F_int=f//2))
            self.fusion_blocks.append(GatedFusionBlock(f))
            self.dec_convs.append(DoubleConv(f * 2, f))
            bottleneck_out = f

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x1, x2):
        skips1 = []
        skips2 = []

        for layer in self.encoder1_layers:
            x1 = layer(x1); skips1.append(x1); x1 = self.pool(x1)
            
        for layer in self.encoder2_layers:
            x2 = layer(x2); skips2.append(x2); x2 = self.pool(x2)

        fused_bn = self.bottleneck_fusion(x1, x2)
        x = self.bottleneck(fused_bn)
        skips1 = skips1[::-1]
        skips2 = skips2[::-1]

        for idx in range(len(self.ups)):
            g = self.ups[idx](x)
            s1 = skips1[idx]
            s2 = skips2[idx]
            
            if g.shape != s1.shape: g = TF.resize(g, size=s1.shape[2:])

            s2_filtered = self.attentions[idx](g=g, x=s2)
            fused_skip = self.fusion_blocks[idx](s1, s2_filtered)
            concat = torch.cat((fused_skip, g), dim=1)
            x = self.dec_convs[idx](concat)

        return self.final_conv(x)

class HAGYNet(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int, device: torch.device, weights_path: str) -> None:
        super(HAGYNet, self).__init__()
        self.harmnet = HarmNet(input_shape, hidden_units, input_shape).to(device)
        self.ynet = AsymmetricGatedYNet(input_shape, output_shape).to(device)
      
        self.harmnet.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
            
    def forward(self, x):
        proc = self.harmnet(x)
        out = self.ynet(proc, x)
        return out