import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        x = x / (rms + self.eps)
        x = x * self.scale
        return x

class RMSNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1, 1))
    def forward(self, x):
        rms = x.pow(2).mean(dim=[2, 3], keepdim=True).sqrt()  # [Batch, Channels, 1, 1]
        x = x / (rms + self.eps)
        x = x * self.scale
        return x

class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        padding = dilation
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=padding, dilation=dilation)
        self.act = nn.GELU()
        self.norm = RMSNorm2d(out_channels)
        self.downsample = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None
    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.act(x)
        x = self.norm(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x = x + residual
        return x

class MultiScaleConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, layers):
        super().__init__()
        self.layers = nn.ModuleList([
            DilatedConvBlock(in_channels if i == 0 else channels, channels, dilation=2**i)
            for i in range(layers)
        ])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)  # [Batch, Embed_dim, H/ps, W/ps]
        x = x.flatten(2).transpose(1, 2)  # [Batch, Num_patches, Embed_dim]
        return x

class DynamicTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class DynamicVisionTransformer(nn.Module):
    def __init__(self, dim, layers, num_heads):
        super().__init__()
        self.layers = nn.ModuleList([
            DynamicTransformerBlock(dim, num_heads) for _ in range(layers)
        ])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class HierarchicalSpatialAggregation(nn.Module):
    def __init__(self, channels, embed_dim, scales):
        super().__init__()
        self.scales = scales
        self.projs = nn.ModuleList([
            nn.Sequential(
                RMSNorm2d(channels),
                nn.Conv2d(channels, embed_dim, kernel_size=1)
            ) for _ in range(scales)
        ])
    def forward(self, x):
        hs = []
        for i, proj in enumerate(self.projs):
            scale_factor = 1 / (2 ** (i + 1))
            h = F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=False)
            h = proj(h)
            h = h.flatten(2).mean(-1)  # Global Average Pooling
            hs.append(h)
        h = torch.stack(hs, dim=1)  # [Batch, Scales, Embed_dim]
        attn_scores = torch.matmul(h, h.transpose(1, 2)) / h.size(-1) ** 0.5
        attn = F.softmax(attn_scores, dim=-1)
        h = torch.matmul(attn, h).sum(dim=1)  # Aggregated feature vector
        return h

class DiscreteResidualQuantization(nn.Module):
    def __init__(self, dim, codebook_size, temp=1.0):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, dim)
        self.temp = temp
        self.mlp = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim)
        )
    def forward(self, h):
        dist = (h.unsqueeze(1) - self.codebook.weight.unsqueeze(0)).pow(2).sum(-1)  # [Batch, Codebook_size]
        q = F.softmax(-dist / self.temp, dim=-1)  # [Batch, Codebook_size]
        z_e = torch.matmul(q, self.codebook.weight)  # [Batch, Embed_dim]
        z_q = z_e + self.mlp(h - z_e)  # Straight-through estimator
        return z_q

class GeneralizedFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, channels=64, conv_layers=4, embed_dim=256, patch_size=16,
                 transformer_layers=6, num_heads=8, scales=3, codebook_size=512):
        super().__init__()
        self.encoder = MultiScaleConvEncoder(in_channels, channels, conv_layers)
        self.patch_embed = PatchEmbedding(channels, embed_dim, patch_size)
        self.transformer = DynamicVisionTransformer(embed_dim, transformer_layers, num_heads)
        self.agg = HierarchicalSpatialAggregation(channels, embed_dim, scales)
        self.quant = DiscreteResidualQuantization(embed_dim, codebook_size)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, channels * (patch_size ** 2)),
            nn.GELU(),
            nn.Unflatten(1, (channels, patch_size, patch_size)),
            nn.ConvTranspose2d(channels, in_channels, kernel_size=patch_size, stride=patch_size),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.encoder(x)  # [Batch, Channels, Height, Width]
        patches = self.patch_embed(x)  # [Batch, Num_patches, Embed_dim]
        tokens = self.transformer(patches)  # [Batch, Num_patches, Embed_dim]
        h = self.agg(x)  # [Batch, Embed_dim]
        z_q = self.quant(h)  # [Batch, Embed_dim]
        recon = self.decoder(z_q)  # [Batch, Channels, Height, Width]
        return z_q, recon

# Testing the model with random input
if __name__ == "__main__":
    model = GeneralizedFeatureExtractor()
    input_tensor = torch.randn(1, 3, 224, 224)  # Batch size 1, RGB image of size 224x224
    z_q, recon = model(input_tensor)
    print("Latent vector shape:", z_q.shape)
    print("Reconstructed image shape:", recon.shape)