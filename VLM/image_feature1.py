import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True)
        return x / (norm + self.eps) * self.scale

class DilatedConvBlock(nn.Module):
    """Dilated Convolutional Block with Residual Connection"""
    def __init__(self, in_channels, out_channels, dilation):
        super(DilatedConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              padding=dilation, dilation=dilation)
        self.norm = RMSNorm(out_channels)
        # Residual projection if in_channels != out_channels
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else None

    def forward(self, x):
        identity = x  # Save the original input for the residual connection
        y = F.gelu(self.conv(x))
        # Apply normalization: permute to [batch_size, height, width, channels] for RMSNorm
        y = self.norm(y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # Project the identity if channels don't match
        if self.residual_conv is not None:
            identity = self.residual_conv(identity)
        return identity + y  # Now the shapes match
class SparseTransformerBlock(nn.Module):
    """Dynamic Sparse Transformer Block"""
    def __init__(self, d_model, spatial_shape):
        super(SparseTransformerBlock, self).__init__()
        self.d_model = d_model
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=8)
        self.proj = nn.Linear(d_model, d_model)
        self.spatial_shape = spatial_shape  # (h_patches, w_patches)

    def forward(self, x):
        # x shape: [seq_len, batch_size, d_model]
        x_norm = self.norm1(x)
        # Compute gate values
        gate = torch.sigmoid(self.proj(x_norm))
        mean = gate.mean()
        std = gate.std()
        threshold = mean - std
        # Sparse selection
        keep = gate > threshold
        x_sparse = x_norm * keep
        # Attention
        attn_output, _ = self.attn(x_sparse, x_norm, x_norm)
        # Residual connection
        x = x + attn_output
        x = x + self.norm2(F.gelu(self.proj(x)))
        return x

class MultiScalePooling(nn.Module):
    """Hierarchical Spatial Aggregation with Multi-Scale Pooling"""
    def __init__(self, d_model, scales):
        super(MultiScalePooling, self).__init__()
        self.scales = scales
        self.pool_layers = nn.ModuleList([
            nn.AvgPool2d(kernel_size=2**s, stride=2**s) for s in scales
        ])
        self.proj_layers = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in scales
        ])
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        # x shape: [batch_size, d_model, h, w]
        features = []
        for pool, proj in zip(self.pool_layers, self.proj_layers):
            h = pool(x)
            h = self.norm(h)
            h = proj(h.flatten(2).transpose(1, 2))  # [batch_size, seq_len, d_model]
            features.append(h)
        # Attention-based fusion
        features = torch.stack(features, dim=1)  # [batch_size, scales, seq_len, d_model]
        query = features.mean(dim=1)  # Mean over scales
        key = features
        attn_scores = torch.einsum('bqd,bsvd->bqsv', query, key) / (self.d_model ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=1)
        fused_features = (attn_weights * features).sum(dim=1)
        return fused_features  # [batch_size, seq_len, d_model]

class CodebookQuantizer(nn.Module):
    """Discrete Residual Quantization with Codebook"""
    def __init__(self, d_model, K, temp=0.1):
        super(CodebookQuantizer, self).__init__()
        self.codebook = nn.Embedding(K, d_model)
        self.temp = temp

    def forward(self, h):
        # h shape: [batch_size, seq_len, d_model]
        distances = (h.unsqueeze(2) - self.codebook.weight).pow(2).sum(-1)  # [batch_size, seq_len, K]
        q_k = F.softmax(-distances / self.temp, dim=-1)
        z_e = torch.einsum('bqk,kd->bqd', q_k, self.codebook.weight)
        # Straight-through estimator
        z_q = h + (z_e - h).detach()
        return z_q

class DecoderBlock(nn.Module):
    """Spatial-Aware Decoder Block"""
    def __init__(self, d_model):
        super(DecoderBlock, self).__init__()
        self.norm = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=8)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, enc_kv):
        x_norm = self.norm(x)
        attn_output, _ = self.attn(x_norm, enc_kv, enc_kv)
        x = x + attn_output
        x = x + self.proj(F.gelu(self.norm(x)))
        return x

class GeneralizedFeatureExtractor(nn.Module):
    """Generalized Feature Extraction Network"""
    def __init__(self, in_channels, d_model, L, T, S, K, M, patch_size):
        super(GeneralizedFeatureExtractor, self).__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        # Multi-Scale Convolutional Encoding
        self.conv_blocks = nn.ModuleList([
            DilatedConvBlock(
                in_channels if l == 0 else d_model, 
                d_model, 
                dilation=2**l
            )
            for l in range(L)
        ])
        # Dynamic Vision Transformer Processing
        self.patch_embed = nn.Linear(d_model * patch_size ** 2, d_model)
        self.pos_embed = None  # Initialized in forward based on input size
        self.transformer_blocks = nn.ModuleList([
            SparseTransformerBlock(d_model, spatial_shape=None) for _ in range(T)
        ])
        # Hierarchical Spatial Aggregation
        self.ms_pooling = MultiScalePooling(d_model, scales=range(1, S+1))
        # Contrastive Latent Projection
        self.proj_img = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, d_model)
        )
        # Codebook Quantization
        self.quantizer = CodebookQuantizer(d_model, K)
        # Spatial-Aware Decoding
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model) for _ in range(M)
        ])
        self.output_conv = nn.Sequential(
            RMSNorm(d_model),
            nn.ConvTranspose2d(d_model, in_channels, kernel_size=patch_size, stride=patch_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: Input image tensor of shape [batch_size, in_channels, height, width]
        """
        batch_size, _, height, width = x.shape

        # Multi-Scale Convolutional Encoding
        for conv_block in self.conv_blocks:
            x = conv_block(x)  # x shape: [B, d_model, H, W]

        # Patch Embedding
        p = self.patch_size
        C = x.shape[1]  # Should be d_model
        x_patches = F.unfold(x, kernel_size=p, stride=p)  # [B, C*p*p, num_patches]
        x_patches = x_patches.transpose(1, 2)  # [B, num_patches, C*p*p]
        # Now apply the Linear layer
        embeddings = self.patch_embed(x_patches)  # [B, num_patches, d_model]
        seq_len = embeddings.shape[1]

        # Positional Embedding
        if self.pos_embed is None or self.pos_embed.shape[0] != seq_len:
            self.pos_embed = nn.Parameter(
                torch.zeros(seq_len, self.d_model, device=x.device)
            )
        embeddings = embeddings + self.pos_embed.unsqueeze(0)  # [B, seq_len, d_model]
        z = embeddings.transpose(0, 1)  # [seq_len, B, d_model]

        # Transformer Blocks
        for blk in self.transformer_blocks:
            z = blk(z)

        # Hierarchical Spatial Aggregation
        h = self.ms_pooling(x)  # [B, seq_len_pool, d_model]

        # Contrastive Latent Projection
        h_img = self.proj_img(h)
        # Normally, h_txt would come from a text encoder; omitted here

        # Discrete Residual Quantization
        z_q = self.quantizer(h_img)

        # Spatial-Aware Decoding
        # Initialize decoder input
        h_dec = torch.zeros_like(z_q)
        for blk in self.decoder_blocks:
            h_dec = blk(h_dec.transpose(0,1), z_q.transpose(0, 1))  # Ensure dimensions match

        # Pixel Reconstruction
        h_dec = h_dec.transpose(0, 1).transpose(1, 2)  # [B, seq_len_dec, d_model]
        num_patches = int(height / p) * int(width / p)
        h_dec = h_dec.contiguous().view(batch_size, num_patches, self.d_model)
        h_dec = h_dec.transpose(1, 2)  # [B, d_model, num_patches]
        # Fold back to image
        x_rec = F.fold(h_dec, output_size=(height, width), kernel_size=p, stride=p)
        x_rec = self.output_conv(x_rec)  # [B, in_channels, H, W]
        return z_q, x_rec

# Example usage:
if __name__ == "__main__":
    # Hyperparameters
    in_channels = 3
    d_model = 64
    L = 3  # Conv layers
    T = 2  # Transformer blocks
    S = 2  # Scales
    K = 512  # Codebook size
    M = 2  # Decoder steps
    patch_size = 4

    # Create model
    model = GeneralizedFeatureExtractor(in_channels, d_model, L, T, S, K, M, patch_size)
    # Generate a random image tensor
    img = torch.randn(8, in_channels, 64, 64)
    # Forward pass
    latent, reconstructed = model(img)
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstructed image shape: {reconstructed.shape}")