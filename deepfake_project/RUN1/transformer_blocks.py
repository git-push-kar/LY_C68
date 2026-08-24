"""
transformer_blocks.py
=====================
Vision Transformer components implemented from scratch.
Includes: PatchEmbedding, PositionalEncoding, MultiHeadSelfAttention,
          FeedForward, TransformerEncoderBlock, ViTEncoder.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """
    Splits an image into non-overlapping patches and linearly projects them
    to the model's hidden dimension.

    Args:
        image_size  : int  – side length of the (square) input image.
        patch_size  : int  – side length of each (square) patch.
        in_channels : int  – number of input channels (3 for RGB).
        embed_dim   : int  – projection dimension.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        assert image_size % patch_size == 0, (
            f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # A single Conv2d acts as the patch projection (kernel = stride = patch_size)
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)  →  (B, num_patches, embed_dim)
        x = self.projection(x)          # (B, embed_dim, H/P, W/P)
        x = rearrange(x, "b d h w -> b (h w) d")
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# Learnable Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """
    Adds a learnable positional embedding to each patch token.
    An extra CLS token position is included.

    Args:
        num_patches : int – number of patch tokens (excluding CLS).
        embed_dim   : int – embedding dimension.
        dropout     : float – dropout applied after adding positional encoding.
    """

    def __init__(self, num_patches: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        # +1 for the CLS token
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_patches, embed_dim)
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls, x], dim=1)           # (B, num_patches+1, D)
        x = x + self.pos_embedding
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """
    Standard scaled dot-product multi-head self-attention.

    Args:
        embed_dim   : int   – input/output dimension.
        num_heads   : int   – number of attention heads.
        dropout     : float – attention weight dropout.
        bias        : bool  – whether to use bias in projections.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ):
        B, N, D = x.shape
        # Project to Q, K, V
        qkv = self.qkv(x)                             # (B, N, 3*D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)             # (3, B, H, N, Dh)
        q, k, v = qkv.unbind(0)                       # each (B, H, N, Dh)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / self.scale  # (B, H, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v                                  # (B, H, N, Dh)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.out_proj(out)

        if return_attn:
            return out, attn
        return out


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network (two linear layers with GELU).

    Args:
        embed_dim   : int   – input/output dimension.
        hidden_dim  : int   – inner dimension (commonly 4 × embed_dim).
        dropout     : float – dropout applied inside the network.
    """

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Transformer Encoder Block
# ---------------------------------------------------------------------------

class TransformerEncoderBlock(nn.Module):
    """
    Standard Pre-LN Transformer encoder block:
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    Args:
        embed_dim   : int   – model dimension.
        num_heads   : int   – number of attention heads.
        mlp_ratio   : float – multiplier for FFN hidden dimension.
        dropout     : float – general dropout.
        attn_drop   : float – attention weight dropout.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout=attn_drop)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, int(embed_dim * mlp_ratio), dropout=dropout)
        self.drop_path = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ):
        # Self-attention with residual
        if return_attn:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attn=True)
            x = x + self.drop_path(attn_out)
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            attn_weights = None

        # FFN with residual
        x = x + self.drop_path(self.ffn(self.norm2(x)))

        if return_attn:
            return x, attn_weights
        return x


# ---------------------------------------------------------------------------
# Full ViT Encoder
# ---------------------------------------------------------------------------

class ViTEncoder(nn.Module):
    """
    Stacks multiple TransformerEncoderBlocks and returns the final token
    representations together with the CLS token.

    Args:
        image_size  : int   – input image side length.
        patch_size  : int   – patch side length.
        in_channels : int   – RGB = 3.
        embed_dim   : int   – model dimension.
        depth       : int   – number of encoder layers.
        num_heads   : int   – attention heads.
        mlp_ratio   : float – FFN expansion factor.
        dropout     : float – dropout rate.
        attn_drop   : float – attention dropout.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_enc = PositionalEncoding(num_patches, embed_dim, dropout=dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout, attn_drop)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        """
        Returns:
            cls_token   : (B, embed_dim)
            all_tokens  : (B, num_patches+1, embed_dim)
            attn_maps   : list of attention weight tensors if return_attn=True
        """
        x = self.patch_embed(x)      # (B, N, D)
        x = self.pos_enc(x)          # (B, N+1, D)

        attn_maps = []
        for block in self.blocks:
            if return_attn:
                x, attn = block(x, return_attn=True)
                attn_maps.append(attn)
            else:
                x = block(x)

        x = self.norm(x)
        cls_token = x[:, 0]          # (B, D)
        all_tokens = x               # (B, N+1, D)

        if return_attn:
            return cls_token, all_tokens, attn_maps
        return cls_token, all_tokens
