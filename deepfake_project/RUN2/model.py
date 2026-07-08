"""
model.py
========
HydraFakeNet:
    CLIP ViT  (pretrained, prebuilt)   →  768-dim CLS token
         ↓
    FeatureProjection                  (from scratch)
         ↓
    ClassificationHead                 (from scratch, transformer-based)
    ReasoningDecoder                   (from scratch, GRU-based)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel


# ---------------------------------------------------------------------------
# Feature Projection  —  from scratch
# ---------------------------------------------------------------------------

class FeatureProjection(nn.Module):
    """
    Projects CLIP's 768-dim CLS token into model_dim (512).
    Fully randomly initialized, learned only from HydraFake data.
    """

    def __init__(self, clip_dim: int = 768, model_dim: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(clip_dim, model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, model_dim),
        )
        self.norm    = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.dropout(self.net(x)))


# ---------------------------------------------------------------------------
# From-scratch Multi-Head Self-Attention
# ---------------------------------------------------------------------------

class ScratchMHSA(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv       = nn.Linear(dim, dim * 3, bias=True)
        self.out       = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        nn.init.trunc_normal_(self.qkv.weight, std=0.02)
        nn.init.trunc_normal_(self.out.weight, std=0.02)
        nn.init.zeros_(self.qkv.bias)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv  = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv  = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        attn = self.attn_drop(attn)
        out  = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.out(out)


# ---------------------------------------------------------------------------
# From-scratch Feed-Forward Network
# ---------------------------------------------------------------------------

class ScratchFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# From-scratch Transformer Block
# ---------------------------------------------------------------------------

class ScratchTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.1, attn_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = ScratchMHSA(dim, num_heads, attn_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = ScratchFFN(dim, int(dim * mlp_ratio), dropout)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# From-scratch Classification Head
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    """
    Expands projected CLS token into learnable query tokens,
    runs through from-scratch transformer blocks, collapses to 2 logits.
    Every weight randomly initialized and learned from HydraFake labels.
    """

    def __init__(self, model_dim: int = 512, num_heads: int = 8,
                 depth: int = 4, num_queries: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.2,
                 attn_drop: float = 0.0):
        super().__init__()
        self.num_queries  = num_queries
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, model_dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

        self.expand = nn.Linear(model_dim, num_queries * model_dim)
        self.blocks = nn.ModuleList([
            ScratchTransformerBlock(model_dim, num_heads, mlp_ratio,
                                    dropout, attn_drop)
            for _ in range(depth)
        ])
        self.norm       = nn.LayerNorm(model_dim)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, 2),
        )
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B       = x.size(0)
        queries = self.expand(x).view(B, self.num_queries, -1)
        queries = queries + self.query_tokens.expand(B, -1, -1)
        for block in self.blocks:
            queries = block(queries)
        out = self.norm(queries).mean(dim=1)
        return self.classifier(self.dropout(out))


# ---------------------------------------------------------------------------
# From-scratch GRU Reasoning Decoder
# ---------------------------------------------------------------------------

class ReasoningDecoder(nn.Module):
    """
    GRU decoder conditioned on projected visual feature.
    Token embedding, GRU, output projection all randomly initialized
    and learned from reasoning text in HydraFake JSON files.
    """

    def __init__(self, vocab_size: int, model_dim: int = 512,
                 hidden_dim: int = 512, num_layers: int = 2,
                 max_length: int = 128, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers
        self.max_length  = max_length
        self.vocab_size  = vocab_size

        self.h0_proj     = nn.Linear(model_dim, hidden_dim * num_layers)
        self.token_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.gru         = nn.GRU(
            input_size  = hidden_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.out_proj = nn.Linear(hidden_dim, vocab_size)
        self.dropout  = nn.Dropout(dropout)

        nn.init.trunc_normal_(self.token_embed.weight, std=0.02)
        nn.init.zeros_(self.token_embed.weight[0])
        nn.init.trunc_normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

    def _h0(self, feat: torch.Tensor) -> torch.Tensor:
        B  = feat.size(0)
        h0 = self.h0_proj(feat).view(B, self.num_layers, self.hidden_dim)
        return torch.tanh(h0.permute(1, 0, 2).contiguous())

    def forward(self, feat: torch.Tensor,
                target_ids: torch.Tensor) -> torch.Tensor:
        h      = self._h0(feat)
        emb    = self.dropout(self.token_embed(target_ids))
        out, _ = self.gru(emb, h)
        return self.out_proj(out)

    @torch.no_grad()
    def generate(self, feat: torch.Tensor, bos_id: int, eos_id: int,
                 tokenizer=None) -> list:
        B         = feat.size(0)
        h         = self._h0(feat)
        current   = torch.full((B, 1), bos_id, dtype=torch.long,
                               device=feat.device)
        generated = [[] for _ in range(B)]
        finished  = [False] * B

        for _ in range(self.max_length):
            emb      = self.token_embed(current)
            out, h   = self.gru(emb, h)
            next_tok = self.out_proj(out[:, 0, :]).argmax(dim=-1)
            for b in range(B):
                if not finished[b]:
                    t = next_tok[b].item()
                    if t == eos_id:
                        finished[b] = True
                    else:
                        generated[b].append(t)
            if all(finished):
                break
            current = next_tok.unsqueeze(1)

        if tokenizer is not None:
            return [tokenizer.decode(ids, skip_special_tokens=True)
                    for ids in generated]
        return generated


# ---------------------------------------------------------------------------
# HydraFakeNet  —  full model
# ---------------------------------------------------------------------------

class HydraFakeNet(nn.Module):
    """
    CLIP ViT encodes the image (prebuilt, pretrained).
    All downstream components are from scratch.

    freeze_mode:
        'all'     — freeze all CLIP weights
        'partial' — freeze early blocks, unfreeze last unfreeze_last_n
        'none'    — fine-tune full CLIP encoder
    """

    CLIP_DIM = 768

    def __init__(
        self,
        clip_model_name:   str   = "openai/clip-vit-base-patch32",
        freeze_mode:       str   = "partial",
        unfreeze_last_n:   int   = 4,
        model_dim:         int   = 512,
        cls_depth:         int   = 4,
        cls_heads:         int   = 8,
        cls_queries:       int   = 8,
        mlp_ratio:         float = 4.0,
        vocab_size:        int   = 30522,
        reasoning_hidden:  int   = 512,
        reasoning_layers:  int   = 2,
        max_reasoning_len: int   = 128,
        dropout:           float = 0.1,
        attn_drop:         float = 0.0,
    ):
        super().__init__()

        # ── CLIP ViT  (pretrained, prebuilt) ────────────────────────────
        self.clip_encoder = CLIPVisionModel.from_pretrained(clip_model_name)
        self._freeze_clip(freeze_mode, unfreeze_last_n)

        # ── From-scratch components ──────────────────────────────────────
        self.feature_proj = FeatureProjection(self.CLIP_DIM, model_dim, dropout)

        self.cls_head = ClassificationHead(
            model_dim  = model_dim,
            num_heads  = cls_heads,
            depth      = cls_depth,
            num_queries= cls_queries,
            mlp_ratio  = mlp_ratio,
            dropout    = dropout,
            attn_drop  = attn_drop,
        )

        self.reasoning_decoder = ReasoningDecoder(
            vocab_size  = vocab_size,
            model_dim   = model_dim,
            hidden_dim  = reasoning_hidden,
            num_layers  = reasoning_layers,
            max_length  = max_reasoning_len,
            dropout     = dropout,
        )

    def _freeze_clip(self, mode: str, unfreeze_last_n: int):
        if mode == "all":
            for p in self.clip_encoder.parameters():
                p.requires_grad = False

        elif mode == "partial":
            for p in self.clip_encoder.parameters():
                p.requires_grad = False
            blocks = self.clip_encoder.vision_model.encoder.layers
            for block in blocks[-unfreeze_last_n:]:
                for p in block.parameters():
                    p.requires_grad = True
            for p in self.clip_encoder.vision_model.post_layernorm.parameters():
                p.requires_grad = True

        elif mode == "none":
            for p in self.clip_encoder.parameters():
                p.requires_grad = True

        frozen    = sum(p.numel() for p in self.clip_encoder.parameters()
                        if not p.requires_grad)
        trainable = sum(p.numel() for p in self.clip_encoder.parameters()
                        if p.requires_grad)
        scratch   = sum(p.numel() for n, p in self.named_parameters()
                        if "clip_encoder" not in n)
        print(f"CLIP   — frozen: {frozen:,}  |  trainable: {trainable:,}")
        print(f"Scratch heads — trainable: {scratch:,}")

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        clip_out  = self.clip_encoder(pixel_values=images)
        cls_token = clip_out.pooler_output        # (B, 768)
        return self.feature_proj(cls_token)       # (B, 512)

    def forward(self, images: torch.Tensor,
                reasoning_ids: torch.Tensor = None) -> dict:
        projected  = self._encode(images)
        cls_logits = self.cls_head(projected)
        out        = {"cls_logits": cls_logits}
        if reasoning_ids is not None:
            out["reasoning_logits"] = self.reasoning_decoder(
                projected, reasoning_ids)
        return out

    @torch.no_grad()
    def predict(self, images: torch.Tensor, bos_id: int, eos_id: int,
                tokenizer=None) -> dict:
        self.eval()
        projected   = self._encode(images)
        cls_logits  = self.cls_head(projected)
        probs       = F.softmax(cls_logits, dim=-1)
        pred_labels = cls_logits.argmax(dim=-1)
        pred_confs  = probs.gather(1, pred_labels.unsqueeze(1)).squeeze(1)
        reasoning   = self.reasoning_decoder.generate(
            projected, bos_id, eos_id, tokenizer)
        return {
            "labels":      pred_labels.cpu().tolist(),
            "confidences": pred_confs.cpu().tolist(),
            "reasoning":   reasoning,
        }