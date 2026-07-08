"""
model.py
========
HydraFakeNet  —  domain-generalizable deepfake detector.

Architecture:
    CLIP ViT  (pretrained)          →  768-dim CLS token
         ↓
    FeatureProjection               →  256-dim  (from scratch)
         ↓  concatenated with ↓
    FFTBranch                       →  256-dim  (from scratch)
         ↓
    [512-dim fused feature]
         ↓
    ClassificationHead              →  2 logits  (from scratch)
    ReasoningDecoder                →  kept for inference only, NOT trained

Key changes vs original:
  - FFTBranch added: captures spectral GAN fingerprints that generalize
    across unseen generators better than pixel-space features alone.
  - DropPath (stochastic depth) added to transformer blocks: stronger
    regularization than Dropout alone, prevents head from memorizing
    generator-specific artifacts.
  - ClassificationHead default depth reduced to 2, queries to 4:
    less capacity = less overfitting to training GANs.
  - ReasoningDecoder kept for inference/predict() but forward() no
    longer accepts reasoning_ids during training, so GRU gradients
    never corrupt the classification signal.
  - model_dim is now the FUSED dim (256 CLIP + 256 FFT = 512).
    FeatureProjection outputs 256, not model_dim, to match.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel


# ---------------------------------------------------------------------------
# DropPath (Stochastic Depth)
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    """
    Randomly drops entire residual paths during training.
    Stronger regularizer than Dropout for transformer blocks —
    prevents each block from memorizing generator-specific patterns.
    """
    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        # shape: (B, 1, 1, ...) so it broadcasts over all spatial dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask  = torch.bernoulli(
            torch.full(shape, keep_prob, device=x.device, dtype=x.dtype)
        )
        return x * mask / keep_prob


# ---------------------------------------------------------------------------
# FFT Frequency Branch  —  from scratch
# ---------------------------------------------------------------------------

class FFTBranch(nn.Module):
    """
    Extracts frequency-domain features via 2D FFT magnitude spectrum.

    WHY THIS HELPS:
      Different GAN architectures (StyleGAN, DALL-E, Midjourney, etc.)
      leave completely different spatial artifacts in pixel space, making
      pixel-based features GAN-specific and non-generalizable.
      However, ALL GAN/diffusion models share spectral fingerprints:
        - GAN upsampling grid artifacts appear as spikes in FFT
        - Unnatural frequency distributions vs real camera images
        - Diffusion model noise scheduling leaves frequency residuals
      These spectral features are more universal across unseen generators.

    INPUT:  (B, 3, H, W)  — CLIP-normalized image tensor
    OUTPUT: (B, out_dim)  — frequency feature vector
    """

    def __init__(self, out_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            # Stage 1: local frequency patterns
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),                    # 224 → 112

            # Stage 2: mid-frequency patterns
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),                    # 112 → 56

            # Stage 3: global frequency summary
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(4),            # → 4×4 spatial
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 2D FFT per channel, take log-compressed magnitude
        fft = torch.fft.fft2(x, norm="ortho")           # complex (B,3,H,W)
        mag = torch.abs(fft)                              # magnitude
        mag = torch.log1p(mag)                            # compress dynamic range
        mag = torch.fft.fftshift(mag, dim=(-2, -1))      # center DC component
        # mag is now (B, 3, H, W) real tensor — same spatial size as input
        feat = self.conv(mag)
        return self.norm(self.proj(feat))


# ---------------------------------------------------------------------------
# Feature Projection  —  from scratch
# ---------------------------------------------------------------------------

class FeatureProjection(nn.Module):
    """
    Projects CLIP's 768-dim CLS token to proj_dim (default 256).
    Output is concatenated with FFTBranch output to form the 512-dim
    fused feature that feeds the classification head.
    """

    def __init__(self, clip_dim: int = 768, proj_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(clip_dim, proj_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim * 2, proj_dim),
        )
        self.norm    = nn.LayerNorm(proj_dim)
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
    def __init__(self, dim: int, num_heads: int,
                 attn_drop: float = 0.0):
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
        qkv  = self.qkv(x).reshape(
            B, N, 3, self.num_heads, self.head_dim)
        qkv  = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = F.softmax(
            (q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        attn = self.attn_drop(attn)
        out  = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.out(out)


# ---------------------------------------------------------------------------
# From-scratch Feed-Forward Network
# ---------------------------------------------------------------------------

class ScratchFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int,
                 dropout: float = 0.1):
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
# From-scratch Transformer Block  with DropPath
# ---------------------------------------------------------------------------

class ScratchTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int,
                 mlp_ratio: float = 4.0,
                 dropout: float   = 0.1,
                 attn_drop: float = 0.0,
                 drop_path: float = 0.1):
        super().__init__()
        self.norm1     = nn.LayerNorm(dim)
        self.attn      = ScratchMHSA(dim, num_heads, attn_drop)
        self.norm2     = nn.LayerNorm(dim)
        self.ffn       = ScratchFFN(dim, int(dim * mlp_ratio), dropout)
        # DropPath replaces plain Dropout on residual paths:
        # during training each block is randomly skipped entirely,
        # forcing other blocks to learn independently — better generalization.
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# From-scratch Classification Head
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    """
    Expands the fused feature into learnable query tokens,
    runs through from-scratch transformer blocks, collapses to 2 logits.

    Default depth=2, queries=4 (down from 4/8):
      Less capacity deliberately reduces overfitting to training GANs.
    """

    def __init__(self, model_dim: int   = 512,
                 num_heads: int         = 8,
                 depth: int             = 2,
                 num_queries: int       = 4,
                 mlp_ratio: float       = 4.0,
                 dropout: float         = 0.4,
                 attn_drop: float       = 0.0,
                 drop_path: float       = 0.1):
        super().__init__()
        self.num_queries  = num_queries
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_queries, model_dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

        self.expand = nn.Linear(model_dim, num_queries * model_dim)

        # Linearly increasing drop_path rate across blocks
        dpr = [drop_path * i / max(depth - 1, 1)
               for i in range(depth)]
        self.blocks = nn.ModuleList([
            ScratchTransformerBlock(
                model_dim, num_heads, mlp_ratio,
                dropout, attn_drop, dpr[i])
            for i in range(depth)
        ])
        self.norm    = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
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
    Used ONLY for inference/predict() — never receives gradients
    during training (train.py no longer passes reasoning_ids to forward).

    Kept in the architecture so that checkpoints remain compatible and
    inference.py continues to work unchanged.
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
        self.token_embed = nn.Embedding(
            vocab_size, hidden_dim, padding_idx=0)
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
        h0 = self.h0_proj(feat).view(
            B, self.num_layers, self.hidden_dim)
        return torch.tanh(h0.permute(1, 0, 2).contiguous())

    def forward(self, feat: torch.Tensor,
                target_ids: torch.Tensor) -> torch.Tensor:
        h      = self._h0(feat)
        emb    = self.dropout(self.token_embed(target_ids))
        out, _ = self.gru(emb, h)
        return self.out_proj(out)

    @torch.no_grad()
    def generate(self, feat: torch.Tensor,
                 bos_id: int, eos_id: int,
                 tokenizer=None) -> list:
        B         = feat.size(0)
        h         = self._h0(feat)
        current   = torch.full(
            (B, 1), bos_id, dtype=torch.long, device=feat.device)
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
            return [
                tokenizer.decode(ids, skip_special_tokens=True)
                for ids in generated
            ]
        return generated


# ---------------------------------------------------------------------------
# HydraFakeNet  —  full model
# ---------------------------------------------------------------------------

class HydraFakeNet(nn.Module):
    """
    Dual-branch deepfake detector.

    Branch 1 — CLIP (pixel space):
        Pretrained CLIP ViT encodes semantic/texture features.
        Partially fine-tuned (last N blocks unfrozen).

    Branch 2 — FFT (frequency space):
        Raw FFT magnitude spectrum processed by a small CNN.
        Captures spectral GAN fingerprints that generalize to
        unseen generators better than pixel features alone.

    Both branches project to 256-dim and are concatenated → 512-dim
    fused feature that feeds the classification head.

    freeze_mode:
        'all'     — freeze all CLIP weights
        'partial' — freeze early blocks, unfreeze last unfreeze_last_n
        'none'    — fine-tune full CLIP encoder

    model_dim:
        The FUSED feature dimension = clip_proj_dim + fft_out_dim.
        Default 512 = 256 + 256. Do not change without updating both
        FeatureProjection and FFTBranch out_dim accordingly.
    """

    CLIP_DIM     = 768   # CLIP ViT-B/32 CLS token dimension
    CLIP_PROJ    = 256   # CLIP projection output (half of model_dim)
    FFT_OUT      = 256   # FFT branch output (half of model_dim)

    def __init__(
        self,
        clip_model_name:   str   = "openai/clip-vit-base-patch32",
        freeze_mode:       str   = "partial",
        unfreeze_last_n:   int   = 4,
        model_dim:         int   = 512,    # must equal CLIP_PROJ + FFT_OUT
        cls_depth:         int   = 2,
        cls_heads:         int   = 8,
        cls_queries:       int   = 4,
        mlp_ratio:         float = 4.0,
        vocab_size:        int   = 30522,
        reasoning_hidden:  int   = 512,
        reasoning_layers:  int   = 2,
        max_reasoning_len: int   = 128,
        dropout:           float = 0.4,
        attn_drop:         float = 0.0,
        drop_path:         float = 0.1,
    ):
        super().__init__()

        assert model_dim == self.CLIP_PROJ + self.FFT_OUT, (
            f"model_dim ({model_dim}) must equal "
            f"CLIP_PROJ ({self.CLIP_PROJ}) + FFT_OUT ({self.FFT_OUT})"
        )

        # ── CLIP ViT  (pretrained) ───────────────────────────────────
        self.clip_encoder = CLIPVisionModel.from_pretrained(
            clip_model_name)
        self._freeze_clip(freeze_mode, unfreeze_last_n)

        # ── Dual branches (from scratch) ─────────────────────────────
        self.feature_proj = FeatureProjection(
            self.CLIP_DIM, self.CLIP_PROJ, dropout)

        self.fft_branch = FFTBranch(
            out_dim=self.FFT_OUT, dropout=dropout)

        # ── Classification head (from scratch) ───────────────────────
        self.cls_head = ClassificationHead(
            model_dim  = model_dim,
            num_heads  = cls_heads,
            depth      = cls_depth,
            num_queries= cls_queries,
            mlp_ratio  = mlp_ratio,
            dropout    = dropout,
            attn_drop  = attn_drop,
            drop_path  = drop_path,
        )

        # ── Reasoning decoder (inference only, not trained) ──────────
        self.reasoning_decoder = ReasoningDecoder(
            vocab_size  = vocab_size,
            model_dim   = model_dim,
            hidden_dim  = reasoning_hidden,
            num_layers  = reasoning_layers,
            max_length  = max_reasoning_len,
            dropout     = dropout,
        )

    # ------------------------------------------------------------------

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
            for p in (self.clip_encoder
                          .vision_model.post_layernorm.parameters()):
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

    # ------------------------------------------------------------------

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Returns the 512-dim fused feature (CLIP 256 || FFT 256).
        """
        clip_out  = self.clip_encoder(pixel_values=images)
        clip_feat = self.feature_proj(clip_out.pooler_output)  # (B, 256)
        fft_feat  = self.fft_branch(images)                    # (B, 256)
        return torch.cat([clip_feat, fft_feat], dim=-1)        # (B, 512)

    # ------------------------------------------------------------------

    def forward(self, images: torch.Tensor,
                reasoning_ids: torch.Tensor = None) -> dict:
        """
        Training forward pass.

        reasoning_ids is accepted as a parameter for API compatibility
        but is intentionally IGNORED — the GRU decoder is not trained.
        All gradient signal comes from classification loss only.

        Returns:
            dict with key "cls_logits" (B, 2)
        """
        projected  = self._encode(images)
        cls_logits = self.cls_head(projected)
        return {"cls_logits": cls_logits}

    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, images: torch.Tensor,
                bos_id: int, eos_id: int,
                tokenizer=None) -> dict:
        """
        Inference-time prediction with optional reasoning generation.
        """
        self.eval()
        projected   = self._encode(images)
        cls_logits  = self.cls_head(projected)
        probs       = F.softmax(cls_logits, dim=-1)
        pred_labels = cls_logits.argmax(dim=-1)
        pred_confs  = probs.gather(
            1, pred_labels.unsqueeze(1)).squeeze(1)
        reasoning   = self.reasoning_decoder.generate(
            projected, bos_id, eos_id, tokenizer)
        return {
            "labels":      pred_labels.cpu().tolist(),
            "confidences": pred_confs.cpu().tolist(),
            "reasoning":   reasoning,
        }