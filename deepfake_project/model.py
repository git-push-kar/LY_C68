"""
model.py
========
DeepfakeReasoningModel v2 — InternVL3-2B (InternViT-300M + Qwen2.5-1.5B) + LoRA + v2 Architecture.

Architectural Upgrades in v2:
  #1  Vision-encoder LoRA   — Second LoRA adapter applied to InternViT attention (qkv/proj)
                              and MLP (fc1/fc2) layers before freezing non-LoRA params.
  #2  Attention Pooling     — Learned query vector cross-attends over all patch tokens (hs[:, 1:, :]),
                              replacing mean-pooling to preserve localized forgery artifacts.
  #3  Frequency Branch      — 2D FFT log-magnitude convolutional branch (256-dim) fused with
                              ViT features into cls_head.
  #4  Native Visual Tokens  — LLM reasoning path uses InternVL3's native multi-token visual sequence
                              (e.g. 256 tokens) instead of the 2-token compression bottleneck.
  #5  Self-Consistency Loss — Bidirectional KL / consistency loss between cls_head decision and
                              LM logit distribution at the generated <answer> token position.
  #6  Loss Curriculum       — Linear ramp for lm_loss_weight across joint epochs (0.3 -> 0.6).
  #7  Focal Loss Option     — Hard-example focal loss support for classification imbalances.
  #8  Arch Version Guard    — Versioning flag ('v1' vs 'v2') with strict checkpoint loading guards.

IMPORTANT ARCHITECTURAL NOTE:
Adding LoRA to the vision encoder (ViT) does NOT eliminate the need for `cls_head`.
LoRA only transforms the ViT features to be richer and forgery-aware; a dedicated mapping layer
(`cls_head`) is still strictly necessary to project the combined ViT + frequency feature representation
(vision_dim*2 + freq_dim) into binary classification logits (real vs fake).
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from peft import LoraConfig, get_peft_model, TaskType


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an image authenticity expert. Your task is to determine "
    "the authenticity of the given facial image.\n\n"
    "Firstly, give an overall judgement enclosed in <fast> </fast> tags.\n"
    "Then make a careful structured analysis using <planning>, <reasoning>, "
    "<reflection>, <conclusion> tags as needed.\n"
    "Finally give your answer with \"real\" or \"fake\" enclosed in "
    "<answer> </answer> tags."
)
USER_PROMPT = "Please determine the authenticity of this image."


# ── Attention Pooling (v2: replaces mean-pooling) ──────────────────────────────

class AttentionPool(nn.Module):
    """
    Learned query vector cross-attends over all patch tokens (hs[:, 1:, :]).
    Replaces patch_tokens.mean(dim=1).

    Mean-pooling crushes fine-grained localized spatial artifacts (e.g. boundary seam
    inconsistencies, localized warping, face-swap blending edges).
    Attention pooling enables the model to dynamically focus on suspicious localized regions.
    """
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self._init()

    def _init(self):
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: (B, N, D) — all patch tokens from vision encoder
        Returns:
            (B, D) — single attention-pooled visual vector
        """
        B = patch_tokens.size(0)
        q = self.query.expand(B, -1, -1)  # (B, 1, D)
        # Query attends over patch key/values
        attn_out, _ = self.mha(query=q, key=patch_tokens, value=patch_tokens)
        return self.norm(attn_out.squeeze(1))  # (B, D)


# ── Frequency Branch (v2: FFT / spectral fusion) ──────────────────────────────

class FrequencyBranch(nn.Module):
    """
    Convolutional feature extractor operating on log-magnitude 2D FFT inputs.
    Outputs a fixed 256-dimensional frequency representation vector.

    Deepfake generative pipelines (GANs, Diffusion, FaceSwap) leave distinct frequency
    artifacts (spectral peaks, grid regularities, high-frequency dropoffs) that are
    starkly visible in the Fourier domain even when invisible in raw RGB space.
    """
    def __init__(self, in_channels: int = 3, out_dim: int = 256):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),   # 224 -> 112
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),            # 112 -> 56
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),           # 56 -> 28
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, out_dim, kernel_size=3, stride=2, padding=1),       # 28 -> 14
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W) — log-magnitude 2D FFT spectrum
        Returns:
            (B, out_dim) — 256-dim frequency embedding
        """
        return self.net(x.float())


# ── Classification Head ───────────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """
    Classification head mapping fused ViT + Frequency features to binary logits.

    IMPORTANT:
    Adding LoRA to ViT does NOT remove the need for cls_head.
    ViT LoRA and Attention Pooling enrich the representation; cls_head is the mandatory
    layer that translates that representation (vision_dim*2 + freq_dim) into class logits.
    """
    def __init__(self, in_features: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Focal Loss (v2: hard-example mining for forgery imbalances) ───────────────

class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Downweights easy negative/positive examples and focuses gradient on hard boundary cases.
    """
    def __init__(self, gamma: float = 2.0, alpha: Optional[Union[float, torch.Tensor]] = None, label_smoothing: float = 0.05):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none", label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
            else:
                alpha_t = self.alpha.to(targets.device)[targets]
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean()


# ── Custom Projector (retained for compact summary & ckpt compat) ────────────

class CustomProjector(nn.Module):
    """
    Maps vision features → 2 summary LLM prefix tokens.
    Retained for compact summary representation and backward compatibility.
    In v2, the primary reasoning path uses InternVL3's native multi-token visual representation.
    """
    def __init__(self, vision_dim: int, llm_dim: int, dropout: float = 0.1):
        super().__init__()
        self.cls_proj = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_dim, llm_dim),
            nn.LayerNorm(llm_dim),
        )
        self.patch_proj = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_dim, llm_dim),
            nn.LayerNorm(llm_dim),
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, cls_token: torch.Tensor, patch_mean: torch.Tensor):
        t1 = self.cls_proj(cls_token).unsqueeze(1)     # (B, 1, llm_dim)
        t2 = self.patch_proj(patch_mean).unsqueeze(1)  # (B, 1, llm_dim)
        return torch.cat([t1, t2], dim=1)              # (B, 2, llm_dim)


# ── Domain Router (checkpoint-compat stub) ────────────────────────────────────

class DomainRouter(nn.Module):
    def __init__(self, vision_dim: int, num_domains: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vision_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Main Model (v2) ──────────────────────────────────────────────────────────

class DeepfakeReasoningModel(nn.Module):
    """
    DeepfakeReasoningModel v2:
    - Backbone: InternVL3-2B (InternViT-300M + Qwen2.5-1.5B)
    - LLM LoRA: Rank 64, Alpha 128
    - Vision LoRA: Rank 16, Alpha 32 (targets InternViT attention + MLP)
    - Spatial Pooling: AttentionPool over full patch grid
    - Frequency Domain: 2D FFT log-magnitude CNN branch (256-dim)
    - Reasoning Grounding: Full multi-token visual representation via native InternVL3 pipeline
    - Joint Loss: Classification Loss + Scheduled LM Loss + Self-Consistency Loss
    """
    def __init__(
        self,
        model_path:              str   = "./models/InternVL3-2B",
        lora_rank:               int   = 64,
        lora_alpha:              int   = 128,
        lora_dropout:            float = 0.05,
        vision_lora_rank:        int   = 16,
        vision_lora_alpha:       int   = 32,
        vision_lora_dropout:     float = 0.05,
        cls_dropout:             float = 0.3,
        lm_loss_weight:          float = 0.3,
        consistency_loss_weight: float = 0.05,
        use_focal_loss:          bool  = False,
        focal_gamma:             float = 2.0,
        arch_version:            str   = "v2",
        real_token_id:           Optional[int] = None,
        fake_token_id:           Optional[int] = None,
    ):
        super().__init__()
        self.arch_version = arch_version
        self.lora_rank = lora_rank
        self.vision_lora_rank = vision_lora_rank
        self.lm_loss_weight = lm_loss_weight
        self.consistency_loss_weight = consistency_loss_weight
        self.use_focal_loss = use_focal_loss
        self.real_token_id = real_token_id
        self.fake_token_id = fake_token_id

        # ── Step 1: Load backbone ──────────────────────────────────────────
        print(f"Loading InternVL3-2B from {model_path} (arch_version={arch_version}) ...")
        self.backbone = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        # ── Step 2: Apply LLM LoRA BEFORE freezing (Fix #1) ────────────────
        llm_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"]
        llm_lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=llm_target_modules,
            bias="none",
        )
        self.backbone.language_model = get_peft_model(
            self.backbone.language_model, llm_lora_cfg
        )

        # ── Step 3: Apply Vision LoRA BEFORE freezing (2.1) ────────────────
        if vision_lora_rank > 0:
            # Inspect actual module names dynamically to avoid blind hardcoding
            vision_module_names = [n for n, _ in self.backbone.vision_model.named_modules()]
            vision_target_candidates = ["qkv", "proj", "fc1", "fc2", "q_proj", "k_proj", "v_proj", "out_proj", "w1", "w2"]
            vision_targets = set()
            for cand in vision_target_candidates:
                if any(n.endswith(f".{cand}") or n == cand for n in vision_module_names):
                    vision_targets.add(cand)
            if not vision_targets:
                # Fallback to standard InternViT module names
                vision_targets = {"qkv", "proj", "fc1", "fc2"}

            print(f"  Injecting Vision LoRA (r={vision_lora_rank}, a={vision_lora_alpha}) into: {sorted(vision_targets)}")
            vision_lora_cfg = LoraConfig(
                r=vision_lora_rank,
                lora_alpha=vision_lora_alpha,
                lora_dropout=vision_lora_dropout,
                target_modules=list(vision_targets),
                bias="none",
            )
            self.backbone.vision_model = get_peft_model(
                self.backbone.vision_model, vision_lora_cfg
            )

        # ── Step 4: Freeze everything that is NOT a LoRA param ─────────────
        for name, param in self.backbone.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

        if vision_lora_rank == 0:
            for param in self.backbone.vision_model.parameters():
                param.requires_grad = False

        # ── Step 5: Dimensions & Custom Heads ──────────────────────────────
        cfg = self.backbone.config
        vision_dim = getattr(cfg.vision_config, "hidden_size", 1024)
        llm_dim    = getattr(cfg.llm_config, "hidden_size", 2048)
        self.vision_dim = vision_dim
        self.llm_dim = llm_dim
        self.freq_dim = 256

        # v2 Attention pooling over patch tokens
        self.attn_pool = AttentionPool(vision_dim)

        # v2 Frequency-domain branch (FFT)
        self.freq_branch = FrequencyBranch(in_channels=3, out_dim=self.freq_dim)

        # v2 Classification Head: [CLS, Attention-Pooled Patches, Frequency Embedding]
        # in_features = vision_dim * 2 + 256
        self.cls_head = ClassificationHead(vision_dim * 2 + self.freq_dim, cls_dropout)

        # Custom projector & Domain router (retained for compatibility)
        self.custom_projector = CustomProjector(vision_dim, llm_dim)
        self.domain_router    = DomainRouter(vision_dim, num_domains=4)

        # Loss functions
        if use_focal_loss:
            self.cls_criterion = FocalLoss(gamma=focal_gamma, label_smoothing=0.05)
        else:
            self.cls_criterion = None  # standard cross_entropy with label_smoothing=0.1

        # ── Step 6: Assertions & Parameter breakdown ───────────────────────
        self._assert_lora_trainable()
        self._print_params()

    # ── Sanity checks ─────────────────────────────────────────────────────────

    def _assert_lora_trainable(self):
        """
        Hard assertion that LoRA params are active and trainable.
        Raises RuntimeError immediately if LoRA injection failed or was frozen out.
        """
        llm_lora = [
            (n, p) for n, p in self.named_parameters()
            if "language_model" in n and "lora_" in n and p.requires_grad
        ]
        if self.lora_rank > 0 and not llm_lora:
            raise RuntimeError(
                "FATAL: No LLM LoRA parameters are trainable after model build. "
                "Check that freeze was applied AFTER LoRA injection."
            )

        if self.vision_lora_rank > 0:
            vis_lora = [
                (n, p) for n, p in self.named_parameters()
                if "vision_model" in n and "lora_" in n and p.requires_grad
            ]
            if not vis_lora:
                raise RuntimeError(
                    "FATAL: No Vision LoRA parameters are trainable after model build. "
                    "Check that vision_model LoraConfig target_modules match InternViT layers."
                )

    def _print_params(self):
        """Break down parameter counts by module category."""
        counts = {
            "llm_lora": 0,
            "vision_lora": 0,
            "cls_head": 0,
            "freq_branch": 0,
            "attn_pool": 0,
            "custom_projector": 0,
            "domain_router": 0,
            "other_trainable": 0,
        }
        total = sum(p.numel() for p in self.parameters())
        for n, p in self.named_parameters():
            if p.requires_grad:
                if "language_model" in n and "lora_" in n:
                    counts["llm_lora"] += p.numel()
                elif "vision_model" in n and "lora_" in n:
                    counts["vision_lora"] += p.numel()
                elif "cls_head" in n:
                    counts["cls_head"] += p.numel()
                elif "freq_branch" in n:
                    counts["freq_branch"] += p.numel()
                elif "attn_pool" in n:
                    counts["attn_pool"] += p.numel()
                elif "custom_projector" in n:
                    counts["custom_projector"] += p.numel()
                elif "domain_router" in n:
                    counts["domain_router"] += p.numel()
                else:
                    counts["other_trainable"] += p.numel()

        trainable = sum(counts.values())
        print("  " + "=" * 58)
        print("  DeepfakeReasoningModel v2 Parameter Breakdown:")
        print(f"    Total params     : {total:,}")
        print(f"    Trainable params : {trainable:,} ({100 * trainable / max(1, total):.2f}%)")
        for k, v in counts.items():
            if v > 0 or k in ["llm_lora", "vision_lora", "cls_head", "freq_branch", "attn_pool"]:
                print(f"      - {k:<18}: {v:>10,} params")
        print("  " + "=" * 58)

    # ── Feature Extraction ────────────────────────────────────────────────────

    def _pool_patches(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Attention-pool patch tokens using learned query."""
        return self.attn_pool(patch_tokens.float())

    def _get_vision_features(self, pixel_values: torch.Tensor):
        """
        Extracts CLS token, attention-pooled patch vector, and full patch tokens.
        
        Returns:
            cls_token     : (B, vision_dim)
            pooled_patches: (B, vision_dim) — attention-pooled over all N patches
            patch_tokens  : (B, N, vision_dim) — full patch token grid
        """
        vision_out = self.backbone.vision_model(
            pixel_values=pixel_values.to(torch.bfloat16)
        )
        hs = vision_out.last_hidden_state           # (B, N+1, D)
        cls_token = hs[:, 0, :]                    # (B, D)
        patch_tokens = hs[:, 1:, :]                # (B, N, D)
        pooled_patches = self._pool_patches(patch_tokens)  # (B, D)
        return cls_token, pooled_patches, patch_tokens

    def _get_native_visual_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Route pixel_values through InternVL3's native visual token pipeline
        (pixel-shuffle + mlp1 projector) to obtain the full multi-token visual representation.
        
        This eliminates the 2-token compression bottleneck and feeds dozens/hundreds of
        spatially grounded visual tokens to the reasoning LLM.
        """
        if hasattr(self.backbone, "extract_feature"):
            return self.backbone.extract_feature(pixel_values.to(torch.bfloat16))
        elif hasattr(self.backbone, "mlp1"):
            vision_out = self.backbone.vision_model(pixel_values=pixel_values.to(torch.bfloat16))
            hs = vision_out.last_hidden_state[:, 1:, :]
            if hasattr(self.backbone, "pixel_shuffle") and hasattr(self.backbone, "downsample_ratio"):
                hs = self.backbone.pixel_shuffle(hs, scale_factor=self.backbone.downsample_ratio)
            elif hasattr(self.backbone, "pixel_shuffle"):
                hs = self.backbone.pixel_shuffle(hs, scale_factor=0.5)
            return self.backbone.mlp1(hs)
        else:
            # Fallback for mock backbones
            vision_out = self.backbone.vision_model(pixel_values=pixel_values.to(torch.bfloat16))
            hs = vision_out.last_hidden_state
            cls_token = hs[:, 0, :]
            patch_mean = hs[:, 1:, :].mean(dim=1)
            return self.custom_projector(cls_token.float(), patch_mean.float())

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        reasoning_tokens: Optional[torch.Tensor] = None,
        reasoning_attention_mask: Optional[torch.Tensor] = None,
        freq_features: Optional[torch.Tensor] = None,
        answer_token_pos: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            pixel_values             : (B, 3, 448, 448)
            labels                   : (B,) binary labels (0=real, 1=fake)
            reasoning_tokens         : (B, lm_seq_len) pre-tokenized token IDs
            reasoning_attention_mask : (B, lm_seq_len) attention mask
            freq_features            : (B, 3, 224, 224) 2D FFT log-magnitude tensor
            answer_token_pos         : (B,) index of 'real'/'fake' word inside reasoning_tokens

        Returns:
            dict containing:
                cls_logits       : (B, 2)
                loss             : total scalar loss
                cls_loss         : classification loss
                lm_loss          : language modeling loss
                consistency_loss : self-consistency loss (or None)
        """
        B = pixel_values.size(0)
        device = pixel_values.device

        # ── 1. Vision features & Frequency fusion ──────────────────────
        cls_token, pooled_patches, patch_tokens = self._get_vision_features(pixel_values)

        if freq_features is not None:
            freq_emb = self.freq_branch(freq_features.to(device))  # (B, 256)
        else:
            # Zero-tensor fallback for validation / inference callers without FFT
            freq_emb = torch.zeros((B, self.freq_dim), dtype=torch.float32, device=device)

        # Concatenate CLS + Attention-Pooled Patches + Frequency Embedding
        vis_for_cls = torch.cat(
            [cls_token.float(), pooled_patches.float(), freq_emb.float()], dim=-1
        )  # (B, vision_dim*2 + 256)

        cls_logits = self.cls_head(vis_for_cls)  # (B, 2)

        loss = None
        cls_loss = None
        lm_loss = None
        consistency_loss = None

        # ── 2. Classification Loss ─────────────────────────────────────
        if labels is not None:
            if self.cls_criterion is not None:
                cls_loss = self.cls_criterion(cls_logits, labels)
            else:
                cls_loss = F.cross_entropy(cls_logits, labels, label_smoothing=0.1)
            loss = cls_loss

        # ── 3. Reasoning Path (Native Visual Tokens) ───────────────────
        if reasoning_tokens is not None and labels is not None:
            # Native multi-token visual representation
            vis_tokens = self._get_native_visual_tokens(pixel_values)  # (B, N_vis, llm_dim)
            N_vis = vis_tokens.size(1)

            token_embeds = self.backbone.language_model.get_input_embeddings()(reasoning_tokens)
            inputs_embeds = torch.cat([vis_tokens.to(token_embeds.dtype), token_embeds], dim=1)

            # Mask visual prefix tokens from LM loss
            lm_labels = reasoning_tokens.clone()
            lm_labels[reasoning_attention_mask == 0] = -100
            prefix_ignore = torch.full((B, N_vis), -100, dtype=torch.long, device=device)
            full_lm_labels = torch.cat([prefix_ignore, lm_labels], dim=1)

            prefix_mask = torch.ones((B, N_vis), dtype=reasoning_attention_mask.dtype, device=device)
            full_mask = torch.cat([prefix_mask, reasoning_attention_mask], dim=1)

            lm_out = self.backbone.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=full_mask,
                labels=full_lm_labels,
            )
            lm_loss = lm_out.loss

            # ── 4. Self-Consistency Loss (2.5) ─────────────────────────
            # Aligns classifier decision with LLM answer token distribution
            if answer_token_pos is not None and self.consistency_loss_weight > 0.0:
                valid_mask = (answer_token_pos >= 0) & (answer_token_pos < reasoning_tokens.size(1))
                if valid_mask.any():
                    valid_idx = torch.where(valid_mask)[0]
                    # In causal LM, token at text pos `ans_pos` is predicted at index `N_vis + ans_pos - 1`
                    pred_pos = N_vis + answer_token_pos[valid_idx] - 1
                    ans_logits = lm_out.logits[valid_idx, pred_pos, :]  # (valid_B, vocab_size)

                    if self.real_token_id is not None and self.fake_token_id is not None:
                        lm_binary_logits = torch.stack([
                            ans_logits[:, self.real_token_id],
                            ans_logits[:, self.fake_token_id]
                        ], dim=-1)  # (valid_B, 2)
                        cls_valid_logits = cls_logits[valid_idx]

                        p_cls = F.softmax(cls_valid_logits.float(), dim=-1)
                        p_lm = F.softmax(lm_binary_logits.float(), dim=-1)

                        # Bidirectional KL divergence between classifier & LLM answer distribution
                        kl_1 = F.kl_div(F.log_softmax(lm_binary_logits.float(), dim=-1), p_cls.detach(), reduction="batchmean")
                        kl_2 = F.kl_div(F.log_softmax(cls_valid_logits.float(), dim=-1), p_lm.detach(), reduction="batchmean")
                        consistency_loss = 0.5 * (kl_1 + kl_2)

            # Combined joint loss
            loss = cls_loss + self.lm_loss_weight * lm_loss
            if consistency_loss is not None:
                loss = loss + self.consistency_loss_weight * consistency_loss

        return {
            "cls_logits":       cls_logits,
            "loss":             loss,
            "cls_loss":         cls_loss,
            "lm_loss":          lm_loss,
            "consistency_loss": consistency_loss,
        }

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        pixel_values: torch.Tensor,
        tokenizer,
        freq_features: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
    ) -> Dict[str, list]:
        """
        Deterministic (greedy) inference with native visual token sequence.
        """
        self.eval()
        device = pixel_values.device
        B = pixel_values.size(0)

        cls_token, pooled_patches, _ = self._get_vision_features(pixel_values)

        if freq_features is not None:
            freq_emb = self.freq_branch(freq_features.to(device))
        else:
            freq_emb = torch.zeros((B, self.freq_dim), dtype=torch.float32, device=device)

        vis_for_cls = torch.cat(
            [cls_token.float(), pooled_patches.float(), freq_emb.float()], dim=-1
        )
        cls_logits = self.cls_head(vis_for_cls)
        probs = F.softmax(cls_logits.float(), dim=-1)
        pred = cls_logits.argmax(-1)
        conf = probs.gather(1, pred.unsqueeze(1)).squeeze(1)

        results = []
        for i in range(B):
            vis_tokens = self._get_native_visual_tokens(pixel_values[i:i+1])
            inputs_embeds = vis_tokens.to(torch.bfloat16)

            attention_mask = torch.ones(
                inputs_embeds.shape[:2],
                dtype=torch.long,
                device=device,
            )

            out = self.backbone.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            results.append(text)

        return {
            "labels":      pred.cpu().tolist(),
            "confidences": conf.cpu().tolist(),
            "reasoning":   results,
        }
