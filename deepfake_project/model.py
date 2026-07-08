"""
model.py
========
DeepfakeReasoningModel — InternVL3-2B + LoRA + custom heads.

Fixes applied (vs previous version):
  #1  LoRA freeze order  — LoRA applied FIRST, then non-LoRA frozen.
  #2  LM label masking   — uses attention_mask, not token_id == 0.
  #3  Vision bottleneck  — CLS + mean-patch prefix (2 tokens) instead of 1.
  #6  ClassificationHead — simplified to 2-layer; input is CLS+patch concat.
  #8  DomainRouter       — removed from forward compute; kept for ckpt compat.
  #9  lm_loss_weight     — lowered to 0.3 so cls dominates early training.
  #10 CLS-only cls feat  — cls_head uses CLS + mean-patch (richer signal).
  #13 Grad-check assert  — raises at init if LoRA has zero trainable params.
"""

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


# ── Custom Projector ──────────────────────────────────────────────────────────

class CustomProjector(nn.Module):
    """
    Maps vision features → 2 LLM prefix tokens.

    Fix #3: previously projected only the CLS token → 1 prefix token.
    Now projects CLS and mean-patch separately → 2 prefix tokens.
    The LLM sees both a global summary and a spatial average of the image,
    which meaningfully improves reasoning quality without added VRAM cost.
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
        """
        Args:
            cls_token  : (B, vision_dim)
            patch_mean : (B, vision_dim)
        Returns:
            (B, 2, llm_dim) — two visual prefix tokens
        """
        t1 = self.cls_proj(cls_token).unsqueeze(1)     # (B, 1, llm_dim)
        t2 = self.patch_proj(patch_mean).unsqueeze(1)  # (B, 1, llm_dim)
        return torch.cat([t1, t2], dim=1)              # (B, 2, llm_dim)


# ── Classification Head ───────────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """
    Fix #6 + #10: simplified to 2 layers; input is CLS concat mean-patch.
    Previous: 3 layers, CLS token only.
    Now: vision_dim*2 → 256 → 2. Richer input, fewer dead params.
    """
    def __init__(self, vision_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vision_dim * 2, 256),
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


# ── Domain Router (checkpoint-compat stub, not called during training) ────────

class DomainRouter(nn.Module):
    """
    Fix #8: not called in forward() or predict().
    Kept as a registered module so old checkpoints load without missing-key errors.
    Will be wired in during Phase 2 (multi-domain MoE).
    """
    def __init__(self, vision_dim: int, num_domains: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vision_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Main Model ────────────────────────────────────────────────────────────────

class DeepfakeReasoningModel(nn.Module):
    def __init__(
        self,
        model_path:     str   = "./models/InternVL3-2B",
        lora_rank:      int   = 64,
        lora_alpha:     int   = 128,
        lora_dropout:   float = 0.05,
        cls_dropout:    float = 0.3,
        lm_loss_weight: float = 0.3,   # fix #9: was 0.5, cls must dominate early
    ):
        super().__init__()
        self.lm_loss_weight = lm_loss_weight

        # ── Step 1: load backbone ────────────────────────────────────
        print(f"Loading InternVL3-2B from {model_path} ...")
        self.backbone = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        # ── Step 2: apply LoRA BEFORE any freezing (fix #1) ─────────
        #
        # WHY ORDER MATTERS:
        # get_peft_model() wraps target Linear layers with LoraLayer,
        # injecting new nn.Parameter tensors (lora_A, lora_B).
        # These new tensors are created with requires_grad=True by default.
        #
        # If you freeze first (requires_grad=False on all params) and THEN
        # call get_peft_model(), the injected lora_A/lora_B tensors are fine —
        # BUT in practice PEFT may also call .requires_grad_(False) on the
        # wrapped base weight as part of its internal bookkeeping, and some
        # PEFT versions propagate the frozen state to child modules.
        # The safe, unambiguous order is always: LoRA first, freeze second.
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        self.backbone.language_model = get_peft_model(
            self.backbone.language_model, lora_cfg
        )

        # ── Step 3: freeze everything that is NOT a LoRA param ───────
        for name, param in self.backbone.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

        # Vision encoder is always fully frozen
        for param in self.backbone.vision_model.parameters():
            param.requires_grad = False

        # ── Step 4: get dims ─────────────────────────────────────────
        cfg        = self.backbone.config
        vision_dim = cfg.vision_config.hidden_size   # 1024 for InternVL3-2B
        llm_dim    = cfg.llm_config.hidden_size      # 2048 for InternVL3-2B
        self.vision_dim = vision_dim

        # ── Step 5: custom heads ─────────────────────────────────────
        self.custom_projector = CustomProjector(vision_dim, llm_dim)
        self.cls_head         = ClassificationHead(vision_dim, cls_dropout)
        self.domain_router    = DomainRouter(vision_dim, num_domains=4)

        # ── Step 6: assert LoRA is live (fix #13) ────────────────────
        self._assert_lora_trainable()
        self._print_params()

    # ── Sanity checks ─────────────────────────────────────────────────────────

    def _assert_lora_trainable(self):
        """
        Fix #13: hard assertion that LoRA params are trainable.
        Raises RuntimeError at model construction time — not after 6 hours
        of training with silent zero gradients.
        """
        lora_params = [
            (n, p) for n, p in self.named_parameters()
            if "lora_" in n and p.requires_grad
        ]
        if not lora_params:
            raise RuntimeError(
                "FATAL: No LoRA parameters are trainable after model build. "
                "This usually means freeze was applied before LoRA injection. "
                "Check LoraConfig target_modules match the actual layer names."
            )
        n_params = sum(p.numel() for _, p in lora_params)
        print(f"  LoRA trainable: {n_params:,} params across "
              f"{len(lora_params)} tensors  ✓")

    def _print_params(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total params  : {total:,}")
        print(f"  Trainable     : {trainable:,}  ({100*trainable/total:.2f}%)")

    # ── Vision feature extraction ─────────────────────────────────────────────

    def _get_vision_features(self, pixel_values):
        """
        Returns CLS token and mean of patch tokens from InternViT.

        Fix #3 / #10: CLS alone loses all spatial/structural information.
        Patch mean retains the spatial average. Both are cheap to compute.

        Returns:
            cls_token  : (B, vision_dim)
            patch_mean : (B, vision_dim)
        """
        vision_out = self.backbone.vision_model(
            pixel_values=pixel_values.to(torch.bfloat16)
        )
        hs         = vision_out.last_hidden_state   # (B, N+1, D)
        cls_token  = hs[:, 0, :]                   # (B, D)
        patch_mean = hs[:, 1:, :].mean(dim=1)      # (B, D)
        return cls_token, patch_mean

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        pixel_values,
        labels=None,
        reasoning_tokens=None,
        reasoning_attention_mask=None,
    ):
        """
        Args:
            pixel_values             : (B, 3, 448, 448)
            labels                   : (B,)
            reasoning_tokens         : (B, lm_seq_len)  — already on GPU
            reasoning_attention_mask : (B, lm_seq_len)

        Returns:
            dict with cls_logits, loss, cls_loss, lm_loss
        """
        cls_token, patch_mean = self._get_vision_features(pixel_values)

        # Concat for classifier (fix #10)
        vis_for_cls = torch.cat(
            [cls_token.float(), patch_mean.float()], dim=-1
        )                                               # (B, vision_dim*2)
        cls_logits  = self.cls_head(vis_for_cls)        # (B, 2)

        loss     = None
        lm_loss  = None
        cls_loss = None

        if labels is not None:
            cls_loss = F.cross_entropy(
                cls_logits, labels, label_smoothing=0.1
            )
            loss = cls_loss

        if reasoning_tokens is not None and labels is not None:
            # 2-token visual prefix — cast to float32 to match projector weights
            vis_prefix   = self.custom_projector(cls_token.float(), patch_mean.float())
            token_embeds = self.backbone.language_model.get_input_embeddings()(
                reasoning_tokens
            )
            inputs_embeds = torch.cat([vis_prefix, token_embeds], dim=1)

            # Fix #2: mask via attention_mask, not token_id == 0
            lm_labels = reasoning_tokens.clone()
            lm_labels[reasoning_attention_mask == 0] = -100

            # 2 ignore positions for the 2 visual prefix tokens
            prefix_ignore = torch.full(
                (lm_labels.size(0), 2), -100,
                dtype=torch.long, device=lm_labels.device,
            )
            lm_labels = torch.cat([prefix_ignore, lm_labels], dim=1)

            prefix_mask = torch.ones(
                reasoning_attention_mask.size(0), 2,
                dtype=reasoning_attention_mask.dtype,
                device=reasoning_attention_mask.device,
            )
            full_mask = torch.cat([prefix_mask, reasoning_attention_mask], dim=1)

            lm_out  = self.backbone.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=full_mask,
                labels=lm_labels,
            )
            lm_loss = lm_out.loss
            loss    = cls_loss + self.lm_loss_weight * lm_loss

        return {
            "cls_logits": cls_logits,
            "loss":       loss,
            "cls_loss":   cls_loss,
            "lm_loss":    lm_loss,
        }

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, pixel_values, tokenizer, max_new_tokens=512):
        self.eval()
        cls_token, patch_mean = self._get_vision_features(pixel_values)

        vis_for_cls = torch.cat(
            [cls_token.float(), patch_mean.float()], dim=-1
        )
        cls_logits = self.cls_head(vis_for_cls)
        probs      = F.softmax(cls_logits, dim=-1)
        pred       = cls_logits.argmax(-1)
        conf       = probs.gather(1, pred.unsqueeze(1)).squeeze(1)

        B, results = pixel_values.size(0), []
        for i in range(B):
            vis_prefix = self.custom_projector(
                cls_token[i:i+1].float(), patch_mean[i:i+1].float()
            )                                              # (1, 2, llm_dim)
            # No primer — feed visual prefix only.
            # Model learned continuation from visual tokens; a primer
            # token constrains it to one tag and it stops early.
            inputs_embeds = vis_prefix.to(torch.bfloat16)

            attention_mask = torch.ones(
                inputs_embeds.shape[:2],
                dtype=torch.long,
                device=inputs_embeds.device,
            )

            out = self.backbone.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.3,
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
