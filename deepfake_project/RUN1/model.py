"""
model.py
========
HydraFakeNet: dual-head model for deepfake detection.
  - Classification head  → binary (real / fake)
  - Reasoning head       → autoregressive text generation (GRU-based decoder)

The visual backbone is the from-scratch ViTEncoder defined in transformer_blocks.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_blocks import ViTEncoder


# ---------------------------------------------------------------------------
# Classification Head
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    """
    MLP head that maps the CLS token to a 2-class logit.

    Structure: LayerNorm → Linear → GELU → Dropout → Linear
    """

    def __init__(self, embed_dim: int, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        return self.net(cls_token)  # (B, num_classes)


# ---------------------------------------------------------------------------
# Reasoning Decoder (GRU-based)
# ---------------------------------------------------------------------------

class ReasoningDecoder(nn.Module):
    """
    Lightweight GRU decoder that conditions on the visual CLS token
    to generate a reasoning token sequence.

    At training time  : teacher-forced cross-entropy over the full sequence.
    At inference time : greedy decoding token by token.

    Args:
        vocab_size  : int – tokenizer vocabulary size.
        embed_dim   : int – visual feature dimension (from ViT).
        hidden_dim  : int – GRU hidden dimension.
        num_layers  : int – GRU depth.
        max_length  : int – maximum generated sequence length.
        dropout     : float.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 2,
        max_length: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_length = max_length

        # Project visual CLS token to GRU initial hidden state
        self.visual_proj = nn.Linear(embed_dim, hidden_dim * num_layers)

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)

        # GRU decoder
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output projection to vocabulary
        self.out_proj = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.token_embed.weight, std=0.02)
        nn.init.zeros_(self.token_embed.weight[0])  # padding idx

    def _build_initial_hidden(self, cls_token: torch.Tensor) -> torch.Tensor:
        """
        Map CLS token → GRU initial hidden state.
        Returns: (num_layers, B, hidden_dim)
        """
        B = cls_token.size(0)
        h0 = self.visual_proj(cls_token)                     # (B, hidden*layers)
        h0 = h0.view(B, self.num_layers, self.hidden_dim)   # (B, layers, hidden)
        h0 = h0.permute(1, 0, 2).contiguous()               # (layers, B, hidden)
        return torch.tanh(h0)

    def forward(
        self,
        cls_token: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Teacher-forced forward pass.

        Args:
            cls_token  : (B, embed_dim)
            target_ids : (B, seq_len)  – including BOS token at position 0.

        Returns:
            logits     : (B, seq_len, vocab_size)
        """
        h = self._build_initial_hidden(cls_token)           # (layers, B, H)
        emb = self.dropout(self.token_embed(target_ids))    # (B, T, H)
        out, _ = self.gru(emb, h)                           # (B, T, H)
        logits = self.out_proj(out)                         # (B, T, vocab)
        return logits

    @torch.no_grad()
    def generate(
        self,
        cls_token: torch.Tensor,
        bos_id: int,
        eos_id: int,
        tokenizer=None,
    ) -> list[str]:
        """
        Greedy decoding; returns a list of generated strings (one per batch item).
        """
        B = cls_token.size(0)
        h = self._build_initial_hidden(cls_token)

        # Start with BOS token
        current = torch.full((B, 1), bos_id, dtype=torch.long, device=cls_token.device)
        generated = [[] for _ in range(B)]
        finished = [False] * B

        for _ in range(self.max_length):
            emb = self.token_embed(current)       # (B, 1, H)
            out, h = self.gru(emb, h)             # (B, 1, H)
            logits = self.out_proj(out[:, 0, :])  # (B, vocab)
            next_tokens = logits.argmax(dim=-1)   # (B,)

            for b in range(B):
                tok = next_tokens[b].item()
                if not finished[b]:
                    if tok == eos_id:
                        finished[b] = True
                    else:
                        generated[b].append(tok)

            if all(finished):
                break
            current = next_tokens.unsqueeze(1)   # (B, 1)

        if tokenizer is not None:
            return [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated]
        return generated


# ---------------------------------------------------------------------------
# HydraFakeNet  – the combined model
# ---------------------------------------------------------------------------

class HydraFakeNet(nn.Module):
    """
    Complete deepfake detection network.

    Args
    ----
    image_size    : side length of input image (default 224).
    patch_size    : ViT patch size (default 16).
    embed_dim     : visual feature dimension.
    depth         : number of ViT encoder layers.
    num_heads     : number of attention heads.
    mlp_ratio     : FFN expansion factor.
    dropout       : general dropout.
    attn_drop     : attention dropout.
    vocab_size    : tokenizer vocabulary size.
    reasoning_hidden_dim : GRU hidden size.
    reasoning_layers     : GRU depth.
    max_reasoning_len    : max tokens to generate.
    cls_dropout   : dropout inside classification head.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_drop: float = 0.0,
        vocab_size: int = 32000,
        reasoning_hidden_dim: int = 512,
        reasoning_layers: int = 2,
        max_reasoning_len: int = 128,
        cls_dropout: float = 0.3,
    ):
        super().__init__()

        # ── Visual Backbone ──────────────────────────────────────────────
        self.encoder = ViTEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attn_drop=attn_drop,
        )

        # ── Classification Head ──────────────────────────────────────────
        self.cls_head = ClassificationHead(embed_dim, num_classes=2, dropout=cls_dropout)

        # ── Reasoning Decoder ────────────────────────────────────────────
        self.reasoning_decoder = ReasoningDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=reasoning_hidden_dim,
            num_layers=reasoning_layers,
            max_length=max_reasoning_len,
            dropout=dropout,
        )

        self.embed_dim = embed_dim

    # ------------------------------------------------------------------
    def forward(
        self,
        images: torch.Tensor,
        reasoning_ids: torch.Tensor | None = None,
        return_attn: bool = False,
    ):
        """
        Args:
            images        : (B, 3, H, W)
            reasoning_ids : (B, T)  optional – teacher-forced token ids
            return_attn   : whether to return attention maps

        Returns (dict):
            cls_logits    : (B, 2)
            reasoning_logits : (B, T, vocab) – only if reasoning_ids provided
            attn_maps     : list of tensors  – only if return_attn=True
        """
        if return_attn:
            cls_token, all_tokens, attn_maps = self.encoder(images, return_attn=True)
        else:
            cls_token, all_tokens = self.encoder(images)
            attn_maps = None

        cls_logits = self.cls_head(cls_token)

        out = {"cls_logits": cls_logits}

        if reasoning_ids is not None:
            reasoning_logits = self.reasoning_decoder(cls_token, reasoning_ids)
            out["reasoning_logits"] = reasoning_logits

        if attn_maps is not None:
            out["attn_maps"] = attn_maps

        return out

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        bos_id: int,
        eos_id: int,
        tokenizer=None,
    ) -> dict:
        """
        Inference helper.
        Returns predicted label (0/1), confidence, and generated reasoning text.
        """
        self.eval()
        cls_token, _ = self.encoder(images)
        cls_logits = self.cls_head(cls_token)
        probs = F.softmax(cls_logits, dim=-1)
        pred_labels = cls_logits.argmax(dim=-1)
        pred_confs = probs.gather(1, pred_labels.unsqueeze(1)).squeeze(1)

        reasoning_texts = self.reasoning_decoder.generate(
            cls_token, bos_id, eos_id, tokenizer
        )

        return {
            "labels": pred_labels.cpu().tolist(),
            "confidences": pred_confs.cpu().tolist(),
            "reasoning": reasoning_texts,
        }
