"""
test_images.py
==============
Test best_model.pth from exp7 on 1.jpg, 2.jpg, 3.jpg, 4.jpg.

The GRU ReasoningDecoder was intentionally NOT trained (see model.py comments),
so we bypass it entirely and generate meaningful rule-based reasoning from:
  - Classification confidence
  - CLIP branch feature norms  (semantic/texture signal)
  - FFT branch feature norms   (spectral/frequency signal)
  - Class logit gap             (how decisive the model was)

Run from the datasets folder:
    python test_images.py
"""

import os
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

from model import HydraFakeNet

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = r"C:\Users\Admin\Desktop\ly project c 68\deepfake_project\datasets"
CHECKPOINT  = os.path.join(BASE_DIR, "runs", "exp7", "checkpoints", "best_model.pth")
IMAGE_FILES = ["1.jpg", "2.jpg", "3.jpg", "4.jpg"]

LABEL = {0: "REAL", 1: "FAKE"}

# ── Device ─────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}\n")

# ── Load checkpoint ────────────────────────────────────────────────────────
print(f"Loading checkpoint: {CHECKPOINT}")
ckpt = torch.load(CHECKPOINT, map_location=device)
args = ckpt.get("args", None)

if args is not None:
    clip_model       = getattr(args, "clip_model",       "openai/clip-vit-base-patch32")
    freeze_mode      = getattr(args, "freeze_mode",      "partial")
    unfreeze_last_n  = getattr(args, "unfreeze_last_n",  4)
    cls_depth        = getattr(args, "cls_depth",        2)
    cls_heads        = getattr(args, "cls_heads",        8)
    cls_queries      = getattr(args, "cls_queries",      4)
    reasoning_hidden = getattr(args, "reasoning_hidden", 512)
    reasoning_layers = getattr(args, "reasoning_layers", 2)
    max_text_len     = getattr(args, "max_text_len",     128)
    dropout          = getattr(args, "dropout",          0.4)
    drop_path        = getattr(args, "drop_path",        0.1)
    vocab_size       = getattr(args, "vocab_size",       30522)
else:
    print("No args in checkpoint — using default config.")
    clip_model       = "openai/clip-vit-base-patch32"
    freeze_mode      = "partial"
    unfreeze_last_n  = 4
    cls_depth        = 2
    cls_heads        = 8
    cls_queries      = 4
    reasoning_hidden = 512
    reasoning_layers = 2
    max_text_len     = 128
    dropout          = 0.4
    drop_path        = 0.1
    vocab_size       = 30522

# ── Build model ────────────────────────────────────────────────────────────
print("Building model...")
model = HydraFakeNet(
    clip_model_name   = clip_model,
    freeze_mode       = freeze_mode,
    unfreeze_last_n   = unfreeze_last_n,
    model_dim         = 512,           # always 512 = CLIP_PROJ(256) + FFT_OUT(256)
    cls_depth         = cls_depth,
    cls_heads         = cls_heads,
    cls_queries       = cls_queries,
    vocab_size        = vocab_size,
    reasoning_hidden  = reasoning_hidden,
    reasoning_layers  = reasoning_layers,
    max_reasoning_len = max_text_len,
    dropout           = dropout,
    drop_path         = drop_path,
).to(device)

state_key = "model_state_dict" if "model_state_dict" in ckpt else "model"
missing, unexpected = model.load_state_dict(ckpt[state_key], strict=False)
if missing:
    print(f"  Missing keys   : {len(missing)}")
if unexpected:
    print(f"  Unexpected keys: {len(unexpected)}")
model.eval()
print(f"Checkpoint loaded  (epoch {ckpt.get('epoch', '?')})\n")

# ── Transform (CLIP normalization — identical to dataset.py val/test) ──────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275,  0.40821073],
        std =[0.26862954, 0.26130258, 0.27577711],
    ),
])


# ── Rule-based reasoning (replaces the untrained GRU decoder) ─────────────

def generate_reasoning(label: int, confidence: float,
                        clip_feat_norm: float, fft_feat_norm: float,
                        logit_gap: float) -> str:
    """
    Produce a human-readable explanation derived from the model's
    internal signals. No GRU needed — the GRU was never trained.

    Parameters
    ----------
    label          : 0 = real, 1 = fake
    confidence     : softmax probability of predicted class  (0–1)
    clip_feat_norm : L2 norm of CLIP branch 256-dim feature
    fft_feat_norm  : L2 norm of FFT branch 256-dim feature
    logit_gap      : |logit_fake - logit_real|  (decisiveness)
    """

    # ── Confidence tier ───────────────────────────────────────────────
    if confidence >= 0.90:
        conf_desc = "very high confidence"
    elif confidence >= 0.75:
        conf_desc = "high confidence"
    elif confidence >= 0.60:
        conf_desc = "moderate confidence"
    else:
        conf_desc = "low confidence — result is uncertain"

    # ── Spectral (FFT) signal ─────────────────────────────────────────
    # High FFT norm → strong spectral fingerprint (GAN/diffusion artefacts)
    if fft_feat_norm > 14.0:
        fft_desc = (
            "strong spectral anomalies in the frequency domain "
            "(likely GAN upsampling grid artefacts or diffusion noise residuals)"
        )
    elif fft_feat_norm > 9.0:
        fft_desc = (
            "moderate spectral irregularities; "
            "frequency distribution is atypical for a real camera image"
        )
    else:
        fft_desc = (
            "frequency spectrum consistent with natural camera noise "
            "— no clear GAN or diffusion fingerprint detected"
        )

    # ── Semantic/texture (CLIP) signal ────────────────────────────────
    if clip_feat_norm > 14.0:
        clip_desc = (
            "CLIP semantic features are highly activating, "
            "suggesting over-smooth textures characteristic of generative models"
        )
    elif clip_feat_norm > 9.0:
        clip_desc = (
            "CLIP semantic features show moderate activation "
            "with some texture patterns that warrant scrutiny"
        )
    else:
        clip_desc = (
            "CLIP semantic features exhibit normal activation "
            "consistent with authentic photographic content"
        )

    # ── Decisiveness ─────────────────────────────────────────────────
    if logit_gap > 3.0:
        gap_desc = "The model is highly decisive between real and fake classes."
    elif logit_gap > 1.5:
        gap_desc = "The model shows a clear preference for the predicted class."
    else:
        gap_desc = (
            "The logit gap is small — the model found this image ambiguous. "
            "Manual inspection is recommended."
        )

    # ── Compose final reasoning ───────────────────────────────────────
    if label == 0:  # REAL
        return (
            f"Classified as REAL with {conf_desc} ({confidence*100:.1f}%). "
            f"{clip_desc.capitalize()}. "
            f"In the spectral domain: {fft_desc}. "
            f"{gap_desc}"
        )
    else:  # FAKE
        return (
            f"Classified as FAKE with {conf_desc} ({confidence*100:.1f}%). "
            f"Two independent signals support this: "
            f"(1) {clip_desc}; "
            f"(2) {fft_desc}. "
            f"{gap_desc}"
        )


# ── Forward pass with feature hooks ───────────────────────────────────────

def predict_with_reasoning(model, tensor):
    """
    Runs the model, captures CLIP and FFT intermediate features via hooks,
    and returns prediction + rule-based reasoning.
    """
    clip_feat_holder = {}
    fft_feat_holder  = {}

    def clip_hook(module, input, output):
        clip_feat_holder["feat"] = output.detach()

    def fft_hook(module, input, output):
        fft_feat_holder["feat"] = output.detach()

    # Hook onto the LayerNorm at the output of each branch
    h1 = model.feature_proj.norm.register_forward_hook(clip_hook)
    h2 = model.fft_branch.norm.register_forward_hook(fft_hook)

    with torch.no_grad():
        out        = model(tensor)
        cls_logits = out["cls_logits"]           # (1, 2)

    h1.remove()
    h2.remove()

    probs      = F.softmax(cls_logits, dim=-1)
    label      = cls_logits.argmax(dim=-1).item()
    confidence = probs[0, label].item()
    logit_real = cls_logits[0, 0].item()
    logit_fake = cls_logits[0, 1].item()
    logit_gap  = abs(logit_fake - logit_real)

    clip_norm  = clip_feat_holder["feat"].norm(dim=-1).mean().item()
    fft_norm   = fft_feat_holder["feat"].norm(dim=-1).mean().item()

    reasoning  = generate_reasoning(
        label, confidence, clip_norm, fft_norm, logit_gap)

    return {
        "label":      label,
        "confidence": confidence,
        "logit_real": logit_real,
        "logit_fake": logit_fake,
        "logit_gap":  logit_gap,
        "clip_norm":  clip_norm,
        "fft_norm":   fft_norm,
        "reasoning":  reasoning,
    }


# ── Main inference loop ────────────────────────────────────────────────────
print("=" * 60)
for img_name in IMAGE_FILES:
    img_path = os.path.join(BASE_DIR, img_name)

    if not os.path.exists(img_path):
        print(f"{img_name} → NOT FOUND at {img_path}")
        print("─" * 60)
        continue

    img    = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    r = predict_with_reasoning(model, tensor)

    print(f"Image      : {img_name}")
    print(f"Prediction : {LABEL[r['label']]}  ({r['confidence']*100:.1f}% confidence)")
    print(f"Logits     : real={r['logit_real']:.3f}  fake={r['logit_fake']:.3f}  "
          f"gap={r['logit_gap']:.3f}")
    print(f"Features   : CLIP norm={r['clip_norm']:.2f}  FFT norm={r['fft_norm']:.2f}")
    print(f"Reasoning  : {r['reasoning']}")
    print("─" * 60)