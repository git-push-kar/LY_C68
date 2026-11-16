# DeepfakeReasoningModel v2: Production Integration & Engineering Guide

**Complete Architecture Specification, API Contracts, and Deployment Guide for System Integrators**  
*Target Audience: Backend Engineers, ML Ops Teams, and API Integrators*

---

## 1. Executive System Overview

The **DeepfakeReasoningModel v2** is a multimodal deepfake detection and explainability system. Unlike traditional black-box classification networks, it provides a dual-output interface on every single image:

1. **High-Precision Deterministic Classification:** A mathematical prediction from fused visual and frequency-domain representations ($P(\text{real}), P(\text{fake})$ and confidence score).
2. **Faithful Causal Reasoning:** Human-readable forensic thought traces with `<fast>`, `<planning>`, `<reasoning>`, `<reflection>`, `<conclusion>`, and `<answer>` XML tags, mathematically grounded in real-time sensor telemetry.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                             CLIENT / APPLICATION                         │
│                    (REST API, Webhook, Microservice, UI)                 │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │ Raw Image (File / URL / Base64)
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         PREPROCESSING PIPELINE                           │
│  - Spatial Resize & Normalization: (3, 448, 448)                         │
│  - 2D Fast Fourier Transform:      (3, 224, 224) log-magnitude           │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                   DEEPFAKE REASONING ENGINE (CORE MODEL)                 │
│                                                                          │
│  ┌───────────────────────────┐         ┌──────────────────────────────┐  │
│  │   Classification Branch   │         │       Reasoning Branch       │  │
│  │  - InternViT-300M (LoRA)  │         │  - 256 Native Visual Tokens  │  │
│  │  - 2D FFT CNN Branch      │         │  - Real-time Telemetry Embed │  │
│  │  - Spatial AttentionPool  │         │  - Qwen2.5-1.5B Causal LM    │  │
│  └─────────────┬─────────────┘         └──────────────┬───────────────┘  │
│                │                                      │                  │
│                ▼                                      ▼                  │
│      [P_fake, Hotspots, FFT]               [Structured XML Chain]        │
└────────────────┬──────────────────────────────────────┬──────────────────┘
                 │                                      │
                 └───────────────────┬──────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       STANDARDIZED JSON RESPONSE                         │
│  - verdict: "FAKE" | "REAL"                                              │
│  - confidence: 0.987                                                     │
│  - telemetry: { hotspots: [...], fft_ratio: 0.48 }                       │
│  - reasoning: { quick_take, planning, analysis, reflection, conclusion } │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Hardware and Environment Requirements

### 2.1 Compute and VRAM Specifications
* **Minimum GPU:** NVIDIA GPU with $\ge 12\text{ GB}$ VRAM (e.g., RTX 3060 12GB, RTX 4070, T4).
* **Recommended GPU (Batch size $\ge 4$):** NVIDIA GPU with $\ge 16\text{ GB}$–$24\text{ GB}$ VRAM (e.g., RTX 3090, RTX 4090, RTX A5000, A10, V100, A100).
* **RAM:** Minimum 16 GB system memory (32 GB recommended).
* **Disk Space:** ~8.0 GB for full-checkpoint deployment (vanilla base `model.safetensors` ~3.89 GB + `runs/intern_v2_ground/checkpoints/best.pth` ~4.07 GB), or ~4.1 GB for adapter deployment (vanilla base ~3.89 GB + `llm_adapter` ~141 MB + `vision_adapter` ~13 MB + `custom_heads` ~26 MB).

### 2.2 Pinned Python Dependencies
```txt
torch==2.6.0
torchvision==0.21.0
transformers==4.37.2
peft==0.7.1
accelerate==1.14.0
safetensors>=0.4.0
einops>=0.7.0
timm>=0.9.0
tokenizers>=0.15.0
Pillow>=10.0.0
numpy>=1.24.0,<2.3.0
scikit-learn>=1.3.0
PyYAML>=6.0
```

> **CRITICAL NOTE ON TRANSFORMERS VERSION:**  
> `transformers` must remain pinned to `4.37.2` (or `<5.0.0`). The custom InternVL3 remote architecture relies on `_tied_weights_keys`, which is altered in Transformers 5.x.

---

## 3. Data Inputs and Preprocessing Pipeline

Every inference request requires two synchronized tensors derived from the source image:

```
Source Image (PIL Image / RGB)
 ├──> Resize & CenterCrop (448x448) ──> Normalize ──> pixel_values:  (B, 3, 448, 448) [bfloat16/float32]
 └──> Resize (224x224) ──> 2D FFT ──> Log-Magnitude ──> freq_features: (B, 3, 224, 224) [float32]
```

### 3.1 Preprocessing Code
```python
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

def build_transform(image_size: int = 448):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

def compute_fft_features(img: Image.Image, target_size: int = 224) -> torch.Tensor:
    """
    Computes normalized log-magnitude 2D Fourier spectrum across RGB channels.
    Output shape: (3, target_size, target_size)
    """
    img_resized = img.resize((target_size, target_size), Image.BILINEAR)
    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    elif img_np.shape[2] == 4:
        img_np = img_np[:, :, :3]

    img_np = np.transpose(img_np, (2, 0, 1)) # (3, H, W)
    fft = np.fft.fft2(img_np, axes=(-2, -1))
    fft_shift = np.fft.fftshift(fft, axes=(-2, -1))
    magnitude = np.abs(fft_shift)
    log_magnitude = np.log1p(magnitude)

    # Normalize across spatial plane
    norm = np.zeros_like(log_magnitude)
    for c in range(log_magnitude.shape[0]):
        c_min, c_max = log_magnitude[c].min(), log_magnitude[c].max()
        norm[c] = (log_magnitude[c] - c_min) / (c_max - c_min + 1e-8)

    return torch.from_numpy(norm).float()
```

---

## 4. Model Architecture & Tensor Flow Contract

### 4.1 Internal Tensor Dimensions Table

| Stage | Input Tensor | Output Tensor | Shape / Dimension | Component / Layer |
| :--- | :--- | :--- | :--- | :--- |
| **Vision Encoding** | `pixel_values` | `patch_tokens`<br>`cls_token` | `(B, 1024, 1024)`<br>`(B, 1024)` | `backbone.vision_model` (InternViT-300M + LoRA $r=16$) |
| **Spatial Pooling** | `patch_tokens` | `pooled_patches`<br>`attn_weights` | `(B, 1024)`<br>`(B, 1024)` | `AttentionPool` (Learned cross-attention query) |
| **Frequency CNN** | `freq_features`| `freq_emb` | `(B, 256)` | `FrequencyBranch` (4-stage Conv2D + AdaptiveAvgPool) |
| **Classifier Fusion**| Concatenated | `cls_logits` | `(B, 2)` | `ClassificationHead` (Linear $2304 \rightarrow 512 \rightarrow 2$) |
| **Token Projection** | `patch_tokens` | `vis_tokens` | `(B, 256, 2048)` | `PixelShuffle(0.5x)` + `backbone.mlp1` |
| **Telemetry Prefix** | Telemetry Str | `tele_embeds` | `(B, N_tele, 2048)`| Tokenizer + `language_model.get_input_embeddings()` |
| **LLM Reasoning** | `[vis \| tele]` | Autoregressive text | Up to $512$ tokens | `backbone.language_model` (Qwen2.5-1.5B + LoRA $r=64$) |

## 5. Model Initialization and Checkpoint Loading

> **Deployed model (current):** the shipped detector is the **grounded vanilla** checkpoint
> `runs/intern_v2_ground/checkpoints/best.pth` (epoch 3, `best_auc` 0.990270, 1,328 tensors,
> ~4.07 GB) attached to the **upstream `./models/InternVL3-2B`** base, with the adapter package in
> `runs/intern_v2_ground/adapters/` (18/18 verdict parity, Δp_fake = 0). Lineage A
> (`internvl3_2b_lr_fused` + `runs/intern_fused_ground`) is retained as the comparison model —
> it is *not* what `config.yaml`, `inference_faces.py`, `face_calibration.py`, `face_timing.py`
> or `verify_adapter_parity.py` default to any more. §5.1 documents the fused base for completeness
> because it is still the foundation of the Lineage A numbers in the paper.

### 5.1 Base Model Architecture & Fusion Provenance (`internvl3_2b_lr_fused` — Lineage A, comparison)
The foundation weights located at `./models/internvl3_2b_lr_fused` are not an off-the-shelf vanilla checkpoint. The base model is **InternVL3-2B (InternViT-300M + Qwen2.5-1.5B)** that was pre-adapted using **Parameter-Efficient Fine-Tuning (PEFT/LoRA) on a specialized Logical Reasoning (LR) dataset**, after which the LoRA adapter weights were **permanently merged and fused back into the base transformer matrices**:
$$W_{\text{fused}} = W_{\text{base}} + \frac{\alpha_{\text{LR}}}{r_{\text{LR}}} (B_{\text{LR}} \cdot A_{\text{LR}})$$

**Why this matters for engineering and deployment:**
* **Enhanced Causal Discipline:** Pre-adapting on logical reasoning prevents the language model from jumping to intuitive, superficial conclusions or outputting generic mode-collapsed text. It establishes structured deductive pathways before deepfake-specific training.
* **Standard Hugging Face Format:** Because the adapter was merged into the base weights, `./models/internvl3_2b_lr_fused` functions as a standard, self-contained Hugging Face model directory (`config.json`, `model.safetensors`). System integrators do not need to download or load a secondary base adapter at runtime.
* **Independent Forensic Adapter Injection:** The fine-tuned forensic weights inject their own independent, task-specific LoRA adapters and custom classifier heads on top of the base model. The **deployed** package does this on the upstream vanilla base (`runs/intern_v2_ground` + `./models/InternVL3-2B`); the Lineage A package does the same on the LR-fused base (`runs/intern_fused_ground` + `./models/internvl3_2b_lr_fused`).

### 5.2 Hybrid Weight Loading Pipeline
The runtime loading pipeline operates in two clean phases:
1. **Base Model:** Loaded directly via Hugging Face `AutoModel.from_pretrained()` — `./models/InternVL3-2B` for the deployed model.
2. **Forensic Checkpoint State Dict:** Loaded into the instantiated model using PyTorch's native `load_state_dict(..., strict=False)`.

> **Checkpoint-type note (deployed model, verified 2026-11-16):** `runs/intern_v2_ground/checkpoints/best.pth` (epoch 3, `best_auc` 0.990270, 1,328 tensors, ~4.07 GB) is a **whole-model** checkpoint — frozen vanilla backbone + 584 LoRA tensors (392 LLM r=64 + 192 vision r=16) + 59 custom-head tensors (`cls_head` frozen during grounded post-training) — **not** adapter-only. Use §5.2 when you can ship the 4.07 GB file; use §5.3 (adapters, ~180 MB total) for lightweight redistribution. Both paths require the **same** base: `./models/InternVL3-2B`. Lineage A equivalents: `runs/intern_fused_ground/checkpoints/best.pth` (~4.36 GB) on `./models/internvl3_2b_lr_fused`.

```python
import torch
from transformers import AutoTokenizer
from model import DeepfakeReasoningModel

def load_detector_service(
    model_path: str = "./models/InternVL3-2B",
    checkpoint_path: str = "./runs/intern_v2_ground/checkpoints/best.pth",
    device: str = "cuda"
):
    print(f"Loading base model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Instantiate model
    model = DeepfakeReasoningModel(
        model_path=model_path,
        arch_version="v2",
        attn_implementation="sdpa",
        use_gradient_checkpointing=False, # Disable during inference for speed
    ).to(device)

    print(f"Applying checkpoint weights from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Model ready. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    model.eval()
    return model, tokenizer
```

### 5.3 Lightweight Adapter Deployment (PEFT + Custom Heads)
Generated by the updated `extract_adapters.py` (defaults already point at the final model):
```bash
python extract_adapters.py \
    --checkpoint ./runs/intern_v2_ground/checkpoints/best.pth \
    --output_dir ./runs/intern_v2_ground/adapters \
    --base_model ./models/InternVL3-2B
```
Output package (`runs/intern_v2_ground/adapters/`, see `extraction_report.json`):
| Path | Contents | Size |
| :--- | :--- | :--- |
| `llm_adapter/adapter_model.safetensors` + `adapter_config.json` | Qwen2.5 LoRA, r=64/a=128, 392 tensors, 73,859,072 params, bf16 | ~141 MB |
| `vision_adapter/adapter_model.safetensors` + `adapter_config.json` | InternViT LoRA, r=16/a=32, targets `qkv/proj/fc1/fc2`, 192 tensors, 6,291,456 params, bf16 | ~13 MB |
| `custom_heads/custom_heads.safetensors` | `cls_head` + `attn_pool` + `freq_branch` + `custom_projector` + `domain_router`, 59 tensors, 13,257,034 params | ~26.5 MB |

The two adapters cooperate at inference: the **vision adapter** enriches every image forward (classification features + attention hotspots + visual tokens), the **LLM adapter** drives the grounded reasoning generation, and the **custom heads** supply the classifier/pooling/spectral math the adapters alone cannot reproduce (`cls_head` was frozen during ground post-training but is still load-bearing at inference).

```python
import torch
from safetensors.torch import load_file as safetensors_load
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
from model import DeepfakeReasoningModel

def load_detector_service_adapters(
    model_path: str = "./models/InternVL3-2B",
    adapter_dir: str = "./runs/intern_v2_ground/adapters",
    device: str = "cuda",
):
    import os
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Fresh shell provides correctly-shaped custom heads + PEFT scaffolding.
    model = DeepfakeReasoningModel(
        model_path=model_path, arch_version="v2",
        attn_implementation="sdpa", use_gradient_checkpointing=False,
    ).to(device)

    # 1. Attach LoRA adapters onto the live sub-modules.
    #    (Equivalent manual form: PeftModel.from_pretrained(base.language_model, ...).)
    from peft import PeftModel, get_peft_model_state_dict  # noqa
    for subdir, attr in (("llm_adapter", "language_model"), ("vision_adapter", "vision_model")):
        sd = safetensors_load(os.path.join(adapter_dir, subdir, "adapter_model.safetensors"), device="cpu")
        tgt = getattr(model.backbone, attr)
        # PEFT adapter-name-free key space matches the extracted files 1:1;
        # copy into the live lora_A/lora_B default-adapter tensors.
        live = dict(tgt.named_parameters())
        for k, v in sd.items():
            for suffix in (".lora_A.default.weight", ".lora_B.default.weight"):
                if k.endswith(suffix):
                    stem = k[: -len(suffix)]
                    cand = next((lk for lk in live if lk.endswith(stem + suffix)), None)
                    if cand is not None:
                        live[cand].data.copy_(v.to(live[cand].dtype))
                    break

    # 2. Load full-finetune custom heads (strict=False: backbone keys intentionally absent).
    ch = safetensors_load(os.path.join(adapter_dir, "custom_heads", "custom_heads.safetensors"), device="cpu")
    model.load_state_dict(ch, strict=False)

    model.eval()
    return model, tokenizer
```

> Round-trip integrity (exact tensor equality after save/reload) is verified by the extractor on every run. Pass `--skip_functional_check` to skip the clean-base `PeftModel.from_pretrained` attach test, `--no_custom_heads` for adapters-only output.

---

## 6. End-to-End Inference API Wrapper

Below is a self-contained, production-ready inference function that takes an image and returns a clean, structured Python dictionary:

```python
import re
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Dict, Any, Optional

_TAG_RE = re.compile(
    r"<(fast|planning|reasoning|reflection|conclusion)>(.*?)</\1>",
    re.DOTALL | re.IGNORECASE,
)
_ANSWER_RE = re.compile(r"<answer>\s*(real|fake)\s*</answer>", re.IGNORECASE)

def parse_reasoning_tags(text: str) -> Dict[str, str]:
    tags = {m.group(1).lower(): m.group(2).strip() for m in _TAG_RE.finditer(text)}
    m = _ANSWER_RE.search(text)
    tags["answer"] = m.group(1).lower() if m else "unknown"
    return tags

@torch.no_grad()
def analyze_image(
    image: Image.Image,
    model: DeepfakeReasoningModel,
    tokenizer,
    device: str = "cuda",
    use_grounded_telemetry: bool = True,
    temperature: float = 0.6,
    repetition_penalty: float = 1.15
) -> Dict[str, Any]:
    """
    Executes complete dual-branch detection on a single PIL Image.
    """
    transform = build_transform(448)
    pv = transform(image.convert("RGB")).unsqueeze(0).to(device)
    freq = compute_fft_features(image, target_size=224).unsqueeze(0).to(device)

    # 1. Classifier Forward Pass
    with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.bfloat16):
        out = model(pv, freq_features=freq)
        logits = out["cls_logits"].float()
        probs = F.softmax(logits, dim=-1)
        cls_pred = logits.argmax(-1).item()
        cls_conf = probs[0, cls_pred].item()

    # 2. Extract Telemetry & Hotspots
    telemetry = None
    prompt_text = None
    if use_grounded_telemetry:
        from inference import extract_forensic_telemetry, build_telemetry_prompt
        _, _, _, attn_weights = model._get_vision_features(pv, return_attn_weights=True)
        telemetry = extract_forensic_telemetry(
            cls_pred=cls_pred,
            cls_conf=cls_conf,
            freq_features=freq,
            attn_weights=attn_weights,
        )
        prompt_text = build_telemetry_prompt(telemetry)

    # 3. LLM Autoregressive Generation
    result = model.predict(
        pv, tokenizer, freq_features=freq,
        max_new_tokens=512,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=repetition_penalty,
        prompt_text=prompt_text,
    )

    raw_text = result["reasoning"][0]
    tags = parse_reasoning_tags(raw_text)
    verdict = "FAKE" if cls_pred == 1 else "REAL"
    llm_verdict = tags.get("answer", "unknown").upper()

    # 4. Arbitration / Disagreement Detection
    has_disagreement = (llm_verdict in ["REAL", "FAKE"]) and (llm_verdict != verdict)

    return {
        "verdict": verdict,
        "confidence": round(cls_conf, 4),
        "raw_probabilities": {
            "real": round(probs[0, 0].item(), 4),
            "fake": round(probs[0, 1].item(), 4),
        },
        "telemetry": {
            "hotspots": telemetry.get("regions") if telemetry else None,
            "spectral_analysis": telemetry.get("spectral_desc") if telemetry else None,
        },
        "reasoning": {
            "quick_take": tags.get("fast", ""),
            "planning": tags.get("planning", ""),
            "analysis": tags.get("reasoning", ""),
            "reflection": tags.get("reflection", ""),
            "conclusion": tags.get("conclusion", ""),
            "generated_answer": llm_verdict,
        },
        "flags": {
            "model_disagreement": has_disagreement,
            "requires_human_review": has_disagreement or (cls_conf < 0.70)
        }
    }
```

---

## 7. Sample JSON API Response Contract

Below is the exact schema returned by the integration wrapper:

```json
{
  "verdict": "FAKE",
  "confidence": 0.9872,
  "raw_probabilities": {
    "real": 0.0128,
    "fake": 0.9872
  },
  "telemetry": {
    "hotspots": "Forehead / Upper hairline, Nose & Cheek region",
    "spectral_analysis": "High-frequency attenuation (ratio=0.48); generative over-smoothing detected"
  },
  "reasoning": {
    "quick_take": "Inspection indicates synthetic generation, corroborated by spectral telemetry.",
    "planning": "Focus investigation on high-frequency boundary textures at the hairline and skin pore coherence.",
    "analysis": "The frequency sensor detects noticeable attenuation in fine detail (ratio 0.48). Corroborating this, the upper hairline displays an absence of fine individual hair strands, exhibiting a characteristic synthetic blur into the background. Skin pores along the upper nose and cheeks are artificially smoothed.",
    "reflection": "Checked facial symmetry; while macro-geometry appears plausible, localized frequency degradation confirms synthetic origin.",
    "conclusion": "Physical textural blurring at the hairline aligns with the classifier's spectral anomaly detection.",
    "generated_answer": "FAKE"
  },
  "flags": {
    "model_disagreement": false,
    "requires_human_review": false
  }
}
```

---

## 8. Microservice Integration (FastAPI Boilerplate)

To expose this model as a production REST API, use the following template:

```python
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from pydantic import BaseModel

app = FastAPI(title="Deepfake Reasoning Detection API", version="2.0")

# Global variables for warm service
MODEL = None
TOKENIZER = None

@app.on_event("startup")
def startup_event():
    global MODEL, TOKENIZER
    MODEL, TOKENIZER = load_detector_service()

@app.post("/api/v1/detect")
async def detect_deepfake(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    result = analyze_image(image, MODEL, TOKENIZER, device="cuda")
    return result

@app.get("/health")
def health_check():
    return {"status": "healthy", "gpu_available": torch.cuda.is_available()}
```

To run the server:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```
*(Note: Keep `--workers 1` per GPU to prevent duplicating the 5GB model in VRAM).*

---

## 9. Decision Arbitration & Guardrails Policy

In production environments, automated decision policies should follow these rules:

1. **High Confidence ($\text{Conf} \ge 0.85$) & Agreement:**  
   $\rightarrow$ Automated execution (verdict accepted immediately).
2. **Borderline Confidence ($0.60 \le \text{Conf} < 0.85$):**  
   $\rightarrow$ Accept classifier verdict, but log telemetry hotspots for review.
3. **Disagreement ($\text{Classifier} \ne \text{LLM}$):**  
   $\rightarrow$ Set `requires_human_review = true`. In all forensic and security deployments, **the Classifier verdict is treated as primary authority** because it directly inspects the Fourier frequency spectrum and attention matrices, whereas the LLM can occasionally be fooled by hyper-realistic semantic visuals.

---

## 10. File & Directory Reference

| File Path | Description | Integration Role |
| :--- | :--- | :--- |
| `model.py` | Complete PyTorch model definition (`DeepfakeReasoningModel`) | Core architecture import |
| `inference.py` | Command-line inference script with telemetry builders | Logic reference & CLI testing |
| `extract_adapters.py` | PEFT + custom-head extractor (experiment defaults → Lineage A `intern_fused_ground` / `internvl3_2b_lr_fused`; the deployed package is built with the §5.3 command) | Build lightweight deployable artifact |
| `dataset.py` | Data loaders, transformations, and FFT calculators | Preprocessing reference |
| `config.yaml` | Hyperparameters, paths, and benchmark metrics | Configuration defaults |
| `./models/InternVL3-2B` | Upstream vanilla base Hugging Face weights (~3.89 GB `model.safetensors`) | Required model directory (both deployment paths) |
| `./runs/intern_v2_ground/checkpoints/best.pth` | **Deployed** whole-model checkpoint, epoch 3 (1,328 tensors, ~4.07 GB; VaAcc 95.23%, VaAUC 0.990270) | Required weights file (full-checkpoint path, §5.2) |
| `./runs/intern_v2_ground/adapters/` | **Deployed** adapter package: `llm_adapter/` (~141 MB) + `vision_adapter/` (~13 MB) + `custom_heads/` (~26 MB) + `extraction_report.json` | Required weights dir (adapter path, §5.3) |
| `./models/internvl3_2b_lr_fused` | Lineage A comparison base (LR-fused, ~3.89 GB) | Comparison only — not deployed |
| `./runs/intern_fused_ground/` | Lineage A comparison checkpoint + adapters (epoch 3, ~4.36 GB) | Comparison only — not deployed |
| `adapter_loader.py` | Strict adapter/base loader (`load_model`, `attach_adapters`) | Same weights, verified key-for-key |
| `face_detect.py` | YOLOv8-face detector + context-crop geometry (`FaceBox`, `crop_with_context`, `filter_faces`) | Multi-face routing (§11) |
| `inference_faces.py` | Multi-face CLI: detect → batch classify → per-face telemetry → reasoning | Multi-face entry point (§11) |
| `run_faces.bat` | Windows launcher for `inference_faces.py` | Multi-face one-click run |
| `face_calibration.py` | Margin/threshold/floor sweep against FFIW | Re-derive the numbers in §11.5 |
| `FACE_PIPELINE_CALIBRATION.md` | Measured margin, threshold and floor results | Calibration record (§11.5) |
| `./models/yolov8n-face.pt` | Face detector weights, 6,247,065 bytes, SHA-256 pinned | Fetched by `fetch_detector_weights.py` (AGPL-3.0) |
| `verify_adapter_parity.py` | Adapter-vs-checkpoint verdict comparison | Pre-deployment parity gate |
| `test_passthrough_parity.py` | Bitwise check that single-face images match `inference.py` | Regression gate |

---

## 11. Multi-Face Images (detection + per-face crops)

The single-image path in §6 answers "is this face fake?". Given a group photo, it answers for the
whole scene instead: the FFT spectrum is computed over the frame and the attention hotspots span
every person, so one manipulated face can be averaged away, and a fake background can be blamed
on an innocent face. Multi-face mode fixes that by detecting the faces and asking the same model
about each one.

Nothing about the model changes. No retraining, no new preprocessing, no new thresholds inside
the network. Detection only decides **which image the model sees**, so the validated single-face
behaviour survives intact.

### 11.1 Routing rules

Detection runs once per image, then one of three paths is taken
(`inference_faces.route_image`, shared with the calibration sweep):

| Detected faces | Face area / frame | Path | What the model sees |
| :--- | :--- | :--- | :--- |
| 0 | — | `no_face` | whole image, `faces_detected: 0` (flagged, never silently "REAL") |
| 1 | ≥ 0.05 | `passthrough` | the whole image — **bitwise identical to `inference.py`** |
| 1 | < 0.05 | `per_face` | a context crop of that face (a lone small face usually means the detector missed others) |
| ≥ 2 | any | `per_face` | one context crop per face, largest first, max 8 |
| any | box shorter side < 96 px | skipped | not judged: reported as `face_too_small` rather than upscaled into a verdict |

Two ordering details matter and are covered by tests:

- **Passthrough is decided before the 96 px gate.** Passthrough crops nothing, so the crop-size
  gate must not veto it — a 200×200 training-style crop has an ~80 px face box and would
  otherwise be mislabelled `no_face`.
- **The crop is square.** The model does `Resize((448, 448))` with no letterboxing, so a 2:1 crop
  is distorted, not padded. Each box is expanded 30 % per side, squared about its centre, then
  clamped. Overflow up to 10 % of the side is mirror-padded; beyond that it degrades to the
  clamped rectangle (`pad_mode: "stretch"`) rather than inventing mirrored skin for the FFT
  branch to read as "evidence".

### 11.2 Running it

On Windows, `run_faces.bat` wraps the CLI (it fetches the pinned detector weights on first run):

```bat
run_faces.bat
run_faces.bat test_images\3-people.jpg test_images\multiple-faces.jpg
```

Directly:

```bash
python inference_faces.py \
    --adapter_dir ./runs/intern_v2_ground/adapters \
    --images test_images/3-people.jpg test_images/multiple-faces.jpg \
    --json_out runs/face_results.json
```

| Flag | Default | Meaning |
| :--- | :--- | :--- |
| `--adapter_dir` / `--checkpoint` | — | exactly one weight source; both paths verified equal by `verify_adapter_parity.py` |
| `--images` | 2 sample images | input paths |
| `--json_out` | — | writes the §11.3 array |
| `--reason_faces {none,top1,all}` | `top1` | which faces get LLM reasoning; `top1` = highest `p_fake` |
| `--margin` | `0.30` | context ring around each face box |
| `--face_threshold` | `0.50` | per-face `p_fake` decision boundary |
| `--passthrough_min_face_frac` | `0.05` | lone-face area floor for the whole-image path |
| `--min_face_px` / `--max_faces` | `96` / `8` | skip below this box size / keep this many largest |
| `--aggregate {max,mean,any_fake,per_face}` | `max` | document verdict from per-face probabilities |
| `--no_detect` | off | reference mode: no detection, exactly `inference.py`'s behaviour |
| `--save_crops`, `--save_annotated` | off | dump crops / box overlays for debugging |

`max` is the default aggregation because one compromised identity is enough; `requires_human_review`
rises when faces disagree, when any face is below 0.70 confidence, or when the LLM answer
contradicts the classifier.

### 11.3 JSON contract

One array element per image (values below are a real 3-face run):

```json
[
  {
    "image": "3-people.jpg",
    "path": "per_face",
    "faces_detected": 3,
    "faces_evaluated": 3,
    "faces_skipped": [],
    "document_verdict": "FAKE",
    "aggregate": "max",
    "threshold": 0.5,
    "reason_mode": "top1",
    "faces": [
      {
        "id": 0,
        "kind": "face",
        "verdict": "FAKE",
        "p_fake": 0.965119,
        "confidence": 0.965119,
        "box": [311.0, 1204.0, 802.0, 1695.0],
        "crop": {"margin": 0.3, "pad_mode": "none", "crop_size": [786, 786],
                 "face_frac": 0.0294, "upscale": 0.57},
        "telemetry": {"hotspots": ["Peripheral / Ear & Hair blending seams", "Mouth & Lower jaw"],
                      "spectral": "HF/LF log-magnitude energy ratio=0.61 (...)"},
        "reasoning": {"quick_take": "...", "planning": "...", "analysis": "...",
                      "reflection": "...", "conclusion": "...",
                      "generated_answer": "fake"}
      },
      {"id": 1, "verdict": "REAL", "p_fake": 0.102305, "reasoning": null},
      {"id": 2, "verdict": "FAKE", "p_fake": 0.697197, "reasoning": null}
    ],
    "flags": {
      "faces_disagree": true,
      "model_disagreement": false,
      "requires_human_review": true,
      "faces_skipped": 0,
      "reasoned_faces": [0]
    }
  }
]
```

Notes that prevent integration bugs:

- `faces[].id` is always an integer. A whole-image row (passthrough, `no_face`, `no_detect`) is
  `id: 0` with `kind: "whole_image"`, so `reasoned_faces` never contains `null`.
- `p_fake` is `null`-free and rounded to 6 dp; `document_verdict` is `null` only when
  `--aggregate per_face` is requested.
- `faces_skipped` appears twice by design: the list (which boxes, and why) at image level, the
  count inside `flags` for cheap routing.

### 11.4 Embedding it in a service

The CLI is the tested path; a service that already holds the model can call the same functions
directly. This returns per-face probabilities with no LLM cost:

```python
from types import SimpleNamespace

import torch
from inference import build_transform
from face_detect import FaceDetector
from inference_faces import route_image, classify, aggregate_verdict

ROUTING = SimpleNamespace(margin=0.30, min_face_px=96, max_faces=8,
                          image_size=448, passthrough_min_face_frac=0.05)

@torch.no_grad()
def detect_document(image, model, detector, device, threshold=0.50):
    """Faces in, per-face verdicts and a document verdict out. No reasoning pass."""
    route = route_image(image, detector, ROUTING)          # detection happens once
    probs, _, _ = classify(model, [item["image"] for item in route["items"]],
                           build_transform(448), device)
    rows = [{"p_fake": round(float(probs[i, 1]), 6),
             "verdict": "FAKE" if float(probs[i, 1]) >= threshold else "REAL"}
            for i, item in enumerate(route["items"])]
    return {
        "path": route["path"],
        "faces_detected": len(route["boxes"]),
        "faces_skipped": route["skipped"],
        "faces": rows,
        "document_verdict": aggregate_verdict(rows, "max", threshold),
    }
```

`detector = FaceDetector("./models/yolov8n-face.pt")` and the model comes from
`adapter_loader.load_model(...)` (§5.3). Build order does not matter: the model build leaves
torch defaulting to bf16, and `FaceDetector` pins fp32 inside its own predict call so detection
never inherits it.

### 11.5 Calibration status

Defaults are measured, not guessed — full tables in `FACE_PIPELINE_CALIBRATION.md`:

| Parameter | Value | Evidence |
| :--- | :--- | :--- |
| `margin` | 0.30 | FFIW document AUC 0.8845 vs 0.8782 for tight crops (+0.63 pt); 0.45 scores higher but only by reducing upscaling on FFIW's 256 px frames |
| `face_threshold` | 0.50 | sweep flat within one standard error across 0.42–0.55 (n = 400); 0.50 also matches `inference.py`'s argmax |
| `passthrough_min_face_frac` | 0.05 | floors 0.03–0.06 route all 33 unique local images identically, with measured failures bracketing the range |
| `aggregate` | `max` | one compromised identity is enough; documented in §9 |

Two honest limits from that evaluation: FFIW as stored is **256 px and 92 % single-face**, so it
never exercises the crop path (the sweep scores crops directly, bypassing routing), and it ships
without ground-truth face boxes, so detector recall is unmeasured. Treat per-face numbers as
"competitive with", not "better than", the whole-image path.

### 11.6 Cost and operational limits

- **Classification is cheap**: all faces of an image go through one batched forward pass.
- **Reasoning is linear in face count** (`generate()` runs once per face), which is why `top1` is
  the default. On a 6-face photo, `--reason_faces all` is 6× the LLM latency. Use `none` for
  screening volumes and reason only about escalated cases.
- **Per-face verdicts on padded crops are out-of-distribution** relative to HydraFake training
  crops. Do not quote the benchmark figures from `RESEARCH_PAPER_REPORT.md` for this path.
- **A missed face cannot be reported as clean.** Always surface `faces_detected` alongside
  `document_verdict`; a REAL document on a single detected face is not evidence the photo is clean.
- **Not for identity**: no recognition or clustering is performed, and boxes are not tracked
  across video frames.
- **Detector licensing**: `models/yolov8n-face.pt` is AGPL-3.0. `FaceDetector` is a thin wrapper,
  so a permissively licensed detector (YuNet/SCRFD) can be swapped in one file without touching
  the classifier.
