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
* **Disk Space:** $\sim 8\text{ GB}$ total storage (base fused weights: $\sim 4.8\text{ GB}$, checkpoint: $\sim 300\text{ MB}$).

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

### 5.1 Base Model Architecture & Fusion Provenance (`internvl3_2b_lr_fused`)
The foundation weights located at `./models/internvl3_2b_lr_fused` are not an off-the-shelf vanilla checkpoint. The base model is **InternVL3-2B (InternViT-300M + Qwen2.5-1.5B)** that was pre-adapted using **Parameter-Efficient Fine-Tuning (PEFT/LoRA) on a specialized Logical Reasoning (LR) dataset**, after which the LoRA adapter weights were **permanently merged and fused back into the base transformer matrices**:
$$W_{\text{fused}} = W_{\text{base}} + \frac{\alpha_{\text{LR}}}{r_{\text{LR}}} (B_{\text{LR}} \cdot A_{\text{LR}})$$

**Why this matters for engineering and deployment:**
* **Enhanced Causal Discipline:** Pre-adapting on logical reasoning prevents the language model from jumping to intuitive, superficial conclusions or outputting generic mode-collapsed text. It establishes structured deductive pathways before deepfake-specific training.
* **Standard Hugging Face Format:** Because the adapter was merged into the base weights, `./models/internvl3_2b_lr_fused` functions as a standard, self-contained Hugging Face model directory (`config.json`, `model.safetensors`). System integrators do not need to download or load a secondary base adapter at runtime.
* **Independent Forensic Adapter Injection:** The fine-tuned forensic weights from `./runs/intern_fused_ground/checkpoints/best.pth` inject their own independent, task-specific LoRA adapters and custom classifier heads on top of this reasoning-enhanced base model.

### 5.2 Hybrid Weight Loading Pipeline
The runtime loading pipeline operates in two clean phases:
1. **Base Fused Model:** Loaded directly via Hugging Face `AutoModel.from_pretrained()`.
2. **Forensic Checkpoint State Dict:** Loaded into the instantiated model using PyTorch's native `load_state_dict(..., strict=False)`.

```python
import torch
from transformers import AutoTokenizer
from model import DeepfakeReasoningModel

def load_detector_service(
    model_path: str = "./models/internvl3_2b_lr_fused",
    checkpoint_path: str = "./runs/intern_fused_ground/checkpoints/best.pth",
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
| `dataset.py` | Data loaders, transformations, and FFT calculators | Preprocessing reference |
| `config.yaml` | Hyperparameters, paths, and benchmark metrics | Configuration defaults |
| `./models/internvl3_2b_lr_fused` | Base fused Hugging Face weights | Required model directory |
| `./runs/intern_fused_ground/checkpoints/best.pth` | Peak checkpoint ($0.9936$ AUC) | Required weights file |
