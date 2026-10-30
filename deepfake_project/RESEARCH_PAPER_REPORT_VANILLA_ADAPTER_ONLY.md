# DeepfakeReasoningModel v2 on the Vanilla InternVL3-2B Foundation

**Complete Method, Architecture and Adapter-Only Training Report**

*Subject: Lineage B — vanilla `InternVL3-2B` with frozen backbone, LoRA adapters and task heads. Latest artifact: `runs/intern_v2_ground/adapters/`.*

---

## Abstract

This report documents the **adapter-only** variant of DeepfakeReasoningModel v2: the same dual-branch forensic architecture, trained on the **vanilla InternVL3-2B** foundation with every pretrained transformer weight frozen. Only low-rank adapters (LoRA) and the task-specific heads receive gradients. In the classifier/joint stages this is **93,406,598 trainable parameters out of 2,182,364,038 (4.28%)**; in the grounded stage the classifier head is frozen too, leaving **92,816,004 (4.25%)**. The deployable result is a **178.24 MiB** package — two PEFT adapters plus 59 custom-head tensors — attached to a 3.89 GiB base checkpoint, instead of a 4.36 GiB whole-model file.

The report specifies the artifact completely: adapter configurations, per-module rank decomposition, tensor and parameter counts, dtypes, file sizes, SHA-256 digests, provenance, loading pattern and the verification that was actually performed. It then reports the measured metrics of the vanilla adapter-only models:

| Adapter-only model | Validation | 52,266-sample benchmark |
| :--- | :--- | :--- |
| Stage 1–2 (`intern_v2`, epoch 5) | 94.90% / AUC 0.9891 | **87.80% acc / 0.8659 F1 / 0.9479 AUC** |
| DPO (`intern_v2_dpo`, epoch 2) | not logged | 85.36% acc / 0.8268 F1 / 0.9423 AUC |
| Grounded (`intern_v2_ground`, epoch 3) | **95.23% / AUC 0.9903** | **not run** |

Three qualifications are load-bearing and are not buried: the 52k benchmark belongs to **`intern_v2/checkpoints/ep005.pth`**, not the current `best.pth` (a later run overwrote it); the DPO adapter package contains **no custom heads**, so it cannot classify on its own; and the repo's `eval.py`/`inference.py` accept whole-model checkpoints only — there is no packaged adapter loader yet.

---

## 1. What “Adapter-Only” Means Here

The backbone is loaded in bfloat16 and frozen. LoRA matrices are injected into the vision encoder and the language model, and five task heads are trained from scratch. No pretrained weight is updated. Consequently:

| Quantity | Value |
| :--- | ---: |
| Base model parameters | 2,182,364,038 |
| Frozen in Stages 1–2 | 2,088,957,440 |
| Trainable in Stages 1–2 (LoRA + all heads) | 93,406,598 (4.28%) |
| Trainable in Stage 4 (`cls_head` frozen) | 92,816,004 (4.25%) |
| LoRA-only parameters | 80,150,528 |
| Whole-model checkpoint size | 4,365,579,738 bytes (~4.36 GB) |
| Adapter + heads package | 186,898,028 bytes (178.24 MiB) |
| Base checkpoint on disk | 4,177,999,192 bytes (3.89 GiB) |

Two consequences follow. First, adapter-only training is cheap in optimizer state and portable, but it is **not a smaller model**: the 2.18B backbone is still required at inference. Second, any behaviour change must be expressible as a low-rank update to the frozen weights plus a new head; this limits how far the model can move from the vanilla foundation, which is exactly the variable the LR-fused lineage isolates.

The parameters listed for the 4.28% figure are the sum of the two LoRA adapters (80,150,528) and the five heads (13,256,070). During grounded post-training the 590,594-parameter classifier head is frozen, so the optimizer sees 92,816,004 parameters (584 LoRA tensors + 43 custom-head tensors).

---

## 2. Method and Architecture

### 2.1 Backbone and token layout

| Component | Specification |
| :--- | :--- |
| Vision encoder | InternViT-300M, 24 layers, hidden 1,024, patch 14, input 448×448 |
| Visual tokens | 32×32 = **1,024 patches** + 1 `[CLS]` |
| Language model | Qwen2.5-1.5B, 28 layers, hidden **1,536**, intermediate 8,960, 12 query heads / 2 KV heads (head dim 128), vocab **151,674** |
| Precision | bfloat16, no GradScaler |
| Attention backend | `eager` in every recorded run; the requested SDPA path is rejected by the InternVL chat wrapper |

### 2.2 Dual-branch dataflow

```
                      448×448 RGB image
                         │
        ┌────────────────┴─────────────────┐
        ▼                                  ▼
 InternViT-300M                      2D FFT log-magnitude
 + Vision LoRA (r=16)                (3×224×224, standardized)
        │                                  │
   ┌────┴─────┐                            ▼
   ▼          ▼                   4-stage CNN → 256-d
 [CLS]    1,024 patches
 (1024)      │
             ├── AttentionPool (8-head, learned query) → 1024-d
             │
             └── PixelShuffle 0.5× → 256 tokens → mlp1 → 256×1536
                                                       │
   [CLS] ‖ pooled ‖ freq  = 2,304-d                    │
             │                                         │
             ▼                                         ▼
     cls_head: 2304→256→2                  [256 visual tokens]
             │                                         │
             ▼                                         ▼
      P(real), P(fake)                    + telemetry prefix (≤160 tok)
             │                                         │
             ├── telemetry (verdict, FFT ratio,         ▼
             │   top-8 hotspots)                 Qwen2.5-1.5B
             │                                   + LLM LoRA (r=64)
             └── bidirectional KL ──────────────► six-tag reasoning
                                                   + <answer>
```

### 2.3 Dual LoRA, exactly

| Adapter | Rank | α | Dropout | Target modules | Tensors | Parameters |
| :--- | :---: | :---: | :---: | :--- | :---: | ---: |
| LLM LoRA | 64 | 128 | 0.05 | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | 392 | 73,859,072 |
| Vision LoRA | 16 | 32 | 0.05 | `qkv, proj, fc1, fc2` | 192 | 6,291,456 |

The counts are not estimates; they follow exactly from the module shapes and were confirmed by the extraction report. Per language-model layer:

| Module | Shape (out×in) | LoRA A + B | Parameters |
| :--- | :--- | :--- | ---: |
| `q_proj` | 1,536×1,536 | 64×1,536 + 1,536×64 | 196,608 |
| `k_proj` | 256×1,536 | 64×1,536 + 256×64 | 114,688 |
| `v_proj` | 256×1,536 | 64×1,536 + 256×64 | 114,688 |
| `o_proj` | 1,536×1,536 | 64×1,536 + 1,536×64 | 196,608 |
| `gate_proj` | 8,960×1,536 | 64×1,536 + 8,960×64 | 671,744 |
| `up_proj` | 8,960×1,536 | 64×1,536 + 8,960×64 | 671,744 |
| `down_proj` | 1,536×8,960 | 64×8,960 + 1,536×64 | 671,744 |
| **Per layer** | | | **2,637,824** |
| **× 28 layers** | | | **73,859,072** |

Per vision layer:

| Module | Shape (out×in) | Parameters |
| :--- | :--- | ---: |
| `qkv` | 3,072×1,024 | 65,536 |
| `proj` | 1,024×1,024 | 32,768 |
| `fc1` | 4,096×1,024 | 81,920 |
| `fc2` | 1,024×4,096 | 81,920 |
| **Per layer** | | **262,144** |
| **× 24 layers** | | **6,291,456** |

The standard LoRA update applies: $W = W_0 + \frac{\alpha}{r}BA$, with $B\in\mathbb{R}^{d_{out}\times r}$, $A\in\mathbb{R}^{r\times d_{in}}$, and dropout 0.05 on the adapter input.

### 2.4 Frequency branch

Per image (`dataset.compute_fft_features`): resize to 224×224, per-channel 2D FFT, `fftshift`, $\log(1+|F|)$, then per-channel standardization. The resulting $3\times224\times224$ tensor passes through four stride-2 conv blocks (3→32→64→128→256; BatchNorm + GELU), adaptive average pooling, a 256→256 linear layer and LayerNorm, producing $v_{\text{freq}}\in\mathbb{R}^{256}$.

### 2.5 Attention pooling

A single learned query cross-attends over all 1,024 patch tokens with 8 heads:

$$\alpha=\operatorname{softmax}\!\left(\frac{qK^{\top}}{\sqrt{1024}}\right),\qquad h=\operatorname{LayerNorm}\!\left(\sum_i\alpha_iV_i\right).$$

The head-averaged $\alpha$ is also the localization signal used in telemetry. Weights are requested only on the grounded joint path; classifier-only steps use `need_weights=False`.

### 2.6 Classifier and visual tokens

$$v_{\text{fused}}=[\,v_{\text{cls}}\,\|\,v_{\text{pool}}\,\|\,v_{\text{freq}}\,]\in\mathbb{R}^{2304},$$

$$z=\operatorname{Linear}_{256\to2}\!\Big(\operatorname{Dropout}_{0.3}\big(\operatorname{GELU}(\operatorname{Linear}_{2304\to256}(v_{\text{fused}}))\big)\Big).$$

The reasoning branch uses the same patch tokens: 0.5× pixel shuffle (1,024→256 tokens, 1,024→4,096 channels) followed by `mlp1` (LayerNorm, 4,096→1,536, GELU, 1,536→1,536), giving $V_{\text{vis}}\in\mathbb{R}^{256\times1536}$.

### 2.7 Telemetry and self-consistency

The LLM receives visual tokens, then a telemetry block, then the reasoning target. Visual and telemetry positions carry label −100. The shared prompt is:

```text
[Forensic Telemetry]
- Statistical read-out (verify against what you see, do not blindly repeat): FAKE (dataset label)
- Frequency measurement: HF/LF log-magnitude energy ratio=-0.23 (lower ratio = smoother image with less high-frequency detail)
- Attention hotspots: Nose & Cheek region, Eyes & Facial midline

Weigh the telemetry numbers against the visual evidence, then explain using
<fast>, <planning>, <reasoning>, <reflection> and <conclusion> tags only.
Finish with exactly <answer>real</answer> or <answer>fake</answer> and nothing after it.
```

Training uses the ground-truth verdict ("dataset label"); inference uses the classifier's predicted verdict and softmax confidence. The spectral ratio is the mean standardized log-magnitude in an outer annulus divided by the mean in a central disc ($r>0.35H$ vs $r<0.15H$); it is typically negative (≈ −0.21 to −0.28) and is passed as evidence, not thresholded. Hotspots are the top-8 attention indices mapped to six facial regions.

The consistency term aligns classifier probabilities with the language model's `real` (id 7951) / `fake` (id 30570) logits:

$$\mathcal{L}_{\text{cons}}=\tfrac12\big[D_{\mathrm{KL}}(P_{\text{lm}}\|P_{\text{cls}}^{\text{detach}})+D_{\mathrm{KL}}(P_{\text{cls}}\|P_{\text{lm}}^{\text{detach}})\big],$$

$$\mathcal{L}=\mathcal{L}_{\text{cls}}+\lambda_{\text{lm}}(e)\mathcal{L}_{\text{lm}}+0.05\,\mathcal{L}_{\text{cons}},$$

with label-smoothed cross-entropy (0.1) for classification and $\lambda_{\text{lm}}$ ramped linearly 0.30→0.60 across joint epochs.

### 2.8 Adapter-only training schedule (vanilla lineage)

| Stage | Epochs | Steps/epoch | Active parameters | Objective | Measured time |
| :--- | :---: | :---: | :--- | :--- | :--- |
| 1 — classifier warm-up | 3 | 12,080 | Vision LoRA + 5 heads | $\mathcal{L}_{\text{cls}}$ | ≈2 h 20 m/epoch, 0.69 s/step |
| 2 — joint | 3 | 8,765 | All LoRA + 5 heads | $\mathcal{L}$ above | 5 h 39 m – 5 h 59 m/epoch |
| 3 — DPO | 2 | 870 | LLM + vision LoRA | DPO ($\beta$=0.1, lr 1e-6) | 1 h 10 m/epoch |
| 4 — grounded | 3 | 8,765 | LoRA + 4 heads (`cls_head` frozen) | Grounded $\mathcal{L}$ | 6 h 54 m – 7 h 03 m/epoch; 21 h 11 m total |

Optimizer: AdamW, LoRA at $5\times10^{-5}$, custom heads at $1\times10^{-4}$, weight decay 0.01, cosine decay after 5% warm-up, gradient clipping 1.0, bf16 autocast, LM gradient checkpointing. Data: 48,320 classification images (`all.json`), 35,063 reasoning samples (`sft_36k.json`), 3,480 preference pairs (`mipo_3k.json`), 4,000 balanced validation samples, 52,266 held-out test samples.

---

## 3. The Adapter-Only Artifact

### 3.1 Primary package: `runs/intern_v2_ground/adapters/`

Extracted on 21 Oct 2026 from `runs/intern_v2_ground/checkpoints/best.pth` (grounded epoch 3).

| File | Bytes | MiB | Tensors | Parameters / elements |
| :--- | ---: | ---: | :---: | ---: |
| `llm_adapter/adapter_model.safetensors` | 147,770,856 | 140.93 | 392 | 73,859,072 |
| `vision_adapter/adapter_model.safetensors` | 12,607,408 | 12.02 | 192 | 6,291,456 |
| `custom_heads/custom_heads.safetensors` | 26,519,764 | 25.29 | 59 | 13,257,034 (13,256,070 params + 964 buffers) |
| **Total** | **186,898,028** | **178.24** | **643** | **93,406,598 trainable-equivalent params** |

SHA-256 of the shipped files:

| File | SHA-256 |
| :--- | :--- |
| `llm_adapter/adapter_model.safetensors` | `ec210956b1273cc57e10c5dab9b2d8eedb96f009cb62132c226004583afa63ef` |
| `vision_adapter/adapter_model.safetensors` | `fbf3b50455f02bf5500bfd845a4605852419fa732af71d2d64003664c6b86819` |
| `custom_heads/custom_heads.safetensors` | `55c225843f8abd42eb8594175daf0cf1906878203548aec70d815fb50159b5c7` |

Adapter configurations as written to disk:

| Field | LLM adapter | Vision adapter |
| :--- | :--- | :--- |
| `peft_type` | LORA | LORA |
| `task_type` | CAUSAL_LM | `null` |
| `r` / `lora_alpha` / `lora_dropout` | 64 / 128 / 0.05 | 16 / 32 / 0.05 |
| `target_modules` | 7 projections listed in §2.3 | `fc1, fc2, proj, qkv` |
| `bias` | none | none |
| `base_model_name_or_path` | `./models/InternVL3-2B` | `./models/InternVL3-2B` |
| `init_lora_weights` | true | true |

### 3.2 Custom heads inside the package

| Module | Tensors | Parameters | Role |
| :--- | :---: | ---: | :--- |
| `cls_head` | 4 | 590,594 | 2,304→256→2 classifier |
| `attn_pool` | 7 | 4,201,472 | learned query + 8-head cross-attention + LayerNorm |
| `freq_branch` | 32 | 455,680 | FFT CNN; plus 964 BatchNorm buffers (running stats + counters) |
| `custom_projector` | 12 | 7,876,608 | 2-token fallback projector (1,024→1,536→1,536, twice) |
| `domain_router` | 4 | 131,716 | 1,024→128→4 compatibility stub |
| **Total** | **59** | **13,256,070** | all tensors bfloat16 except 4 int64 counters |

`domain_router` is never called in the forward pass and `custom_projector` is only used when the backbone exposes no `mlp1`; both are nonetheless part of the checkpoint and the package, which is why the package is 178 MiB rather than ~150 MiB.

### 3.3 What the verification actually covers

- **Adapter extraction:** key filtering, A/B pairing assertion, single-dtype assertion, and exact safetensors round-trip for all 584 adapter tensors. Passed.
- **Adapter attach:** a clean `AutoModel.from_pretrained("./models/InternVL3-2B")` is loaded, each adapter attached with `PeftModel.from_pretrained`, and the resulting PEFT state dict compared tensor-for-tensor with the extracted files. Passed for both adapters.
- **Custom heads:** exact safetensors round-trip for all 59 tensors. Passed. Loading into a fresh model with `load_state_dict(..., strict=False)` is documented but **not** exercised by the extractor's functional check.
- **Not verified:** an end-to-end classifier-score reproduction from the package. Neither `eval.py` nor `inference.py` accepts adapter directories; both expect a whole-model `.pth`. Deploying the package therefore requires a small loader (build the model, attach both adapters, load the heads) that is not in the repository.

The artifact records its provenance in `extraction_report.json`:

```json
"provenance": {
  "run": "intern_v2_ground",
  "training": "intern_v2_ground grounded joint post-training from ./runs/intern_v2/checkpoints/ep006.pth",
  "base_model": "./models/InternVL3-2B",
  "frozen": "cls_head (4 tensors); trainable: LLM LoRA r=64 + vision LoRA r=16 + attn_pool/freq_branch/custom_projector/domain_router",
  "checkpoint_type": "whole-model state_dict (backbone + LoRA + custom heads), NOT adapter-only"
}
```

### 3.4 Other vanilla-base adapter artifacts

| Package | Source | Contents | Adapter bytes | Heads? |
| :--- | :--- | :--- | ---: | :---: |
| `runs/intern_v2_ground/adapters/` | grounded `best.pth`, epoch 3 | LLM LoRA + vision LoRA + 5 heads | 186,898,028 (178.24 MiB) | yes |
| `runs/intern_v2_dpo/adapters/` | `dpo_best.pth`, epoch 2 | LLM LoRA + vision LoRA **only** | 160,378,264 (152.95 MiB) | **no** |
| `runs/intern_v2/adapters/` | — | not produced | — | — |

The DPO package is a genuine gap: DPO updates only the 80,150,528 LoRA parameters, leaving `cls_head`, `attn_pool`, `freq_branch`, `custom_projector` and `domain_router` exactly as they were in its init checkpoint (`intern_v2/checkpoints/ep006.pth`). Because those heads are not included, the DPO package **cannot classify on its own**; it needs the heads from `ep006.pth` (which itself has no package) or a re-extraction with heads enabled. Anyone deploying "DPO adapters only" gets a language model without a classifier.

### 3.5 Loading pattern (documented, not packaged)

```python
from transformers import AutoModel
from peft import PeftModel
import torch
from safetensors.torch import load_file

base = AutoModel.from_pretrained(
    "./models/InternVL3-2B", trust_remote_code=True, torch_dtype=torch.bfloat16)

base.language_model = PeftModel.from_pretrained(
    base.language_model, "runs/intern_v2_ground/adapters/llm_adapter")
base.vision_model = PeftModel.from_pretrained(
    base.vision_model, "runs/intern_v2_ground/adapters/vision_adapter")

heads = load_file("runs/intern_v2_ground/adapters/custom_heads/custom_heads.safetensors")
# load `heads` into cls_head / attn_pool / freq_branch / custom_projector /
# domain_router of the DeepfakeReasoningModel wrapper with strict=False
```

Compatibility rules, all of them mandatory:

1. The base must be the **vanilla** `models/InternVL3-2B`. The LR-fused base is a different checkpoint and will not reproduce these weights.
2. All three files must travel together; the two adapters alone do not reproduce classifier scores.
3. bfloat16 is required; the adapters and heads were extracted in bf16.
4. The adapter configs name `./models/InternVL3-2B` as the base; a relocated base needs that path updated or the configs passed explicitly.

---

## 4. Metrics of the Vanilla Adapter-Only Models

### 4.1 Provenance of every number

The vanilla directory contains **two stage-2 runs**, and the distinction matters:

| Date (2026) | Event | Artifact | Validation |
| :--- | :--- | :--- | :--- |
| before 25 Sep 06:12 | Run A Stages 1–2, epochs 1–6 | `ep005.pth`, `ep006.pth` | ep5 94.90% / 0.989061; ep6 93.275% / 0.984804 |
| 25 Sep 16:05 | 52k evaluation of the then-best (epoch 5) | `logs/eval_best.log` | checkpoint `ep005.pth` |
| 25 Sep 18:42 – 26 Sep 05:26 | Run B Stages 1–2, epochs 1–4 | `best.pth` (overwrote Run A's) | ep4 94.525% / 0.991403 |
| 26 Sep 11:08 – 13:29 | DPO from `ep006.pth` | `intern_v2_dpo/dpo_best.pth` (epoch 2) | no validation logged |
| 18–19 Oct | Grounded from `ep006.pth` | `intern_v2_ground/best.pth` (epoch 3) | 95.225% / 0.990270 |
| 21 Oct 09:13 | Adapter extraction | `intern_v2_ground/adapters/` | — |

The evaluated epoch-5 checkpoint still exists as `ep005.pth` (metadata: `epoch=5`, `best_auc=0.9890605`), so the 52k metrics are reproducible against a named file. The current `best.pth` is Run B's epoch-4 checkpoint (metadata: `epoch=4`, `best_auc=0.99140275`) and has **no archived 52k evaluation**.

### 4.2 Validation trajectories

Stage 1–2 on the vanilla base (from `runs/intern_v2/logs/epochs.csv`; two runs appended):

| Run | Epoch | Stage | Val acc | Val AUC | Note |
| :---: | :---: | :--- | ---: | ---: | :--- |
| A | 1 | cls_only | 91.50% | 0.978543 | |
| A | 2 | cls_only | 89.43% | 0.984268 | |
| A | 3 | cls_only | 93.30% | 0.988136 | |
| A | 4 | joint | 90.68% | 0.983361 | |
| A | 5 | joint | **94.90%** | **0.989061** | 52k benchmark |
| A | 6 | joint | 93.28% | 0.984804 | grounded init |
| B | 1 | cls_only | 91.90% | 0.983726 | |
| B | 2 | cls_only | 94.53% | 0.990495 | |
| B | 3 | cls_only | 94.63% | 0.991196 | |
| B | 4 | joint | **94.53%** | **0.991403** | current `best.pth` |

The joint stage is non-monotonic in Run A (94.90% at epoch 5, 93.28% at epoch 6). The grounded stage was continued from the end of the curriculum (`ep006.pth`), not from the peak checkpoint (`ep005.pth`), so the production chain is stage-2 end → grounded rather than stage-2 best → grounded.

Grounded adapter-only model (`runs/intern_v2_ground/logs/epochs.csv`):

| Epoch | $\lambda_{\text{lm}}$ | Train cls | Train LM | Cons | Val loss | Val acc | Val AUC |
| :---: | :---: | :---: | :---: | :---: | :---: | ---: | ---: |
| 1 | 0.30 | 0.3094 | 0.7284 | 0.0122 | 0.3353 | 93.08% | 0.986410 |
| 2 | 0.45 | 0.2860 | 0.6975 | 0.0078 | 0.2887 | 95.10% | 0.989919 |
| 3 | 0.60 | 0.2761 | 0.6782 | 0.0064 | 0.2853 | **95.23%** | **0.990270** |

### 4.3 Held-out benchmark (52,266 samples)

Only two vanilla adapter-only checkpoints have archived evaluations. Cells are **accuracy / AUC**; the mean is the macro-average over the four splits.

| Checkpoint | ID 12,819 | CM 11,249 | CF 12,730 | CD 15,468 | Mean acc / F1 / AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `intern_v2` `ep005.pth` (Stage 2) | 92.42 / 0.9823 | 99.17 / 0.9995 | 81.51 / 0.9418 | 78.11 / 0.8680 | **87.80 / 0.8659 / 0.9479** |
| `intern_v2_dpo` `dpo_best.pth` (DPO) | 89.73 / 0.9788 | 98.91 / 0.9996 | 76.17 / 0.9235 | 76.62 / 0.8675 | 85.36 / 0.8268 / 0.9423 |

Observations specific to the adapter-only configuration:

- The **ungrounded adapter model is the stronger classifier**: DPO costs 2.7 pp on ID and 5.3 pp on CF while gaining 0.0001 CM AUC. DPO targets trace quality, not detection, and this table shows the trade.
- **Cross-model performance is unaffected** by adapter-only training (98.9–99.2% accuracy, AUC ≥ 0.9995), which suggests the vanilla backbone already carries enough generalisable evidence for unseen generators and the adapters mainly sharpen it.
- **Cross-domain is the ceiling** (78.11% accuracy, AUC 0.8680) and is the metric most likely to benefit from a stronger base rather than more adapter capacity.
- The grounded checkpoint's validation AUC (0.9903) is in the same range as the stage-2 checkpoints (0.9891–0.9914), but **no 52k result exists**, so no claim is made that grounding preserves benchmark performance.

### 4.4 Threshold calibration (DPO checkpoint)

| Split | Optimal T* | Base acc | Calibrated acc | Base F1 | Calibrated F1 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| ID | 0.09 | 89.73% | 94.07% | 0.8952 | 0.9439 |
| CM | 0.13 | 98.91% | 99.42% | 0.9890 | 0.9942 |
| CF | 0.06 | 76.17% | 85.76% | 0.6883 | 0.8443 |
| CD | 0.07 | 76.62% | 78.16% | 0.7349 | 0.7977 |
| **Mean** | — | **85.36%** | **89.35%** | **0.8268** | **0.8950** |

Thresholds were tuned and evaluated on the same split, so this is a diagnostic, not a deployable calibration. It does show that the adapter model's scores are systematically shifted high, which matters if the package is deployed with a naive 0.5 threshold.

### 4.5 Qualitative behaviour (grounded adapter package)

The 19 Oct inference log for `intern_v2_ground` covers only two images before the run stopped:

- `test_images/real/161022.jpg` → **REAL, 94.8%** (correct), hotspots "Nose & Cheek region, Eyes & Facial midline", FFT ratio −0.23.
- `test_images/real/160932.png` → **FAKE, 95.1%** (false positive), with a generated explanation citing "unnaturally sharp" hairlines and a "blending boundary" that a human would not confirm.

Two images cannot support an accuracy estimate. They are included because they show both the adapter model's output format and its main failure mode: when the classifier is wrong, the telemetry conditions the LLM on the wrong verdict and the explanation rationalises it. `inference.py --force_verdict` exists to audit exactly this, but no audit results are archived.

### 4.6 Comparison with the LR-fused adapter model

For context only — this is a different foundation, not a controlled ablation:

| Adapter-only model | Validation acc / AUC | 52k mean acc / AUC | Package |
| :--- | :--- | :--- | :--- |
| Vanilla grounded (`intern_v2_ground`) | 95.23% / 0.9903 | not run | 178.24 MiB |
| LR-fused grounded (`intern_fused_ground`) | **96.00% / 0.9936** | 87.57% / 0.9465 | 178.24 MiB |

The fused foundation is 0.77 pp / 0.0034 AUC better on validation, with one run each. Its 52k mean AUC (0.9465) is essentially level with the vanilla stage-2 model (0.9479), while its CF accuracy is 1.15 pp higher. The evidence is consistent with a small LR-fusion benefit but does not establish one.

### 4.7 Evidence map

| Metric | File |
| :--- | :--- |
| Stage-2 validation | `runs/intern_v2/logs/epochs.csv` |
| Stage-2 checkpoint identity | `runs/intern_v2/checkpoints/ep005.pth`, `ep006.pth`, `best.pth` (metadata) |
| 52k benchmark (Stage 2) | `runs/intern_v2/logs/eval_best.log` |
| 52k benchmark + calibration (DPO) | `runs/intern_v2_dpo/logs/eval_dpo_best.log` |
| DPO training | `runs/intern_v2_dpo/logs/dpo_epochs.csv`, `dpo_train.log` |
| Grounded validation | `runs/intern_v2_ground/logs/epochs.csv` |
| Grounded inference | `runs/inference_logs/inference_best_20261019_203120.log` |
| Adapter contents / hashes | `runs/intern_v2_ground/adapters/` |
| Parameter counts | run logs (`train.log`, `launch_stdout.log`), `extraction_report.json` |

---

## 5. Limitations

1. **The grounded adapter-only model has no 52k benchmark.** Its 95.23% / 0.9903 is validation-only; the benchmark table describes the stage-2 and DPO checkpoints.
2. **The benchmark is attributed to `ep005.pth`.** `best.pth` was later overwritten by a second run (epoch 4, AUC 0.991403) and has no evaluation. Any statement of the form "the vanilla best model scores 0.9479" is imprecise.
3. **The DPO package is incomplete.** It ships LoRA adapters without the custom heads, so it cannot classify standalone. No adapter package exists for the stage-2 checkpoint either.
4. **No adapter loader ships in the repo.** `eval.py` and `inference.py` load whole-model checkpoints; the attach pattern is documented and adapter tensors were verified, but end-to-end parity from adapters + heads is unverified.
5. **DPO has no validation log**, so its effect on the classifier is observable only through the 52k benchmark.
6. **Single run per configuration, 4,000-sample validation, no seeds.** Differences under ~1 pp are not statistically established.
7. **The benchmark numbers are from one evaluation per checkpoint**, with no confidence intervals. The mean weights the four splits equally rather than by sample count, so the 11,249-sample CM split counts as much as the 15,468-sample CD split.
8. **Eager attention in all recorded runs.** The requested SDPA implementation is rejected by InternVL; FlashAttention-2 is not installed. Timings are eager-attention timings.
9. **Adapter capacity is a ceiling.** With 4.28% trainable parameters and a frozen backbone, the model cannot learn representations the vanilla foundation does not already support; the CD plateau is consistent with that.
10. **Dead modules in the package.** `domain_router` is unused and `custom_projector` is only a fallback, yet both are shipped and counted.
11. **The Stage-0 LR-fusion for the comparison lineage is external.** No LR data manifest, training log or merge script exists in this repository.

---

## 6. Reproduction

```bash
# Stage 1 + 2 on the vanilla base (adapter-only)
python -u train.py --config config.yaml \
    --model_path ./models/InternVL3-2B \
    --tokenizer_path ./models/InternVL3-2B \
    --batch_size 4 --grad_accum 8 \
    --epochs_cls 3 --epochs_joint 3 \
    --output_dir ./runs/intern_v2

# Stage 3: DPO from the end of the joint curriculum
python -u train_dpo.py --config config.yaml \
    --model_path ./models/InternVL3-2B \
    --checkpoint ./runs/intern_v2/checkpoints/ep006.pth \
    --dpo_epochs 2 --dpo_lr 1e-6 --dpo_beta 0.1 \
    --output_dir ./runs/intern_v2_dpo

# Stage 4: grounded fine-tuning, classifier frozen
python -u train.py --config config_intern_v2_ground.yaml \
    --model_path ./models/InternVL3-2B \
    --tokenizer_path ./models/InternVL3-2B \
    --weights_from ./runs/intern_v2/checkpoints/ep006.pth \
    --epochs_cls 0 --epochs_joint 3 --freeze_cls_head \
    --output_dir ./runs/intern_v2_ground

# Extract the deployable package (adapters + heads)
python extract_adapters.py \
    --checkpoint ./runs/intern_v2_ground/checkpoints/best.pth \
    --output_dir ./runs/intern_v2_ground/adapters \
    --base_model ./models/InternVL3-2B

# Whole-model evaluation (the only path the repo currently supports)
python eval.py --dataset_root ./datasets/hydrafake \
    --json_root ./datasets/hydrafake/jsons \
    --model_path ./models/InternVL3-2B \
    --checkpoint ./runs/intern_v2_ground/checkpoints/best.pth \
    --batch_size 8 --num_workers 8

# Whole-model inference
python inference.py --config config.yaml \
    --model_path ./models/InternVL3-2B \
    --checkpoint ./runs/intern_v2_ground/checkpoints/best.pth \
    --grounded_reasoning --images test_images/real/*
```

---

## 7. Conclusion

Adapter-only training on the vanilla foundation is a practical and well-understood configuration: 93.4 M trainable parameters (4.28%) produce a 178 MiB portable package on top of a frozen 2.18 B backbone, with validation AUC 0.9903 for the latest grounded model and a 52k benchmark mean AUC of 0.9479 for its stage-2 predecessor. The adapters demonstrably carry the task: the same frozen backbone with a small head set reaches 99%+ on cross-model data and 78% on cross-domain data.

The evidence also sets hard boundaries. The grounded adapter model has not been benchmarked on the full test set; the DPO package is missing the heads required to classify; and the repository cannot yet execute an adapter-only evaluation end to end. Those are engineering and evaluation gaps, not architectural ones, and they are the shortest path to making this package production-usable.

---

*Compiled from source code, training logs, evaluation logs, checkpoint metadata and adapter extraction reports. Numbers that could not be traced to an archived artifact are excluded and the attribution caveats are stated in §4.1.*
