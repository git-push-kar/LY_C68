# Adapter integration: reusing the deployed detector in another project

How to take `runs/intern_v2_ground/adapters/` out of this repository and get the **same verdicts**
inside another service. Everything here is verified against the checked-in package; the numbers come
from `extraction_report.json`, from the latency runs in `runs/timing/`, and from
`verify_adapter_parity.py`.

If you need the full architecture/training explanation, read `INTEGRATION_GUIDE.md`; this document is
the portable handoff.

---

## 1. What you are copying

The package is **adapter-only**: it contains the fine-tuned deltas and heads, not the base model and
not the code that runs them.

```
adapters/
├── extraction_report.json              # provenance: source checkpoint, epoch, tensor counts, dtypes
├── llm_adapter/
│   ├── adapter_config.json             # peft config, r=64, alpha=128
│   └── adapter_model.safetensors       #  141 MB
├── vision_adapter/
│   ├── adapter_config.json             # peft config, r=16, alpha=32
│   └── adapter_model.safetensors       #   13 MB
└── custom_heads/
    └── custom_heads.safetensors        #   26 MB
```

| Part | Tensors | Parameters | Precision | Notes |
| :--- | ---: | ---: | :--- | :--- |
| `llm_adapter` | 392 | 73,859,072 | bf16 | LoRA `r=64`, `alpha=128`, targets `q,k,v,o,gate,up,down` |
| `vision_adapter` | 192 | 6,291,456 | bf16 | LoRA `r=16`, `alpha=32`, targets `qkv,proj,fc1,fc2` |
| `custom_heads` | 59 | 13,257,034 | bf16 | `cls_head`, `attn_pool`, `freq_branch`, `custom_projector`, `domain_router` |

**Total 93,407,562 adapter parameters (~180 MB on disk).** Source: `intern_v2_ground/checkpoints/best.pth`,
epoch 3, trained from `runs/intern_v2/checkpoints/ep006.pth` on base `./models/InternVL3-2B`.
`cls_head` was frozen during grounded post-training but is still required — it is the scoring head.

Integrity (first 16 hex of SHA-256, for a quick `sha256sum` diff):

| File | SHA-256 prefix | Size |
| :--- | :--- | ---: |
| `llm_adapter/adapter_model.safetensors` | `ec210956b1273cc5` | 141 MB |
| `vision_adapter/adapter_model.safetensors` | `fbf3b50455f02bf5` | 13 MB |
| `custom_heads/custom_heads.safetensors` | `55c225843f8abd42` | 26 MB |

> The adapters attach to the **unmodified upstream** `InternVL3-2B`. They will not load onto
> `models/internvl3_2b_lr_fused` (Lineage A, comparison only) — `adapter_loader` refuses that
> explicitly rather than silently producing wrong verdicts.

---

## 2. What else has to travel with it

The package has no code, so a target project needs four small modules from this repository:

| File | Why |
| :--- | :--- |
| `model.py` | defines `DeepfakeReasoningModel` (classifier head, attention pooling, FFT branch, grounding prefix, `predict`) |
| `adapter_loader.py` | `load_model()` / `attach_adapters()`, base-model and key-count checks |
| `dataset.py` | only `compute_fft_features(img, target_size=224)` for the classifier's spectral input |
| `inference.py` | `build_transform(image_size)` and the telemetry/parse helpers used by the face pipeline |

Optional, only if you want the multi-face path: `face_detect.py` and `inference_faces.py`.

Runtime: Python 3.10+, `torch==2.6.0` (cu126 build used here), `transformers==4.37.2`,
`peft==0.7.1`, `safetensors`, `timm`, `einops`. **Do not move to transformers 5.x**: the base
model's `trust_remote_code` module targets the 4.x API (`_tied_weights_keys`) and model construction
fails. Full pins are in `requirements.txt`.

---

## 3. Minimal wiring

```python
import sys
sys.path.insert(0, "/opt/detector")          # where model.py / adapter_loader.py / dataset.py / inference.py live

import torch
from PIL import Image
from adapter_loader import load_model
from dataset import compute_fft_features
from inference import build_transform

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model, tokenizer, info = load_model(
    base_model="./models/InternVL3-2B",      # HF snapshot, trust_remote_code=True
    adapter_dir="./adapters",                # <- the copied package
    arch_version="v2",
    attn_implementation="sdpa",
    device=DEVICE,
)
model.eval()
print(info["source"], info["parts"]["custom_heads"]["tensors"])   # adapters 59
```

`load_model` accepts **exactly one** weight source. Passing both (or neither) raises
`AdapterLoadError("pass exactly one of adapter_dir= or checkpoint= ...")`.

### 3.1 Preprocessing contract (the part that silently breaks integrations)

Two tensors must be produced from the **same** PIL image:

```python
transform = build_transform(448)                       # InternViT preprocessing, image_size=448
pv   = transform(image).unsqueeze(0).to(DEVICE)        # (1, 3, 448, 448)
freq = compute_fft_features(image, target_size=224)    # (1, 3, 224, 224) FFT log-magnitude
freq = freq.unsqueeze(0).to(DEVICE)
```

If you omit `freq_features`, the classifier substitutes a **zero** embedding for the third input
stream and the score changes — the model still runs, which is exactly why this is the most common
integration bug. There is no error and no warning.

### 3.2 Scores

```python
with torch.no_grad():
    out = model(pv, freq_features=freq)
probs = torch.softmax(out["cls_logits"].float(), dim=-1)[0]   # [P(real), P(fake)]
verdict = "FAKE" if probs[1] >= 0.5 else "REAL"               # eval.py's rule: p_fake >= 0.5
```

`probs[0]` is REAL, `probs[1]` is fake. The deployed operating point is a face threshold of 0.5 and,
for the document-level pipelines, a crop margin of 0.30 — see §5.

### 3.3 Reasoning trace

```python
res = model.predict(pv, tokenizer, freq_features=freq, max_new_tokens=512)
# {"labels": [0|1], "confidences": [..], "reasoning": ["<fast> ... </fast> ... <answer>fake</answer>"]}
```

`predict` returns `labels` (class indices), `confidences` (max softmax), and `reasoning` (one string
per item, already cut at `<answer>` by the caller in `inference_faces.py`). The trace is conditioned
on the telemetry prefix, not free-form: the prefix is built from the same pooled attention and
spectrum the classifier used. Do not append your own BOS token to a prefix — an injected
`<bos>` mid-sequence derails the fine-tuned continuation into tag soup.

---

## 4. Prove the wiring before you trust it

```bash
python verify_adapter_parity.py \
    --base_model ./models/InternVL3-2B \
    --adapter_dir ./adapters \
    --checkpoint ./runs/intern_v2_ground/checkpoints/best.pth \
    --limit 18 --device cuda
```

Pass condition: identical verdicts on every image **and** `max |p_fake difference| <= 1e-4`. Expected
result for this package: **18/18 identical verdicts, max |Δp_fake| = 0** — re-verified on 2026-11-18
(`runs/timing/adapter_parity_recheck/parity.log`). The script compares a
model built from the whole checkpoint against one built from the adapter package, feeding byte-identical
inputs, so any difference is attributable to weight loading alone.

Run it once after copying, and again after any dependency upgrade: a partially applied adapter still
loads and still returns confident verdicts.

---

## 5. Multi-face path (optional but usually what you want)

```bash
python inference_faces.py \
    --adapter_dir ./adapters \
    --images photo.jpg group.jpg \
    --detector_weights ./models/yolov8n-face.pt \
    --min_face_px 96 --margin 0.30 --face_threshold 0.50 \
    --reason top1 --max_new_tokens 512 \
    --json_out out.json
```

Routing (in `inference_faces.py`; each route is reported as `path`):

| Route | When | Behaviour |
| :--- | :--- | :--- |
| `passthrough` | one face ≥ 5 % of the frame | classify the whole image as one item |
| `per_face` | more than one usable face | crop each face with `--margin`, classify, aggregate |
| `no_face` / `no_detect` | detector finds nothing usable | document verdict `UNKNOWN`, faces skipped with a reason |

Document verdict = `max` over face `p_fake` by default (`--aggregate max|mean|vote`). Per-face results
carry `id, kind, verdict, p_fake, confidence, box, crop, telemetry, reasoning`.

**Do not ship the calibrated T\* thresholds** (0.06–0.22) from the paper: they were fitted and scored
on the same split. 0.5 is the validated operating point that the benchmark table reports.

---

## 6. Operational envelope (measured, one RTX A5000)

| Document | Faces | Detect | Classify all | Reason 1 face | Reason all | Total (all) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| single face, 1644×1534 | 1 | 22.4 ms | 178 ms | 15.5 s | 15.5 s | 15.7 s |
| group photo, 4 faces | 4 | 23.1 ms | 509 ms | 22.6 s | 73.6 s | 74.2 s |
| FFIW frame, 4 faces | 4 | 14.3 ms | 262 ms | 20.0 s | 39.3 s | 39.6 s |

Detection and classification are negligible; autoregressive reasoning dominates and scales with the
number of faces you reason. A four-split evaluation at batch 8 used ~23.4 GB of VRAM; single-image
inference was not separately profiled, so size your deployment from the `--max_faces` you allow
rather than from that figure.

**Token budget:** 512 ships. At 256 the answers are still correct and it is ~6 % faster on the top
face (21.2 s vs 22.6 s) but only ~1 % faster when all four faces are reasoned; at **192 the trace
starts answering `unknown`** on two of four faces (measured), which is why 512 is the default and why
cutting the cap is not a free optimisation.

---

## 7. Failure modes

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| `AdapterLoadError: ... base model ...` | adapters were extracted from `./models/InternVL3-2B`, you pointed at the fused or another snapshot | pass the matching `base_model`; never force it |
| `pass exactly one of adapter_dir= or checkpoint=` | both or neither given | choose one weight source |
| `KeyError`/shape error building the model | `arch_version="v1"` or a different head layout | `arch_version="v2"` |
| Loads but scores drift by 0.1–0.4 | `freq_features` omitted or computed at the wrong size | `compute_fft_features(img, target_size=224)` |
| Verdicts flip after an upgrade | LoRA/key mismatch introduced silently | rerun §4; key counts must be 392 / 192 / 59 |
| Crash on model load after upgrading transformers | base `trust_remote_code` targets 4.x | pin `transformers==4.37.2` |
| Trace is tag soup | a BOS/EOS was injected into the telemetry prefix | build the prompt with `add_special_tokens=False` |
| `RuntimeError: expected ... bf16` | manual dtype casting of the classifier input | cast to `model.cls_head.net[0].weight.dtype` as `adapter_loader`/`model.forward` do |

---

## 8. What not to bring along

- **`models/internvl3_2b_lr_fused`** — Lineage A, +1.04 pp macro but its Stage-0 merge is not
  reproducible from this repository. The shipped package targets the upstream base.
- **DPO checkpoints** (`intern_v2_dpo`, `intern_fused_dpo`) — trained and rejected: trace-only
  objective, classifier/telemetry heads frozen, preference data carries no telemetry.
- **Per-split calibrated thresholds** — post-hoc, fitted on the evaluation split (§5).
- **`config.yaml` training values** — irrelevant at inference; the runtime contract is
  `image_size=448`, `max_new_tokens=512`, `arch_version="v2"`, threshold 0.5.

---

## 9. One-screen checklist

1. `sha256sum` the three `.safetensors` against §1.
2. Copy `adapters/` plus `model.py`, `adapter_loader.py`, `dataset.py`, `inference.py`.
3. Pin `torch 2.6.0` / `transformers 4.37.2` / `peft 0.7.1`.
4. `load_model(base_model=..., adapter_dir=..., arch_version="v2")`.
5. Preprocess with `build_transform(448)` **and** `compute_fft_features(..., 224)`.
6. Score with `softmax(out["cls_logits"])[1] >= 0.5`.
7. Run `verify_adapter_parity.py` → expect 18/18, max |Δ| = 0.
