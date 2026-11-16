# YOLO Multi-Face Pipeline — Implementation Plan

**Goal:** accept images containing one *or several* faces, detect every face with YOLO,
crop each face with a small ring of surrounding background, and run each crop through the
deployed **adapter** package (`llm_adapter` + `vision_adapter` + `custom_heads`) instead of
the 4.36 GB whole-model checkpoint.

**Status:** plan only — no pipeline code written yet.
**Branch base:** `master` (already fast-forwarded to merged `major`, at `5c9f955`).
**Work branch:** `feature/yolo-multiface` off `master`.

---

## 1. Why this is not just "run inference.py N times"

`inference.py` today assumes **one face fills one image**:

| Current behavior | Consequence for a multi-face photo |
| --- | --- |
| `T.Resize((448, 448))` then `CenterCrop` on the whole image | A 4-person group photo is squashed, and 3 of 4 faces are lost |
| `compute_fft_features(img, target_size=224)` on the whole image | Spectral evidence is averaged over the entire scene, not over a face |
| `extract_forensic_telemetry` → `regions_from_topk` maps a 32×32 patch grid | The grid is aligned to the 448 center crop; on a group photo the "hotspots" name the wrong person |
| One verdict per image | No way to say *which* face is fake |
| `--checkpoint` only (`best.pth`) or `--raw` | The extracted adapter package at `runs/intern_fused_ground/adapters/` cannot be used at all |

Also relevant: `test_images/3-people.jpg` is **8495×5663** (~48 MP) and `multiple-faces.jpg` is
1448×1086. Both already exist in the repo, but nothing currently detects faces in either.

So the pipeline needs three new seams: **(a) detection**, **(b) context crop**, **(c) adapter
weight loading**, plus **(d) a documented aggregation rule** for the document verdict.

---

## 2. Target data flow

```
image (PIL, RGB)
      │
      ├─ FaceDetector.detect()            YOLO face model, letterbox 640, conf ≥ 0.25, NMS 0.5
      │        └─ boxes [(x1,y1,x2,y2), score]  → filter min_face_px → area sort → top-K (8)
      │
      ├─ 0 faces ──────────────► fallback: full image, unchanged path, flag faces_detected=0
      ├─ 1 face, fills frame ──► PASSTHROUGH: whole image, unchanged path (§5.1)
      ├─ 1 face, small in frame► still goes to a context crop (probably a missed face)
      └─ ≥2 faces ─────────────► multi-crop path
                                       │
                                       ▼
                     crop_with_context(image, box, margin=0.30) → square 448-ready crop
                                       │
              ┌────────────────────────┴─────────────────────────┐
              ▼                                                  ▼
   pixel_values (B,3,448,448)                        freq_features (B,3,224,224)
   T.Resize + CenterCrop + Normalize                 compute_fft_features on the SAME crop
              │                                                  │
              └────────────────► model(pv, freq_features) ────────┘
                                       │  cls_logits → per-face probability
                                       ▼
                         telemetry per face (attn_topk → region names, FFT ratio)
                                       │
                                       ▼
                    reasoning for selected face(s) only (default: top-1 risk)
                                       │
                                       ▼
                     aggregate_faces() → document verdict + per-face JSON
```

---

## 3. Crop with "a little background" — the exact rule

The detector box is tight on the face. Feeding a tight crop is wrong twice over: it deletes the
jaw/neck/hairline blending seams (`Peripheral / Ear & Hair blending seams` is a named region in
`model.REGION_NAMES`), and it changes the FFT statistics relative to training crops.

**Step 1 — pad the box by a margin on all sides**

```
w, h   = x2 - x1, y2 - y1
x1' = x1 - margin * w      y1' = y1 - margin * h
x2' = x2 + margin * w      y2' = y2 + margin * h        margin default 0.30
```

**Step 2 — square it** (the model resizes to 448×448, so a 2:1 crop is distorted, not letterboxed)

```
side  = max(w', h')        centre = box centre
square = (cx - side/2, cy - side/2, cx + side/2, cy + side/2)
```

**Step 3 — clamp to image bounds**

- If the square lies inside the image → crop it.
- If it overflows the border by **≤ 10 % of `side`** → `ImageOps.expand(..., fill=reflect)`, acceptable
  synthetic edge.
- If it overflows more (face at the frame edge) → fall back to the **clamped rectangle** and let
  `Resize((448,448))` stretch it. Mild aspect distortion beats inventing pixels that the FFT
  branch will read as texture, and the model never saw reflected skin during training.

**Step 4 — minimum size gate**

`min_face_px` = 96 px on the shorter box side, measured **in original pixels**. Below that, the
crop is an upscale to 448 and the resampling creates its own high-frequency spectra (a false
"generative" signature). Such faces are reported as `skipped: face_too_small`, never silently
upscaled into a verdict.

Record in the output: `margin`, `pad_mode`, `upscale_factor = 448 / min(crop_w, crop_h)`, so any
face with `upscale_factor > 2.0` is visibly flagged as low-evidence.

---

## 4. Adapter loading (the "passes through the adapter" requirement)

`INTEGRATION_GUIDE.md` §5.3 documents the manual adapter copy; this plan promotes that snippet
into a real module so it is testable and shared by both CLIs.

New `adapter_loader.py`:

```python
def load_backbone(base_model: str, attn_implementation: str = "sdpa") -> DeepfakeReasoningModel
def attach_adapters(model, adapter_dir: str) -> dict   # returns {"llm": n, "vision": n, "heads": n}
def load_detector(*, adapter_dir=None, checkpoint=None, base_model=..., device=...) -> (model, tokenizer)
```

Rules:

- `--adapter_dir` and `--checkpoint` are mutually exclusive; exactly one is required.
- Adapter attach copies tensors into the live `lora_A.default.weight` / `lora_B.default.weight`
  parameters of `model.backbone.language_model` and `model.backbone.vision_model`, then
  `model.load_state_dict(custom_heads, strict=False)`.
- **Hard assertion:** every adapter key must find a live parameter. Today the guide silently
  `next(..., None)`s and skips misses — a silent miss means an unadapted model with no error.
  The module must raise on a mismatch instead.
- `custom_heads` must load all 59 tensors (13,257,034 params); assert the loaded count equals the
  count in `extraction_report.json`.
- Base must be `./models/internvl3_2b_lr_fused` for these adapters — assert against
  `provenance.base_model` in `extraction_report.json` and fail loudly on a mismatch.

**Parity gate (blocking):** on the single-face test set, adapter-loaded verdicts must match
`best.pth` verdicts 100 %. This is **not** covered by existing evidence: the extractor's own
functional check only proves the tensors attach and compare equal, and
`RESEARCH_PAPER_REPORT_VANILLA_ADAPTER_ONLY.md` documents a *different* lineage (adapter-only
training on the vanilla `./models/InternVL3-2B` base), not extraction parity for
`runs/intern_fused_ground/adapters` (base `internvl3_2b_lr_fused`). Run the parity test before
building on the adapter path.

---

## 5. Per-face inference and document aggregation

### 5.1 Routing contract (a single face keeps today's behaviour)

The model, its weights and its preprocessing are **unchanged** — no retraining, no crop-specific
conditioning, no new tokens, no architecture change. Detection is routing only: it decides *which
image* the existing model sees. The model never knows whether it was handed a full photo or a
crop, and it classifies and reasons about exactly one crop at a time.

| Detected faces | What the model receives | Preprocessing |
| --- | --- | --- |
| exactly 1, box ≥ `passthrough_min_face_frac` of the frame | **the whole image, untouched** | `build_transform` + `compute_fft_features` on the full PIL image — the exact code path `inference.py` runs today |
| exactly 1, but small in frame | that face's context crop | §3 crop rules |
| ≥ 2 | one context crop per face, each classified and reasoned independently | §3 crop rules |
| 0 | the whole image, flagged `faces_detected: 0` | unchanged path, no confidence claim |

The single-face passthrough is the default and must byte-match today's output on portrait
inputs; M3's gate is that `inference_faces.py` and `inference.py` produce identical verdicts and
confidences on every single-face image in `test_images/`.

**Why the area guard exists:** "exactly one detected face" is not the same as "single-face
photo". On an 8495-px scene, a detector returning one box usually means it missed the others —
passing that whole image through would issue a document-level REAL from a face occupying a few
percent of the frame. The `passthrough_min_face_frac` floor preserves passthrough for genuine
portraits, which is what the model was validated on, and refuses it exactly where it would be
wrong.

The floor is **0.05 of frame area**, measured rather than guessed (2026-11-16): a real portrait
(`omi.jpg`, 1644x1534) puts its lone face at **0.088**, while faces in a group or wide scene land
at **0.021–0.029** (`3-people.jpg` 0.020–0.029, `chick.jpg` 0.021). 0.05 separates the two with
margin on both sides. The first guess of 0.10 wrongly refused that portrait, which is why the
number is now tied to measurements.

M5 widened that to a **measured basin**: floors from 0.03 to 0.06 route all 33 unique local
images identically, with failures bracketing it (`chick.jpg` 0.021 and `om.jpg` 0.026 below;
`4.jpg` 0.073, `6.jpg` 0.080, `omi.jpg` 0.088 above). FFIW cannot test this range at all — its
faces sit at 0.27–0.54 — which is itself a finding. See `FACE_PIPELINE_CALIBRATION.md` §4.

**Consequence to state, not hide:** passthrough verdicts and per-face verdicts are not directly
comparable. One is whole-scene evidence (FFT over the frame, hotspots over the frame), the other
is crop-local evidence. That is inherent in keeping today's behaviour for single-face images, and
is why §7 calibrates a separate `face_threshold`.

**Classifier pass:** all face crops of one image in a single batch (cap 8), bf16 autocast, no
grad. Cheap — no LLM involved.

**Reasoning pass:** `model.predict()` loops over the batch internally and calls
`language_model.generate()` once per item, so reasoning cost is linear in face count. Default
`--reason_faces top1`: reason only about the highest-risk face. `all` is available for forensics
but must be opt-in (a 6-face photo ≈ 6× generation latency).

**Aggregation** (`--aggregate`, default `max`):

| Mode | Document verdict | Use |
| --- | --- | --- |
| `max` (default) | FAKE if `max(p_fake)` over faces ≥ threshold | A single swapped face is the threat; worst-case is the safe answer |
| `mean` | FAKE if mean `p_fake` ≥ threshold | Scene-level "is this media authentic" |
| `any_fake` | same as `max` but reports the offending face ids | Review queues |
| `per_face` | no document verdict, per-face only | Annotation / debugging |

`max` is the default for the same reason `inference.py` treats the classifier as the only decision
authority: one compromised identity is enough. `requires_human_review` is raised when faces
disagree (at least one FAKE and one REAL) or any face is below `0.70` confidence.

Threshold stays **separate** from `eval_threshold` (`face_threshold`, default `0.42` — the CF
calibration point in `config.yaml`), because margin-padded crops shift the distribution (§7).

---

## 6. Files, config, dependencies

### 6.1 New / touched files

| File | Change | Purpose |
| --- | --- | --- |
| `face_detect.py` | **new** | `FaceDetector` (ultralytics wrapper) + pure helpers `expand_box`, `square_box`, `clamp_and_pad`, `crop_with_context`, `filter_faces` |
| `adapter_loader.py` | **new** | §4 loading, assertions, single entry point for both CLIs |
| `inference_faces.py` | **new** | CLI: detect → crop → per-face classify/reason → aggregate → log + JSON |
| `inference.py` | **untouched** | Validated single-image behaviour must not move; it stays the reference and is the passthrough target (§5.1) |
| `config.yaml` | +`face_pipeline:` block | detector/crop/aggregation defaults |
| `requirements.txt` | +`ultralytics` pin, +`opencv-python-headless` | detection |
| `fetch_detector_weights.py` | **new** | Pinned, SHA-256-verified download of the detector weights (they stay out of git) |
| `run_faces.bat` | **new** (optional) | Launcher in the style of `run_intern_v2_ground.bat` |
| `test_face_pipeline.py` | **new** | `unittest` suite matching `test_regression_fixes.py` conventions |

### 6.2 Config block

```yaml
face_pipeline:
  # fetched + digest-verified by fetch_detector_weights.py (see 6.3); no upstream
  # URL serves a file literally named yolov8n-face.pt, this is the local name
  detector_weights: "./models/yolov8n-face.pt"
  detector_imgsz: 640
  detector_conf: 0.25
  detector_iou: 0.50
  max_detect_pixels: 4096        # downscale 8495-px inputs before detection
  min_face_px: 96                # shorter box side, original pixels
  max_faces: 8
  passthrough_single_face: true    # 1 face filling the frame -> today's whole-image path
  passthrough_min_face_frac: 0.05  # face box area / image area floor for passthrough
                                   # (portraits ~0.088, group/wide-scene faces 0.021-0.029)
  margin: 0.30
  face_threshold: 0.42
  aggregate: "max"
  reason_faces: "top1"
  save_crops: null               # "./runs/face_crops"
  save_annotated: null           # box overlay for eyeballing detections
```

### 6.3 Dependency discipline (important)

`ultralytics` declares `torch`, `torchvision`, `numpy`, `opencv-python`, `pandas`, `psutil`,
`py-cpuinfo`, `requests` and will happily **upgrade torch 2.6 / numpy** if installed casually —
which would break the pinned stack (`transformers==4.37.2` is load-bearing, per `requirements.txt`
and `INTEGRATION_GUIDE.md` §2.2).

Install procedure:

```bash
# from deepfake_project/, venv activated
pip install "ultralytics==8.3.*" --no-deps
pip install opencv-python-headless psutil py-cpuinfo requests   # only if missing
pip check
python -c "import torch, numpy, ultralytics; print(torch.__version__, numpy.__version__, ultralytics.__version__)"
```

`opencv-python-headless` (not `opencv-python`) avoids a GUI dependency on the inference box; it
supplies the same `cv2` module, so `pip check` keeps one benign line about the `opencv-python`
distribution name. Commit the resolved versions to `requirements.txt` only after the
`test_regression_fixes.py` suite still passes.

### 6.4 Detector weights (pinned, not committed)

`models/` is gitignored, and no upstream URL serves a file literally named
`yolov8n-face.pt` — so the weights are fetched by name with their identity checked:

```bash
python fetch_detector_weights.py            # download if absent, verify either way
python fetch_detector_weights.py --check    # verify only, no network
```

| Field | Value |
| --- | --- |
| Source | `arnabdhar/YOLOv8-Face-Detection` → `model.pt` (Hugging Face, AGPL-3.0) |
| Local name | `models/yolov8n-face.pt` |
| Size | 6,247,065 bytes |
| SHA-256 | `37396ac6a9601ab9f5177e4231b09d81cf6f65a7f22db99ec3b36ab63f674e71` |

A mismatch aborts the fetch instead of leaving a wrong checkpoint in place: which faces get
cropped changes every downstream verdict, so the detector is part of the measurement, not a
detail. If upstream changes the artifact, update the pin deliberately.

**License note:** ultralytics is AGPL-3.0. If this pipeline ships inside a closed-source product,
either accept AGPL, buy the commercial licence, or swap the detector behind the same
`FaceDetector` interface for InsightFace SCRFD / OpenCV YuNet (both face-specialised, permissive).
The interface in §6.4 exists exactly so that swap is a one-file change.

### 6.4 Interface (so alternatives stay swappable)

```python
@dataclass
class FaceBox:
    x1: float; y1: float; x2: float; y2: float
    score: float

class FaceDetector:
    def detect(self, image: Image.Image) -> list[FaceBox]: ...
```

`crop_with_context(image, box, margin) -> (crop: Image.Image, meta: dict)` is pure PIL/numpy and
unit-testable with no torch and no weights.

---

## 7. Calibration — the step that decides whether this works

Adding background changes two inputs at once: the ViT sees non-face pixels, and the FFT ratio
drops (background is smoother ⇒ lower HF/LF ⇒ looks more "generative"). The model was trained on
HydraFake face crops, so **margin is a hyper-parameter that must be measured, not guessed.**

Experiment matrix (per-face AUC/Acc on the evaluation set from §8):

| margin | expected effect | decision |
| --- | --- | --- |
| 0.00 | tight crop, max face detail, no blend seams | baseline |
| 0.15 | slight context | |
| 0.30 | **default candidate** — jaw/hairline seams visible | ship if AUC within 1 pt of baseline |
| 0.45 | lots of background, FFT drift | only if 0.30 hurts |

Deliverable: a margin table in this file (or a short `FACE_PIPELINE_CALIBRATION.md`) with the
chosen margin, `face_threshold`, and the document-level AUC, written **after** measuring. Until
then the defaults above are hypotheses, not results.

**Measured (M5, 2026-11-16) — full tables in `FACE_PIPELINE_CALIBRATION.md`.** 400 FFIW frames,
margin 0.00 / 0.15 / 0.30 / 0.45, document-level AUC 0.8782 / 0.8787 / **0.8845** / 0.8970:
0.30 beats the tight baseline by +0.63 pt and ships; 0.45's extra gain is confounded (99 % of
FFIW faces are already >2x upscaled at margin 0, and 97 % of 0.45 crops degrade to `stretch`),
so it is reported, not adopted. `face_threshold` stays 0.50: the sweep is flat within one
standard error across 0.42–0.55 and the Youden optimum moves with the margin (0.52 at 0.30,
0.43 at 0.45). The plan's earlier hypothesis of 0.42 gained +0.25 pt document accuracy — inside
the noise — and would have diverged from `inference.py`'s argmax for nothing.

**Latency (plan §8.5), measured:** reasoning is ~99 % of wall time — 65.8 s on a 1-face image
against 0.6 s for detect+classify, and 115 s (`top1`) vs 326 s (`all`) on a 4-face image, a 2.8x
saving. That is what makes `top1` a default rather than an assumption. Table in
`FACE_PIPELINE_CALIBRATION.md` §5; harness is `face_timing.py`.

**Correction to §8.4:** FFIW as stored here is 256x256 and 92 % single-face, so it routes
passthrough and **does not exercise the multi-face crop path**; it also ships without GT face
boxes, so detector recall (IoU ≥ 0.5) is still unmeasured. The margin sweep is therefore scored
per-face directly, bypassing routing, and the crop path's numbers should be read as
"competitive with, not better than" the whole-image path (best crop doc AUC 0.897 vs 0.9037).

---

## 8. Evaluation plan

1. **Parity (blocking, cheap):** adapter path vs `best.pth` on ~100 single-face images from
   `test_images/real/` + `test_images/*.jpg` → identical verdict strings. Reuses the existing
   `runs/inference_logs/` format for comparison.
2. **Crop-geometry unit tests** (no GPU): box expansion, squaring, clamping at each border,
   reflection threshold, `min_face_px` rejection, `max_faces` truncation order, aggregation modes.
3. **Multi-face detection sanity:** `test_images/3-people.jpg` (expect 3 faces),
   `test_images/multiple-faces.jpg`, plus a portrait with exactly 1 face (must take the
   **passthrough** path and reproduce `inference.py` output exactly), plus a wide scene holding
   one small face (must be refused passthrough and routed to a crop, `path: "per_face"`), plus
   `test_images/chick.jpg`-style non-face imagery (expect 0 faces → fallback flag, no crash).
4. **Quantitative multi-face set:** `FFIW` is already mapped to the `cd` split in
   `dataset.py` and is a multi-face benchmark. Derive a small frame set (a few hundred frames,
   doc-level label = FAKE if any GT face is fake) and report document-level AUC/Acc plus
   per-face recall of the detector (detected boxes vs GT face boxes, IoU ≥ 0.5).
5. **Latency:** per-image wall time split into detect / classify / reason, 1-face vs 4-face,
   so §5's `top1` default is justified with numbers.
6. **Regression:** `python test_regression_fixes.py` and one `inference.py` run must be
   unchanged (no edits land in `model.py`, `dataset.py` or `inference.py`).

---

## 9. Milestones

| # | Milestone | Output | Gate |
| --- | --- | --- | --- |
| M0 | Branch + deps | `feature/yolo-multiface`, pinned `ultralytics`, `pip check` clean | existing tests pass |
| M1 | Detector + crop module | `face_detect.py`, `test_face_pipeline.py` geometry tests | unit tests green, boxes visually sane on the 2 multi-face images |
| M2 | Adapter loader | `adapter_loader.py`, strict key assertions | parity check §8.1 = 100 % |
| M3 | Multi-face inference | `inference_faces.py` (detect → batch classify → per-face telemetry) | 4 images produce per-face verdicts |
| M4 | Reasoning routing + aggregation | `--reason_faces`, `--aggregate`, JSON schema | aggregation unit tests green |
| M5 | Calibration + eval set | `FACE_PIPELINE_CALIBRATION.md`, `face_calibration.py`, FFIW 400-frame numbers — **done 2026-11-16** | margin 0.30 ships (+0.63 pt over tight baseline); `face_threshold` 0.50 and frac 0.05 confirmed, not changed; latency measured (115 s top1 vs 326 s all on 4 faces) |
| M6 | Docs + launcher | `INTEGRATION_GUIDE.md` §11 multi-face section, `run_faces.bat` — **done 2026-11-16** | launcher ran both sample images end to end; the §11.4 snippet reproduced the CLI's per-face probabilities exactly (0.965119 / 0.102305 / 0.697197) |

Each milestone is independently mergeable; M0–M2 unblock everything else and carry no risk to
the validated single-image path.

---

## 10. Output contract (per image)

```json
{
  "image": "3-people.jpg",
  "path": "per_face",
  "faces_detected": 3,
  "faces_evaluated": 3,
  "faces_skipped": [],
  "document_verdict": "FAKE",
  "aggregate": "max",
  "faces": [
    {
      "id": 0,
      "box": [311, 1204, 802, 1695],
      "box_score": 0.94,
      "crop_meta": {"margin": 0.30, "pad_mode": "none", "upscale": 1.02},
      "verdict": "REAL",
      "p_fake": 0.03,
      "telemetry": {"hotspots": "Nose & Cheek region", "spectral": "ratio=0.71 ..."},
      "reasoning": null
    },
    {
      "id": 1, "verdict": "FAKE", "p_fake": 0.87,
      "telemetry": {"hotspots": "Peripheral / Ear & Hair blending seams", "spectral": "ratio=0.49 ..."},
      "reasoning": {"quick_take": "...", "analysis": "...", "conclusion": "..."}
    }
  ],
  "flags": {"faces_disagree": true, "requires_human_review": true}
}
```

`3-people.jpg` at 8495 px is the worst case for detection resolution: cap the detector input at
`max_detect_pixels` (4096) and keep `min_face_px` measured in **original** coordinates so the
downscale for detection never causes a small face to be silently accepted.

---

## 11. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| `ultralytics` upgrades torch/numpy and breaks `transformers==4.37.2` | install `--no-deps`, `pip check`, pin after verification (§6.3) |
| AGPL-3.0 blocks commercial shipping | detector behind `FaceDetector`; YuNet/SCRFD swap in one file |
| Background margin drags FFT ratio down → false FAKE | margin sweep + separate `face_threshold` (§7) |
| Tiny faces upscaled to 448 create synthetic spectra | `min_face_px = 96`, `upscale` reported, faces below gate skipped not judged |
| Detector misses profile / occluded faces | report `faces_detected`; document-level `REAL` on a single detected face is not evidence the photo is clean — expose the count in the API |
| Reasoning cost explodes with face count | `reason_faces: top1` default, classifier stays fully batched |
| Adapter key silently unmatched → unadapted model | strict assertion in `adapter_loader.py` (§4) |
| Multi-face crops were never in the training distribution | stated explicitly: per-face verdicts on padded crops are **out-of-distribution** until §7/§8 numbers exist; don't quote benchmark figures from `RESEARCH_PAPER_REPORT.md` for this path |

---

## 12. Non-goals

- Re-training or fine-tuning on multi-face / padded crops (only calibration is planned).
- Tracking faces across video frames (FFIW is used for evaluation only, not temporal logic).
- Face recognition / identity clustering of detected faces.
- Changing `model.py`, `dataset.py`, `eval.py` or the validated single-image `inference.py` path.
