# Face pipeline calibration (M5)

Measured 2026-11-16 on `feature/yolo-multiface`, **re-measured on the deployed grounded vanilla
model** (`runs/intern_v2_ground/checkpoints/best.pth`, epoch 3) on 2026-11-16 after the deployment
switch. Produced by `face_calibration.py`; raw numbers in
`runs/calibration/ffiw_report_vanilla.json` and `runs/timing/vanilla_timing*.log` (untracked).

> **Why a separate cache.** `face_calibration.py` caches per-face scores to
> `runs/calibration/ffiw_cache.jsonl`. That cache holds **Lineage A (fused) probabilities**, so
> re-running with the vanilla adapters without a new cache file would silently analyse the wrong
> model's scores, and it would look like a result. The deployed model's numbers below come from
> `ffiw_cache_vanilla.jsonl`, written by explicit `--cache`. Keep them apart.

This replaces the hypotheses in `YOLO_MULTIFACE_PIPELINE_PLAN.md` §5.1 and §7 with measurements.
Where a finding is **model-specific** it now says which checkpoint it belongs to — one of them
reversed when the deployed model changed, and that is recorded rather than smoothed over.

---

## 1. What was evaluated, and the surprise

| | |
| --- | --- |
| Set | FFIW, `datasets/hydrafake/test/cd/FFIW`, the `cd` split generator |
| Frames | **400** (200 `0_real` / 200 `1_fake`), one frame per video directory, seed 1234 |
| Frame size | **256 x 256** for every frame |
| Faces per frame | 0: 3, 1: **368**, 2: 24, 3: 1, 4: 2, 5: 1, 6: 1 |
| Faces scored | 438 |
| Face area / frame | 0.0007 - 0.7107 (single-face median ~0.38) |
| Face short side | 6 - 216 px (median 134) |
| Ground-truth boxes | **not available** in this copy |

**The surprise: FFIW as stored here is a single-face, 256 px set.** 368 of 400 frames hold
exactly one face, and a lone face at 0.38 of the frame is far above the 0.05 passthrough
floor. So running the shipped pipeline over FFIW would put **~92 % of frames down the
passthrough path** and almost never exercise the context crop — it would have measured the
whole-image path and called it a multi-face evaluation.

`face_calibration.py` therefore classifies, for every frame, the whole image **and** one
crop per detected face at every margin, regardless of which route the frame would take.
That is what makes the margin comparison possible at all.

---

## 2. Margin sweep

Document-level probabilities use the pipeline's `max` aggregation. Threshold columns at 0.50.
Deployed vanilla adapters, margin 0.30 is the shipped value.

| margin | faces | face AUC | doc AUC [95 % CI] | doc acc | mean upscale | share >2x | pad mode mix (none / reflect / stretch) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.00 | 438 | 0.8840 | 0.8812 [0.8471, 0.9104] | 0.7575 | 4.08 | 99 % | 349 / 51 / 38 |
| 0.15 | 438 | 0.8883 | 0.8853 [0.8518, 0.9162] | 0.8050 | 3.26 | 27 % | 40 / 282 / 116 |
| **0.30** | 438 | **0.8975** | **0.8944 [0.8633, 0.9207]** | **0.8050** | 2.88 | 11 % | 15 / 31 / 392 |
| 0.45 | 438 | 0.9037 | 0.8984 [0.8693, 0.9247] | 0.8150 | 2.69 | 8 % | 12 / 3 / 423 |

Intervals are 1,000-sample percentile bootstraps over frames. **They all overlap**, so the
margin ranking is a point-estimate ordering, not a statistically resolved one (n = 400). The
+1.32 pt at 0.30 over tight crops must be written as "no worse, and framed like the training
crops", not as "significantly better". The choice is defensible; the *difference* is not
demonstrated. (For reference this margin ordering is the same on Lineage A, one to two points
lower throughout: 0.8782 / 0.8787 / 0.8845 / 0.8970.)

Reference point, the same frames through the whole image (what passthrough uses):
**AUC 0.9090**, acc@0.50 0.8225.

### Decision: keep `margin = 0.30`

The plan's ship rule was "0.30 if within 1 pt of the tight-crop baseline". Measured:
**+1.32 pt doc AUC, +1.35 pt face AUC, +4.75 pt doc accuracy** over margin 0.00. Passes, and
with more headroom than the fused model had (+0.63 / +0.59 / +5.5).

0.45 scores higher, and this is deliberately **not** adopted, for two reasons visible in the
table:

1. **The gain is confounded with upscaling.** 99 % of these faces are already upscaled more
   than 2x at margin 0.00 (median 2.42x, worst 56x — from the 6 px faces in the group
   frames). Mean upscale falls monotonically as margin rises (4.08 -> 2.69), exactly as AUC
   rises. On 256 px frames, a bigger crop mostly means *less resampling blur*, and the FFT
   branch reads high-frequency detail. This sweep cannot separate "background context helps"
   from "less upscaling helps".
2. **At 0.45 nearly every crop is `stretch` padding** (423 / 438): the padded square no
   longer fits in a 256 px frame, so it degrades to a clamped non-square rectangle and the
   model's `Resize((448, 448))` distorts it. Winning by 0.40 pt doc AUC (+1.0 pt accuracy)
   over the shipped margin while 97 % of crops are aspect-distorted is not a result to ship on.

On full-resolution photos neither effect applies: a 0.30 margin already yields an upscale
of ~1.0, which is why 0.30 stays the default and the sweep above is reported as evidence
that padding does not *hurt*, not as a reason to add more of it.

### 2.1 Where the gate and the padding actually matter

Face AUC at the **shipped** margin 0.30, split two ways (`face_calibration.auc_by`):

| Size gate | Faces | Positives | AUC | 95 % CI |
| --- | --- | --- | --- | --- |
| below gate, < 96 px | 40 | 26 | **0.7596** | [0.525, 0.959] |
| judged, >= 96 px | 398 | 200 | **0.9065** | [0.877, 0.931] |

| Pad mode (margin 0.30) | Faces | Positives | AUC | 95 % CI |
| --- | --- | --- | --- | --- |
| `none` | 15 | 12 | 0.9722 | [0.846, 1.000] |
| `reflect` | 31 | 16 | 0.8875 | [0.754, 0.983] |
| `stretch` | 392 | 198 | 0.8974 | [0.863, 0.927] |

Finer size bins, same margin: 0-48 px **0.6307** [0.318, 0.975] (30 faces), 48-96 px 0.9167
[0.667, 1.000] (10), 96-144 px 0.9148 [0.882, 0.947] (299), 144+ px 0.8790 [0.808, 0.939] (99).

**What this does and does not establish — and a reversal:**

- **On the deployed model the gate is validated as doing real work.** Faces it excludes score
  **0.7596 against 0.9065**, a 14.7 pt gap, and the finest bin is near chance: faces under
  48 px score **0.6307** AUC across 30 faces. The intervals overlap only at the extreme upper
  edge of the small-face CI (0.959); the point estimates do not.
- **This is model-specific, and the earlier version of this document said the opposite.** On
  Lineage A the same split was 0.8626 below the gate against 0.8926 above it — flat — and the
  0-48 px bin was 0.8523. The fused model tolerated tiny upscaled faces; the vanilla model does
  not. Anyone quoting "the gate was not validated" is quoting the comparison checkpoint. The
  96 px default is retained for both, and for the deployed model it is now supported by data
  as well as by the upscaling argument (a 6 px face stretched 56x into 448).
- **Padding mode still shows no measurable effect**: `reflect` 0.8875 and `stretch` 0.8974
  against `none` 0.9722, all overlapping. The worry that mirrored skin would manufacture FFT
  "evidence" is **not supported** here — though `none`/`reflect` are only 15 and 31 faces, so
  this is weak evidence rather than a clearance.

### 2.2 How much surrounding context? (full-resolution sensitivity)

The FFIW margin sweep above **cannot** answer "how wide should the crop be": those frames are
256 px, 99 % of faces are upscaled at margin 0, and at 0.45 some 423/438 crops degrade to
`stretch`. There the sweep measures padding. The question only becomes clean on full-resolution
faces, where every margin is *downscaled* into 448 and the only variable is how much ears, hair
and background sit inside the square.

Two unlabelled local images, every margin, one forward pass each
(`runs/samples/context_sweep.py`, re-run 2026-11-16 with the vanilla defaults; it reproduces the
CLI's p_fake exactly at margin 0.30, which is how the harness was checked). Contact sheets:
`runs/samples/context/{image}_face{n}_margins.png`.

`3-people.jpg`, 8495x5663, faces 823-960 px wide (p_fake per face):

| margin | crop px | face 0 | face 1 | face 2 | verdicts |
| --- | --- | --- | --- | --- | --- |
| 0.00 | 1170-1473 | 0.1468 | 0.0764 | 0.0770 | all REAL |
| 0.10 | 1404-1767 | 0.2396 | 0.0953 | 0.1375 | all REAL |
| 0.20 | 1638-2062 | 0.4220 | 0.1251 | 0.2848 | all REAL |
| **0.30** | 1872-2356 | **0.9230** | **0.2122** | **0.3714** | FAKE / REAL / REAL |
| 0.45 | 2224-2798 | 0.9408 | 0.4771 | 0.3127 | FAKE / REAL / REAL |
| 0.60 | 2574-3240 | 0.9386 | 0.8832 | 0.1755 | FAKE / FAKE / REAL |
| 0.80 | 3043-3829 | 0.9367 | 0.7576 | 0.6945 | all FAKE |
| 1.00 | 3511-3916 | 0.9533 | 0.9329 | 0.9446 | all FAKE |

`multiple-faces.jpg`, 1448x1086, faces 244-318 px wide (pipeline says REAL at the shipped margin):

| margin | face 0 | face 1 | face 2 | verdicts |
| --- | --- | --- | --- | --- |
| 0.00 | 0.0474 | 0.0478 | 0.7074 | REAL / REAL / FAKE |
| **0.30** | **0.0630** | **0.0630** | **0.0567** | all REAL |
| 0.45 | 0.0830 | 0.0742 | 0.0656 | all REAL |
| 0.60 | 0.0947 | 0.0924 | 0.0724 | all REAL |
| 0.80 | 0.1097 | 0.0947 | 0.0988 | all REAL |
| 1.00 | 0.1244 | 0.1093 | 0.1172 | all REAL |

What this establishes, and what it does not:

- **Crop geometry is a first-order factor, not a rounding detail.** The same face spans
  0.077-0.945 in p_fake across the sweep (spreads 0.81 / 0.86 / 0.87 on the three `3-people`
  faces). Any claim of the form "the model scores this face at X" is really "at X *given this
  crop*", and must say so.
- **The direction supports more context for seam evidence.** On `3-people` the tight box (no
  ears, no hair, no background) reads REAL for every face and p_fake rises monotonically as the
  surrounding is added — the ear/hair/background boundary is exactly where blending seams live,
  and margin 0 crops them away. If background-merge discrepancies are the target, 0.30 is closer
  to right than 0.00, and 0.60 already changes one of the three verdicts.
- **The shipped margin is a stable operating point on this pair.** 0.30 and 0.45 agree on all
  six verdicts; the fused model flipped four of six between those two margins. Vanilla is
  *less* sensitive in the range that matters.
- **Vanilla does not manufacture positives at extreme margins.** On Lineage A, `multiple-faces`
  jumped from ~0.04 to 0.51 / 0.83 / 0.67 at margin 0.80 — a REAL document turned FAKE by small
  faces plus mirrored padding. The deployed model stays at 0.09-0.12 through margin 1.00 on the
  same image and same boxes. The earlier "extreme margins manufacture positives" warning is a
  Lineage A observation; it does not reproduce here, so do not state it as a property of the
  shipped pipeline.
- **This is a sensitivity demo, not a selection criterion** (n = 2 unlabelled images, and the
  `3-people` montage has no ground truth beyond "the pipeline was pointed at it"). Choosing which
  box size is *most accurate* needs labelled full-resolution faces: that is the OpenForensics
  work-package in `PAPER_CLAIMS.md` §8, which ships per-face boxes, per-face forgery labels and
  full-resolution images, so margins 0.00/0.15/0.30/0.45 can be scored for AUC rather than
  inspected.
- **Cheap hedge until then:** classify two crops per face (tight 0.00 + context 0.45) and aggregate
  with `max`. It costs one extra batched forward (~0.12 s/face), removes the dependence on which
  single margin happened to be chosen, and would have caught all three `3-people` faces at both
  ends of the sweep. Not adopted in this change — it needs the labelled evaluation first.

---

## 3. `face_threshold`

Document-level sweep, margin 0.30 (the shipped margin), vanilla adapters:

| threshold | acc | F1 | TPR | TNR |
| --- | --- | --- | --- | --- |
| 0.42 | **0.8100** | 0.7946 | 0.735 | 0.885 |
| 0.45 | 0.8075 | 0.7913 | 0.730 | 0.885 |
| 0.50 | 0.8075 | 0.7902 | 0.725 | 0.890 |
| 0.55 | 0.8025 | 0.7836 | 0.715 | 0.890 |
| 0.60 | 0.8000 | 0.7802 | 0.710 | 0.890 |

The best document accuracy anywhere in 0.30-0.75 is **0.8200 at T = 0.34**; per-face, the best
point is **T = 0.39**. At margin 0.45 the document optimum moves again. Overall document accuracy
spans 0.7950-0.8200 across the whole 0.30-0.75 range.

### Decision: keep `face_threshold = 0.50`

The sweep is **flat within noise**. The full range across 0.30-0.75 is 2.5 pt on n = 400, where
one standard error is ~1.5 pt; the four tested points sit inside 1 pt of each other, and the
optimum jumps between 0.34, 0.39 (per-face) and 0.42 depending on aggregation — the signature of
noise rather than a real operating point.

Two candidate changes were tested and rejected:

- **0.42** (the value `config.yaml` quotes from the CF split): doc acc 0.8100 vs 0.8075 at
  0.50 (+0.25 pt), F1 +0.44 pt, at the cost of TNR −0.5 pt. Real but well inside the error bar,
  and it would put per-face verdicts out of step with `inference.py`'s argmax rule for no
  measurable gain.
- **0.55** and above: TNR stays 0.890 while TPR falls to 0.715 (0.55) and 0.710 (0.60).
  For a review-escalating tool, missing fakes is the expensive error.

0.50 is retained because it is also the class boundary the validated single-image path uses;
the sweep supports that choice rather than merely tolerating it. Anyone who wants the
high-specificity point can pass `--face_threshold 0.55`.

---

## 4. `passthrough_min_face_frac`

FFIW contributes **nothing** here: the floor-sweep on 400 frames is identical from 0.02 to
0.20 (AUC 0.90855, acc 0.8225, 92 % passthrough) and only moves at 0.30 (AUC 0.9067, acc 0.820,
88 % passthrough). Its faces simply do not live in the contested range.

The evidence is the 33 unique local images (deduplicated by content — `test_images/real/`
repeats several top-level files byte-for-byte). Routing depends only on the detector and the
detected box areas, not on the adapters, so this table is model-independent and unchanged:

| floor | local images whose route changes | which |
| --- | --- | --- |
| 0.02 | 2 / 33 | `chick.jpg`, `om.jpg` become passthrough |
| 0.03 - 0.06 | **0 / 33** | — |
| 0.08 | 3 / 33 | `4.jpg` -> no_face, `6.jpg`/`9.jpg` -> per_face |
| 0.09 - 0.10 | 4 / 33 | adds `omi.jpg` -> per_face |
| 0.15 | 5 / 33 | adds `2.jpg` -> no_face |

### Decision: keep `passthrough_min_face_frac = 0.05`

0.05 sits in the middle of a measured basin: **floors from 0.03 to 0.06 route every local
image identically**, and each edge is bracketed by a real image (below: `chick.jpg` 0.021,
`om.jpg` 0.026 — group/wide-scene faces; above: `4.jpg` 0.073, `6.jpg` 0.080, `omi.jpg` 0.088).
Choosing a value inside a flat basin, with failures measured on both sides, is the strongest
claim the available data supports.

The floor is not cosmetic. On the local images the two regimes disagree sharply wherever a
face is small in the frame — the whole-image path calls the wide scenes fake while the crops
call them real, and the deployed model still does this:

| image | lone face frac | whole-image p_fake | crop p_fake (margin 0.30) |
| --- | --- | --- | --- |
| `chick.jpg` | 0.021 | 0.8832 | 0.0526 |
| `om.jpg` | 0.026 | 0.7874 | 0.0564 |

That is the mechanism the floor exists to prevent, and why it must stay below the portrait
range (0.073-0.088) rather than being raised to "be safe".

---

## 5. Stage latency and the `top1` default

Measured with `face_timing.py` on the RTX A5000, bf16, defaults (greedy, grounded,
`max_new_tokens=512`), **deployed vanilla adapters** (2026-11-16, `runs/timing/vanilla_timing*.log`).
Detect and classify are **medians of 15 calls** after a discarded warmup; reasoning is **one
generation per face** for this run. Every timed call is bracketed by `torch.cuda.synchronize()`.

| Image | Faces evaluated | Path | Detect | Classify | Reason / face | Total, `top1` | Total, `all` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `test_images/omi.jpg` (1644x1534) | 1 | passthrough | 22 ms | 167 ms | 15.7 s | **15.9 s** | 15.9 s |
| 4-photo montage (1600x1600) | 4 | per_face | 23 ms | 516 ms | 16.0 - 20.9 s | **21.5 s** | 72.6 s |
| FFIW `test_00000929/053.png` | 2 of 4 (2 below 96 px) | per_face | 15 ms | 268 ms | 19.7, 20.6 s | **20.9 s** | 40.7 s |

The 4-face image is a montage of four real photographs (tiled 2x2), because FFIW's only rated
4-face frame has just two faces above the 96 px gate — it is measured separately in row 3, where
it shows exactly what the gate is doing: two faces judged, two reported `face_too_small`.

**What the numbers say:**

1. **Reasoning is ~99 % of wall time at every face count**: 15.7 s of generation vs 0.17 s of
   classification for one face, and 16.0-20.9 s vs 0.52 s for four. Classification is not the
   cost centre and batching it harder would optimise nothing.
2. **`top1` saves 3.4x on a 4-face image**: 21.5 s instead of 72.6 s, i.e. 51 s per image for
   reasoning about three faces that are not the highest-risk one. This is the whole justification
   for the default; `all` is a deliberate, expensive opt-in.
3. **Per-call reasoning cost varies 16-21 s in this session** because generation length varies
   and greedy decoding stops at `</answer>`. Do not read that as the model being faster than
   Lineage A: §5.2 explains why absolute seconds move between sessions by far more than this.
4. **Classifier cost per face is ~0.12-0.13 s** (167 ms for 1, 516 ms for 4), and detection is
   15-23 ms regardless of frame size (the detector letterboxes to `imgsz=640`).
5. **Screening without reasoning is 0.20 s (1 face) to 0.55 s (4 faces)** by the same medians.
   That is the mode for volumes; reserve reasoning for escalated cases.

### 5.1 Token cap: measured, and a negative result

Generation is 99 % of the cost, so the obvious lever is the 512-token cap. Three caps were run
back to back under identical conditions (one generation per face, `--max_new_tokens`). **This
probe was run on Lineage A**; the deployed model has only been timed at the 512 default, so the
seconds below are the fused model's and are quoted for the *conclusion*, not as vanilla latency:

| Cap | 1-face reason | 4-face, reason per face | 4-face total `top1` | 4-face total `all` | Answers (4-face) |
| --- | --- | --- | --- | --- | --- |
| **512** (default) | 14.9 s | 24.8 / 15.8 / 13.9 / 16.3 s | 25.3 s | 71.4 s | fake, real, real, real |
| 256 | 15.4 s | 24.0 / 16.2 / 14.5 / 16.6 s | 24.5 s | 71.8 s | **unknown**, real, real, real |
| 192 | 15.3 s | 18.0 / 16.2 / 14.3 / 16.5 s | 18.5 s | 65.5 s | **unknown**, real, real, real |

**The cap is not what bounds generation.** Lowering 512 -> 256 saves nothing measurable (25.3 ->
24.5 s) because greedy decoding already stops at `</answer>` well inside 512 tokens. Lowering to
192 appears to save 27 % on `top1` (25.3 -> 18.5 s) but that is an artefact: the only face that
changed was the slowest one, whose trace was **cut off before its answer tag**, so the audit
signal became `unknown` instead of `fake`. `all` moves 71.4 -> 65.5 s (8 %) for the same reason.
On a single face all three caps are identical (14.9 / 15.4 / 15.3 s).

So the honest conclusion is the opposite of the hypothesis this section started with: keep
`max_new_tokens = 512`. A lower cap buys no speed on normal generations and silently discards the
answer on the longest ones — the ones most likely to matter. If reasoning cost must come down,
the lever is *how many faces get reasoned about* (`top1`, already the default), not how long each
one may run. **Re-run this probe on the vanilla package before quoting any cap number as the
deployed model's** — the conclusion is about decoding behaviour rather than weights, but it has
not been measured there.

### 5.2 Variance between sessions

Lineage A measured at 512 tokens produced **14.9-117.9 s per face** across sessions — a ~4x
difference on identical inputs, same GPU, same code. The deployed model's session above landed at
**15.7-20.9 s per face**, i.e. in the fast end of that range; one session is not evidence that
vanilla is faster at reasoning. The likely cause of the spread is machine state (GPU
clocks/contention from other work on the box); it was not instrumented, so it is not explained,
only recorded. What stayed stable across lineages is the **ratio**: `all` / `top1` = 2.84 and
2.82 for Lineage A, **3.38** for the deployed model at 4 faces. Quote the ratio and the range,
not a single absolute figure.

Measurement caveats: single machine, CUDA clocks not locked, and roughly one call in fifteen
lands 3-4 s outside the median on classify (allocator growth), which is why medians are quoted
rather than means. The montage is synthetic in composition though not in pixels. The single-image
path (§6 of `INTEGRATION_GUIDE.md`) adds nothing here: passthrough is the 1-face row.

---

## 6. Limitations

- **No ground-truth face boxes** in this copy of FFIW, so the plan's §8.4 detector recall
  (detected vs GT boxes, IoU >= 0.5) **was not measured**. Only document-level numbers exist.
- The FFIW subset is 256 px, one frame per video, 400 frames. Its doc-level AUC (0.89) is
  not comparable to the paper's CD figures in `RESEARCH_PAPER_REPORT.md`, which are computed
  over the full split at 448 with different framing.
- On FFIW the per-face crop path scores **below** the whole-image path for the deployed model too
  (best crop doc AUC 0.8984 at 0.45 vs 0.9090 whole-image). The crop path's job is multi-face
  images the model cannot otherwise see; this result says it is competitive, not that it is better.
- `face_threshold` was calibrated on document-level `max` aggregation. Per-face and
  document numbers differ (best per-face threshold 0.39 at margin 0.30), and the flat sweep
  means neither is a precise number.
- **Two findings in this document are checkpoint-specific and were reversed by the deployment
  switch**: the 96 px gate (validated for vanilla, flat for fused) and extreme-margin false
  positives (absent for vanilla, present for fused). Do not move a sentence between the two
  lineages without re-measuring it; §2.1 and §2.2 say which is which.
- Everything here is threshold-and-margin calibration. Per-face verdicts on padded crops
  remain out-of-distribution relative to the training set, as the plan states.

---

## 7. Reproducing

```bash
# deployed vanilla model -- note the separate cache file; the default cache holds Lineage A scores
python face_calibration.py --adapter_dir ./runs/intern_v2_ground/adapters \
    --cache runs/calibration/ffiw_cache_vanilla.jsonl \
    --n_per_class 200 --out runs/calibration/ffiw_report_vanilla.json    # ~3 min, A5000
python face_calibration.py --analyze_only \
    --cache runs/calibration/ffiw_cache_vanilla.jsonl \
    --out runs/calibration/ffiw_report_vanilla.json
python test_calibration.py                                        # analysis unit tests

# stage latency (section 5)
python face_timing.py --adapter_dir ./runs/intern_v2_ground/adapters \
    --skip_reason --repeats_fast 15 --out runs/timing/face_timing.json
python face_timing.py --adapter_dir ./runs/intern_v2_ground/adapters \
    --repeats_reason 1 --out runs/timing/face_timing_reason.json

# full-resolution context sweep (section 2.2) -- writes contact sheets to runs/samples/context/
python runs/samples/context_sweep.py test_images/3-people.jpg test_images/multiple-faces.jpg

# Lineage A comparison: same commands with ./models/internvl3_2b_lr_fused,
#   runs/intern_fused_ground/adapters and runs/calibration/ffiw_cache.jsonl
```

Frames are cached as they are scored, so an interrupted run resumes instead of restarting. Because
the cache is keyed by frame and not by model, **always pass `--cache` explicitly** when switching
checkpoints.
