# Face pipeline calibration (M5)

Measured 2026-11-16 on `feature/yolo-multiface`. Produced by `face_calibration.py`;
raw numbers in `runs/calibration/ffiw_report.json` (untracked).

This replaces the hypotheses in `YOLO_MULTIFACE_PIPELINE_PLAN.md` §5.1 and §7 with
measurements. It also records two things the plan assumed and that turned out not to hold.

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

| margin | faces | face AUC | doc AUC | doc acc | mean upscale | share >2x | pad mode mix (none / reflect / stretch) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.00 | 438 | 0.8811 | 0.8782 | 0.7375 | 4.08 | 99 % | 349 / 51 / 38 |
| 0.15 | 438 | 0.8833 | 0.8787 | 0.7825 | 3.26 | 27 % | 40 / 282 / 116 |
| **0.30** | 438 | **0.8870** | **0.8845** | **0.7925** | 2.88 | 11 % | 15 / 31 / 392 |
| 0.45 | 438 | 0.8981 | 0.8970 | 0.8025 | 2.69 | 8 % | 12 / 3 / 423 |

Reference point, the same frames through the whole image (what passthrough uses):
**AUC 0.9037**, acc@0.50 0.8125, acc@0.42 0.8175.

### Decision: keep `margin = 0.30`

The plan's ship rule was "0.30 if within 1 pt of the tight-crop baseline". Measured:
**+0.63 pt doc AUC, +0.59 pt face AUC, +5.5 pt doc accuracy** over margin 0.00. Passes.

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
   model's `Resize((448, 448))` distorts it. Winning by 1.25 pt while 97 % of crops are
   aspect-distorted is not a result to ship on.

On full-resolution photos neither effect applies: a 0.30 margin already yields an upscale
of ~1.0, which is why 0.30 stays the default and the sweep above is reported as evidence
that padding does not *hurt*, not as a reason to add more of it.

---

## 3. `face_threshold`

Document-level sweep, margin 0.30 (the shipped margin):

| threshold | acc | F1 | TPR | TNR |
| --- | --- | --- | --- | --- |
| 0.42 | 0.7950 | 0.7950 | 0.795 | 0.795 |
| 0.45 | 0.7925 | 0.7920 | 0.790 | 0.795 |
| 0.50 | 0.7925 | 0.7877 | 0.770 | 0.815 |
| 0.55 | 0.8000 | 0.7938 | 0.770 | 0.830 |

Best Youden is 0.52 (acc 0.8000) at margin 0.30, and 0.43 (acc 0.8100) at margin 0.45.
Per-face, the best point is 0.64 at margin 0.30 and 0.52 at 0.45.

### Decision: keep `face_threshold = 0.50`

The sweep is **flat within noise**. Across 0.42-0.55 the accuracy spans 0.75 pt on n = 400,
where one standard error is ~1.5 pt; the Youden optimum jumps from 0.52 to 0.43 when only the
margin changes, which is the signature of noise rather than a real operating point.

Two candidate changes were tested and rejected:

- **0.42** (the value `config.yaml` quotes from the CF split): doc acc 0.7950 vs 0.7925 at
  0.50 (+0.25 pt), F1 +0.73 pt, at the cost of TNR −2 pt. Real but inside the error bar, and
  it would put per-face verdicts out of step with `inference.py`'s argmax rule for no
  measurable gain.
- **0.55**: highest TNR (0.830) and the best Youden at margin 0.30, but TPR drops to 0.770.
  For a review-escalating tool, missing fakes is the expensive error.

0.50 is retained because it is also the class boundary the validated single-image path uses;
the sweep supports that choice rather than merely tolerating it. Anyone who wants the
high-specificity point can pass `--face_threshold 0.55`.

---

## 4. `passthrough_min_face_frac`

FFIW contributes **nothing** here: the floor-sweep on 400 frames is identical from 0.02 to
0.20 (AUC 0.9035, acc 0.8125, 92 % passthrough) and only moves at 0.30 (AUC 0.9011, 88 %
passthrough). Its faces simply do not live in the contested range.

The evidence is the 33 unique local images (deduplicated by content — `test_images/real/`
repeats several top-level files byte-for-byte). Routing at each candidate floor, versus 0.05:

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
call them real:

| image | lone face frac | whole-image p_fake | crop p_fake (margin 0.30) |
| --- | --- | --- | --- |
| `chick.jpg` | 0.021 | 0.7367 | 0.0530 |
| `om.jpg` | 0.026 | 0.7104 | 0.0610 |

That is the mechanism the floor exists to prevent, and why it must stay below the portrait
range (0.073-0.088) rather than being raised to "be safe".

---

## 5. Limitations

- **No ground-truth face boxes** in this copy of FFIW, so the plan's §8.4 detector recall
  (detected vs GT boxes, IoU >= 0.5) **was not measured**. Only document-level numbers exist.
- The FFIW subset is 256 px, one frame per video, 400 frames. Its doc-level AUC (0.90) is
  not comparable to the paper's CD figures in `RESEARCH_PAPER_REPORT.md`, which are computed
  over the full split at 448 with different framing.
- On FFIW the per-face crop path scores **below** the whole-image path (best crop doc AUC
  0.897 at 0.45 vs 0.9037 whole-image). The crop path's job is multi-face images the model
  cannot otherwise see; this result says it is competitive, not that it is better.
- `face_threshold` was calibrated on document-level `max` aggregation. Per-face and
  document numbers differ (best per-face threshold 0.64 at margin 0.30), and the flat sweep
  means neither is a precise number.
- Everything here is threshold-and-margin calibration. Per-face verdicts on padded crops
  remain out-of-distribution relative to the training set, as the plan states.

---

## 6. Reproducing

```bash
python face_calibration.py --adapter_dir ./runs/intern_fused_ground/adapters \
    --n_per_class 200 --out runs/calibration/ffiw_report.json     # ~15 min, A5000
python face_calibration.py --analyze_only --out runs/calibration/ffiw_report.json
python test_calibration.py                                        # analysis unit tests
```

Frames are cached to `runs/calibration/ffiw_cache.jsonl` as they are scored, so an
interrupted run resumes instead of restarting.
