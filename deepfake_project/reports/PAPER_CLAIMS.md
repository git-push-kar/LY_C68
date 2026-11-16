# Paper claims and their evidence

Every claim this project can put in a paper, with the number, where it comes from, how it was
verified, and — as importantly — which claims are **not** supported and must not be written.

Status legend:

| Tag | Meaning |
| --- | --- |
| **MEASURED** | produced by a script in this repo, on data in this repo, reproducible from the command given |
| **VERIFIED** | asserted by a test that would fail if the claim broke |
| **BLOCKED** | needs data or hardware this repo does not have |
| **DO NOT CLAIM** | measured, but the honest reading does not support the sentence people want to write |

Measured 2026-11-16 on `feature/yolo-multiface` (commits `b6cc6c5`..`06d9c8c`), RTX A5000, bf16.

---

## 1. The single-image detector (Lineage A, `intern_fused_ground` best.pth, epoch 3)

Source: `RESEARCH_PAPER_REPORT.md` §5.1-5.3, computed from archived `eval_*.log` over the full
52,266-sample held-out set.

Per-split figures with 95 % intervals, extracted by `eval_metrics.py` from the archived log.
Positive class = FAKE; TP/FP/FN/TN are reconstructed from the logged counts and verified against
the logged F1, so precision/recall are derived rather than re-run.

| Split | N | Accuracy [95 % CI] | F1 | Precision [CI] | Recall [CI] | AUC [CI] |
| --- | --- | --- | --- | --- | --- | --- |
| ID (in-domain) | 12,819 | 0.9213 [0.9165, 0.9258] | 0.9219 | 0.9904 [0.9876, 0.9925] | 0.8624 [0.8540, 0.8703] | 0.9881 [0.9863, 0.9899] |
| CM (cross-model) | 11,249 | 0.9888 [0.9867, 0.9906] | 0.9887 | 0.9957 [0.9936, 0.9971] | 0.9819 [0.9780, 0.9850] | 0.9994 [0.9989, 0.9999] |
| CF (cross-forgery) | 12,730 | 0.8266 [0.8200, 0.8331] | 0.7917 | 0.9917 [0.9885, 0.9940] | **0.6588 [0.6471, 0.6704]** | 0.9507 [0.9468, 0.9546] |
| CD (cross-domain) | 15,468 | 0.7660 [0.7593, 0.7726] | 0.7598 | 0.7804 [0.7708, 0.7897] | 0.7403 [0.7304, 0.7500] | 0.8480 [0.8418, 0.8542] |
| Macro mean | — | 0.8757 | 0.8655 | — | — | 0.9465 |
| Validation (epoch 3) | 4,000 | 0.9600 | 0.9483 | — | — | 0.993647 |

**The CF failure is a recall failure, not a precision failure.** Precision 0.9917 but recall
0.6588: when the model says FAKE on a cross-forgery edit it is almost always right, and it misses
a third of them. CD fails the other way (precision 0.7804, recall 0.7403). Those are different
problems with different remedies, and the single accuracy number hides both.

Acc/precision/recall intervals are Wilson 95 %; AUC intervals are Hanley-McNeil approximations
(the log has no per-sample scores). Threshold-calibrated macro mean, grounded vanilla checkpoint:
89.50 % accuracy / 89.62 % F1 — still without an interval, and it is a per-split-optimal-threshold
total, so it is optimistic.

Rules for writing these: they are **log-derived**, not re-runnable from `config.yaml`, whose
metrics block is stale (see report §5.3). Do not describe the CD number as a solved problem —
it is the failure mode of the system.

## 2. Adapter deployment is lossless

Source: `verify_adapter_parity.py`, `test_adapter_loader.py` (21 tests).

| Claim | Number | Status |
| --- | --- | --- |
| Adapter package reproduces the checkpoint | **18/18 identical verdicts, max \|Δp_fake\| = 0.000e+00** (`images=18 verdict mismatches=0 tol=1e-04`, re-run 2026-11-16) | **MEASURED** |
| Package contents | llm_adapter 392 tensors / 73,859,072 params; vision_adapter 192 / 6,291,456; custom_heads 59 / 13,257,034 | **MEASURED** |
| The wrong base is refused | `check_base_model` (pure): different base raises, same base accepted, absent provenance tolerated — plus `_check_base` on a real temp dir, and 9 path-identity tests | **VERIFIED** |
| Dropped / unknown adapter keys are refused | `check_adapter_keys` (pure): dropped key, unknown key, mismatch in both directions, and non-set iterables | **VERIFIED** |
| The real load path uses those checks | wiring tests with a stubbed model and stubbed peft/safetensors: `_attach_one` calls `check_adapter_keys` exactly once with the file keys, the live keys and the subdir; `attach_adapters` delegates to `check_base_model` with the resolved provenance, and skips it when no base is requested | **VERIFIED** |

Safe sentence: "the 93.4 M-parameter adapter package is verdict-identical to the 4.36 GB
checkpoint on the comparison set". Do not say "bit-identical weights" — the claim is about
verdicts and probabilities, which is what was compared.

## 3. Single-face behaviour is preserved exactly — on the passthrough path

Source: `test_passthrough_parity.py` (`RUN_MODEL_PARITY_TESTS=1`), `inference_faces.py --no_detect`.

**Scope, stated first because it is easy to get wrong: not every single-face image takes the
passthrough path.** The router keys on face *area*, not face count. `om.jpg` (lone face at 0.026
of the frame) and `chick.jpg` (0.021) fall below the 0.05 floor and are classified from a context
crop — deliberately, since that is the range where the whole-image verdict is unreliable (§6).

| Claim | Number | Status |
| --- | --- | --- |
| Single-face images **above the area floor** take the whole-image path | 3/3 asserted in the test (`161022.jpg` 0.2047, `310151.png` 0.2446, `omi.jpg` 0.0876) | **VERIFIED** |
| Lone small faces do **not** | `om.jpg` 0.026 and `chick.jpg` 0.021 route `per_face` | **MEASURED** |
| On that path, verdicts and probabilities match `inference.py` | `torch.equal` on probabilities, 3/3 real single-face images, reference forward built by `inference.py`'s own loop | **VERIFIED** |
| The inputs are identical before the model even runs | `pixel_values` and `freq_features` bitwise equal to `build_transform(448)` + `compute_fft_features(224)` of the whole image (always-on layer, synthetic image, runs without weights) | **VERIFIED** |
| The test can tell the paths apart | a `per_face` crop is provably *not* bitwise equal to the passthrough tensor, so a router that cropped or passed everything through could not pass | **VERIFIED** |

This is what lets the paper say "multi-face support is additive": for an image that takes the
passthrough path, the output is bitwise unchanged. It is the strongest guarantee in the repo, and
it is bounded — lone small faces are routed to crops by design, so "single-face images are
unchanged" is **not** the claim.

## 4. Detection and routing

Sources: `face_detect.py`, `test_face_pipeline.py` (29), `test_face_routing.py` (25), FFIW cache
(400 frames).

| Claim | Number | Status |
| --- | --- | --- |
| Finds the visible faces in the sample multi-face photos | 3 boxes in `3-people.jpg` and 3 in `multiple-faces.jpg`, one per visible face (visual check of the overlay; there is no ground-truth count) | **MEASURED** |
| FFIW frames are mostly single-face | 368 × 1 face, 24 × 2, 3 × 0, 5 with ≥3 (1 × 3, 2 × 4, 1 × 5, 1 × 6) — n = 400 | **MEASURED** |
| Face area distribution | 0.0007-0.7107 of frame over all 438 faces; the separate 40-frame detection probe showed a single-face median of 0.38 | **MEASURED** |
| Routing rules are deterministic and tested | passthrough / per_face / no_face, ordering of the passthrough check before `min_face_px` | **VERIFIED** |
| The floor sweep measures the shipped rule | `lone_face_fills_frame` is shared; a parity test checks all three paths | **VERIFIED** |

**DO NOT CLAIM** detector recall on FFIW. This copy has no ground-truth boxes, so
detected-vs-GT at IoU ≥ 0.5 was never computed (**BLOCKED**). The 3/3 sample result is not a
recall estimate.

## 5. Per-face classification and aggregation

Sources: `inference_faces.py`, `test_face_routing.py`, real CLI runs.

| Claim | Number | Status |
| --- | --- | --- |
| All crops go through one forward call | `classify()` stacks items with `_to_batch`; measured 519 ms for 4 crops vs 151 ms for 1. Single-call behaviour is asserted by a test only for the 1-face case, so the multi-crop part is code-level | **MEASURED** |
| Document verdict = `max` over faces by default | `3-people.jpg`: 0.9651/0.1023/0.6972 → FAKE; `multiple-faces.jpg`: 0.0418/0.0471/0.0418 → REAL | **MEASURED** |
| Face disagreement is reported | `3-people.jpg` → `faces_disagree: true` (FAKE/REAL/FAKE); `multiple-faces.jpg` → `false` (all REAL). Computed inline in `evaluate_image`: no dedicated unit test, so this rests on the run outputs | **MEASURED** |
| The classifier is the only decision authority | LLM answer is recorded; a contradiction sets `model_disagreement` + review, never overrides the verdict — `find_disagreements` is unit-tested | **VERIFIED** |
| Whole-image rows carry an integer id | test asserts `id == 0` for all three whole-image routes (passthrough, no_face, no_detect) | **VERIFIED** |
| `reasoned_faces` is emitted without nulls | real run: `omi.jpg` → `reasoned_faces: [0]` | **MEASURED** |

**DO NOT CLAIM** that per-face verdicts are as accurate as whole-image verdicts. On FFIW the
crop path scores *below* the whole-image path (best doc AUC 0.8970 vs 0.9037). The crop path
exists to see faces the model otherwise cannot; it is competitive, not superior.

## 6. Calibration: margin, threshold, passthrough floor

Sources: `face_calibration.py`, `FACE_PIPELINE_CALIBRATION.md`, 400 FFIW frames (200/200,
seed 1234, one frame per video).

| Claim | Number | Status |
| --- | --- | --- |
| Margin 0.30 is **no worse** than tight crops | doc AUC 0.8845 [0.8545, 0.9142] vs 0.8782 [0.8454, 0.9079] (+0.63 pt, **intervals overlap**); acc 0.7925 vs 0.7375 | **MEASURED** |
| The size gate excludes faces that are no harder | < 96 px AUC 0.8626 [0.726, 0.953] (40 faces) vs >= 96 px 0.8926 [0.858, 0.922] (398); flat across 0-48/48-96/96-144/144+ px bins | **MEASURED** |
| Padding mode has no measurable effect | `none` 0.9444 [0.786, 1.000] (15 faces), `reflect` 0.9167 [0.811, 0.988] (31), `stretch` 0.8836 [0.849, 0.914] (392) at margin 0.30 | **MEASURED** |
| Margin 0.45 looks better, for the wrong reason | 0.8970 AUC, but 99 % of faces are already >2× upscaled at margin 0 and 423/438 crops degrade to `stretch` | **MEASURED** |
| `face_threshold` is not a tuned number | sweep flat within 1 SE across 0.42-0.55; Youden optimum moves 0.52 → 0.43 with the margin | **MEASURED** |
| Passthrough floor sits in a stable basin | floors 0.03, 0.05 and 0.06 route all 33 unique local images identically (0.04 is not in the sweep; the rule is monotone, so it cannot differ) | **MEASURED** |
| The floor changes real verdicts when it matters | `chick.jpg` whole-image 0.7367 vs crop 0.0530; `om.jpg` 0.7104 vs 0.0610 | **MEASURED** |

**DO NOT CLAIM** that FFIW validates the multi-face path: it is 256 px and 92 % single-face, so
it routes passthrough. The margin sweep bypasses routing to score crops, which is why the margin
question is answerable at all — say that explicitly.

**DO NOT CLAIM** a calibrated `face_threshold` improvement. The sweep's spread is inside the
error bar (n = 400, 1 SE ≈ 1.5 pt); the value is confirmed, not optimised.

**DO NOT CLAIM** that margin 0.30 is *significantly* better than tight crops, that `min_face_px`
was validated as necessary, or that mirrored padding is harmful. The intervals overlap in all
three comparisons. 0.30 is chosen for non-inferiority plus training-frame agreement; the 96 px
gate is a stated conservative choice about upscaling, and the reflect-padding worry is not
supported by the 31 measured faces. Say *why* the default was chosen, not that the data proved it.

## 7. Latency

Source: `face_timing.py`, `FACE_PIPELINE_CALIBRATION.md` §5. Medians of 15 calls for
detect/classify, mean of 2 generations per face for reasoning, `torch.cuda.synchronize()` around
every call.

| Claim | Number | Status |
| --- | --- | --- |
| Detect | 13.6 ms on a 256² frame to 21.7 ms on 1600² — near-flat in input size (two sizes measured, not a size sweep) | **MEASURED** |
| Classify | 151.4 ms (1 face), 518.9 ms (4 faces) — ~0.13 s/face | **MEASURED** |
| Reason | 14.9-117.9 s per face depending on session (~4× machine-state spread); ~98-99 % of total wall time | **MEASURED** |
| 1-face end to end, `top1` | 15.1-65.8 s | **MEASURED** |
| 4-face end to end | `top1` **25.3-115.1 s** vs `all` **71.4-326.0 s**; the ratio is 2.82-2.84× in both sessions | **MEASURED** |
| Screening without reasoning | 0.17 s (1 face) to 0.54 s (4 faces) by medians | **MEASURED** |
| Lowering `max_new_tokens` does not save time | 512 vs 256: 25.3 vs 24.5 s; 512 vs 192: 18.5 s but the slowest face's answer was truncated to `unknown`; single-face identical at all three caps | **MEASURED** |

**DO NOT CLAIM** these as throughput figures: single machine, unlocked clocks, one 4-face image
(a synthetic montage), and ~1 in 15 classify calls lands 3-4 s outside the median.

**DO NOT CLAIM** that a lower token cap makes reasoning cheaper. The same configuration differed
~4× between two sessions, while the `all`/`top1` **ratio** held at 2.82-2.84×. Quote the ratio and
the range, not one absolute number — and do not use the 27 % `top1` "saving" at cap 192, which is
the slowest generation being cut off before its answer.

## 8. What is still unmeasured (do not paper over)

| Gap | Why it is blocked | What it would take |
| --- | --- | --- |
| Detector recall / per-face accuracy on multi-face ground truth | FFIW here has no GT boxes | annotations, or a licensed full FFIW10K download |
| Reasoning quality (specificity, correctness, grounding) | no rubric or human labels exist | judge/human evaluation over ~50 samples |
| Multi-face evaluation on full-resolution group photos | the eval set is 256 px, 92 % single-face | a labelled high-res multi-face set |
| Latency on other hardware | measured on one A5000 | repeat `face_timing.py` on the target GPU |
| Effect of margin on high-res crops | FFIW's upscaling confounds it | high-res faces where margin 0 already has upscale ≈ 1 |

The honest summary for a reviewer: classification, deployment parity and the passthrough
guarantee (bounded to images that take that path) are measured and testable; the multi-face crop
path is calibrated on a proxy set and its per-face accuracy on true multi-face ground truth is an
open measurement.

---

## 9. Claim → artifact map

| Claim group | Code | Test / harness | Written record |
| --- | --- | --- | --- |
| Single-image metrics | `model.py`, `eval.py` | `test_regression_fixes.py` | `RESEARCH_PAPER_REPORT.md` §5 |
| Adapter parity | `adapter_loader.py` | `verify_adapter_parity.py`, `test_adapter_loader.py` | `INTEGRATION_GUIDE.md` §5.3 |
| Passthrough guarantee | `inference_faces.py` | `test_passthrough_parity.py` | `INTEGRATION_GUIDE.md` §11.1 |
| Detection & cropping | `face_detect.py` | `test_face_pipeline.py` | plan §3 |
| Routing & aggregation | `inference_faces.py` | `test_face_routing.py` | `INTEGRATION_GUIDE.md` §11.1-11.3 |
| Calibration | `face_calibration.py` | `test_calibration.py` | `FACE_PIPELINE_CALIBRATION.md` §2-4 |
| Per-split F1 / PR / CIs | `eval_metrics.py` | derivation verified against logged F1 | this ledger §1 |
| Latency | `face_timing.py` | — (measurement, not an assertion) | `FACE_PIPELINE_CALIBRATION.md` §5 |

## 10. Re-verifying everything

```bash
cd deepfake_project
python test_regression_fixes.py        # 8   single-image invariants
python test_face_pipeline.py           # 29  geometry / cropping
python test_face_routing.py            # 25  routing, aggregation, disagreement
python test_adapter_loader.py          # 21  strict adapter loading (keys, base, wiring)
python test_calibration.py             # 15  AUC, sweeps, floor routing
python test_passthrough_parity.py      # 7   bitwise single-face parity
RUN_MODEL_PARITY_TESTS=1 python test_passthrough_parity.py   # loads the model, 7/7 no skips
python verify_adapter_parity.py        # 18/18 identical verdicts, Δ = 0
python eval_metrics.py --log runs/intern_fused_ground/logs/eval_best.log
python face_calibration.py --adapter_dir ./runs/intern_fused_ground/adapters \
    --n_per_class 200                  # ~15 min; analysis is deterministic from the cache
python face_timing.py --adapter_dir ./runs/intern_fused_ground/adapters \
    --skip_reason --repeats_fast 15    # ~90 s, detect/classify only
```
