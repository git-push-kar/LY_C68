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

## 1. The single-image detector (Lineage A, `intern_fused_ground` best.pth)

Source: `runs/timing/epoch3/eval_best.log` — **the shipped epoch-3 weights**
(`{'epoch': 3, 'stage': 'cls_only', 'best_auc': 0.993646875}`), re-run 2026-11-16 22:52-23:43 on
the A5000 over the full 52,266-sample held-out set (`eval.py`, unchanged). 51 minutes wall clock.

**These are the weights that ship.** Adapters, `verify_adapter_parity.py`, the parity tests and
`inference_faces.py` all load this same checkpoint, so every number in this ledger now describes
one model. An earlier table in this section was built from `runs/intern_fused_ground/logs/eval_best.log`,
which records `epoch=1` — a checkpoint that has since been overwritten by epoch 3. That log is
retained for audit only; **do not quote its numbers** (ID 0.9213, CD 0.7660). Moving to epoch 3
changed ID by +2.0 pt accuracy and left CM/CF/CD within 0.4 pt, so the earlier table was not
wrong about the *shape* of the results, only about which weights produced them.

Independently cross-checked: `eval_by_generator.py` (a separate harness that keeps per-sample
scores) reproduces the CD row exactly — 7,935 / 7,533 predictions, 0.7698 acc, 0.8533 AUC.

**Still stale elsewhere:** `RESEARCH_PAPER_REPORT.md` §5 carries the epoch-1 numbers and must be
regenerated before submission (out of scope for this change).

Per-split figures with 95 % intervals, extracted by `eval_metrics.py` from that log.
Positive class = FAKE; TP/FP/FN/TN are reconstructed from the logged counts and verified against
the logged F1, so precision/recall are derived rather than re-run.

| Split | N | Accuracy [95 % CI] | F1 | Precision [CI] | Recall [CI] | AUC [CI] |
| --- | --- | --- | --- | --- | --- | --- |
| ID (in-domain) | 12,819 | 0.9410 [0.9368, 0.9450] | 0.9428 | 0.9872 [0.9841, 0.9897] | 0.9023 [0.8951, 0.9091] | 0.9891 [0.9874, 0.9908] |
| CM (cross-model) | 11,249 | 0.9889 [0.9868, 0.9907] | 0.9888 | 0.9962 [0.9942, 0.9975] | 0.9815 [0.9776, 0.9847] | 0.9993 [0.9988, 0.9998] |
| CF (cross-forgery) | 12,730 | 0.8256 [0.8189, 0.8321] | 0.7897 | 0.9947 [0.9921, 0.9965] | **0.6547 [0.6430, 0.6663]** | 0.9482 [0.9442, 0.9522] |
| CD (cross-domain) | 15,468 | 0.7698 [0.7631, 0.7764] | 0.7668 | 0.7770 [0.7674, 0.7862] | 0.7569 [0.7472, 0.7663] | 0.8533 [0.8473, 0.8593] |
| Macro mean | — | 0.8813 | 0.8720 | — | — | 0.9475 |
| Validation (same checkpoint) | 4,000 | 0.9600 [0.9535, 0.9656] | 0.9483 | — | — | 0.993647 [0.9912, 0.9961] |

The validation row is the checkpoint's own `best_auc` (0.993647), i.e. the score that selected
these weights; its interval is derivable from the counts (3,840 of 4,000 correct, 2,000/2,000
classes) and is Hanley-McNeil like the rest of the AUC column.

**The CF failure is a recall failure, not a precision failure.** Precision 0.9947 but recall
0.6547: when the model says FAKE on a cross-forgery edit it is almost always right, and it misses
a third of them. CD is the opposite shape (precision 0.7770, recall 0.7569) — there it is wrong
both ways. Those are different problems with different remedies, and the single accuracy number
hides both.

**ID improved with the extra epochs (0.9213 → 0.9410) and CF/CD did not.** That is worth one
sentence in the paper: more in-domain training tightened the operating point rather than
improving transfer, which is the expected result if the remaining error is a domain shift and not
an optimisation one.

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

### 1.1 Where CD loses its points (per-generator, epoch-3 weights)

`eval.py` logs only four split totals, so 0.7660 on CD was never attributable. `eval_by_generator.py`
keeps per-sample probabilities and groups them by the generator segment already present in the
image path. Split total on the shipped weights: **0.7698 acc / 0.8533 AUC** (N = 15,468).

| Generator | N | GT R/F | Pred R/F | Acc [95 % CI] | P | R | F1 | AUC [CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FFIW | 6,832 | 3417/3415 | 3624/3208 | 0.8055 [0.7959, 0.8147] | 0.825 | 0.775 | 0.7993 | 0.8936 [0.8858, 0.9014] |
| deepfacelab | 3,094 | 1547/1547 | 459/2635 | 0.6412 [0.6242, 0.6580] | **0.583** | **0.993** | 0.7346 | 0.8667 [0.8538, 0.8796] |
| dreamina | 952 | 476/476 | 498/454 | 0.9685 [0.9554, 0.9778] | 0.991 | 0.945 | 0.9677 | 0.9939 [0.9890, 0.9989] |
| gpt4o | 630 | 315/315 | 407/223 | 0.8476 [0.8175, 0.8736] | 0.991 | **0.702** | 0.8216 | 0.9474 [0.9293, 0.9655] |
| hailuo | 1,000 | 500/500 | 635/365 | 0.8550 [0.8318, 0.8755] | 0.986 | **0.720** | 0.8324 | 0.9875 [0.9805, 0.9944] |
| InfiniteYou-CD | 2,960 | 1480/1480 | 2312/648 | 0.7128 [0.6963, 0.7289] | 0.986 | **0.432** | 0.6006 | 0.9067 [0.8955, 0.9178] |
| **SPLIT TOTAL** | 15,468 | 7735/7733 | 7935/7533 | **0.7698 [0.7631, 0.7764]** | 0.777 | 0.757 | 0.7668 | **0.8533 [0.8472, 0.8593]** |

Mean p_fake separates the two failure modes cleanly:

| Generator | mean p_fake on REAL | mean p_fake on FAKE |
| --- | --- | --- |
| FFIW | 0.2172 | 0.7400 |
| deepfacelab | **0.6713** | 0.9410 |
| dreamina | 0.0736 | 0.8518 |
| gpt4o | 0.0697 | 0.6527 |
| hailuo | 0.0701 | 0.6683 |
| InfiniteYou-CD | 0.0669 | **0.4512** |

**The CD "drop" is the average of two opposite failures, not one weakness.**

1. **deepfacelab over-triggers on REAL faces.** Mean p_fake on genuine images is 0.6713, so the
   model calls 2,635 of 3,094 faces fake: recall 0.993 with precision 0.583. Its reals carry the
   visual signature the classifier associates with face-swap output, and score > 0.5 almost by
   default. This is the only generator where the false-positive rate is the problem.
2. **AI-generated faces under-trigger.** InfiniteYou-CD recall 0.432 (mean p_fake on fakes 0.4512),
   gpt4o 0.702, hailuo 0.720 — precision stays at 0.986-0.991 throughout, so these are silent misses,
   not confusions. Whole-image diffusion/GAN output at this resolution simply does not look fake to
   this classifier as often as face-swap output does.
3. **dreamina (0.9685) and FFIW (0.8055) are the two that behave.** FFIW is the weakest AUC of the
   six (0.8936) despite being the largest group — the in-domain-ish generator is not automatically
   the easy one, which is worth a sentence in the paper.

**DO NOT CLAIM** a single cause for the CD gap, and do not mix in 0.7660 from the retired epoch-1
log: this table and §1's CD row are the same run on the shipped weights (0.7698).

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
| Per-generator CD | `eval_by_generator.py` | reproduces the split total before splitting | this ledger §1.1 |
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
python eval_metrics.py --log runs/timing/epoch3/eval_best.log     # §1 (shipped epoch-3 weights)
python eval_by_generator.py --checkpoint runs/intern_fused_ground/checkpoints/best.pth \
    --split cd --out runs/timing/eval_by_generator_cd.json      # ~15 min, CD only
python face_calibration.py --adapter_dir ./runs/intern_fused_ground/adapters \
    --n_per_class 200                  # ~15 min; analysis is deterministic from the cache
python face_timing.py --adapter_dir ./runs/intern_fused_ground/adapters \
    --skip_reason --repeats_fast 15    # ~90 s, detect/classify only
```
