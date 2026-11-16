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

## 1. The single-image detector (`intern_v2_ground` best.pth — **the deployed model**)

Source: `runs/intern_v2_ground/logs/eval_best.log` — epoch 3, `best_auc` 0.990270, run 31 Oct 2026
00:23–01:14 over the full 52,266-sample held-out set (`eval.py`, batch 8, bf16, 51 min 20 s on the
A5000), with `--calibrate` in the same pass. The log was produced for the checkpoint file that is
still on disk (`epoch=3`, `best_auc=0.990270` match), so it is reused verbatim rather than re-run;
unlike the Lineage A directory, nothing has overwritten this path since.

**This is the deployed model as of 2026-11-16.** `config.yaml`, `inference_faces.py`,
`face_calibration.py`, `face_timing.py`, `verify_adapter_parity.py` and `adapter_loader.load_model`
all default to `./models/InternVL3-2B` + `runs/intern_v2_ground/*`, and the adapter package in
`runs/intern_v2_ground/adapters/` passes parity against this checkpoint (§2).

Per-split figures with 95 % intervals, extracted by `eval_metrics.py`. Positive class = FAKE;
TP/FP/FN/TN are reconstructed from the logged counts and verified against the logged F1, so
precision/recall are derived rather than re-run.

| Split | N | Accuracy [95 % CI] | F1 | Precision [CI] | Recall [CI] | AUC [CI] |
| --- | --- | --- | --- | --- | --- | --- |
| ID (in-domain) | 12,819 | 0.9338 [0.9294, 0.9380] | 0.9354 | 0.9873 [0.9842, 0.9898] | 0.8887 [0.8811, 0.8959] | 0.9872 [0.9853, 0.9891] |
| CM (cross-model) | 11,249 | 0.9908 [0.9889, 0.9924] | 0.9908 | 0.9982 [0.9967, 0.9990] | 0.9835 [0.9798, 0.9865] | 0.9995 [0.9991, 0.9999] |
| CF (cross-forgery) | 12,730 | 0.7813 [0.7740, 0.7884] | 0.7210 | 0.9958 [0.9932, 0.9975] | **0.5650 [0.5528, 0.5772]** | 0.9327 [0.9282, 0.9372] |
| CD (cross-domain) | 15,468 | 0.7777 [0.7711, 0.7842] | 0.7622 | 0.8192 [0.8099, 0.8282] | 0.7127 [0.7025, 0.7226] | 0.8626 [0.8567, 0.8685] |
| Macro mean | — | 0.8709 | 0.8524 | — | — | 0.9455 |
| Validation (same checkpoint) | 4,000 | 0.9523 [0.9452, 0.9584] | — | — | — | 0.990270 [0.9872, 0.9933] |

The validation row is the checkpoint's own `best_auc` (0.990270) — the score that selected these
weights — with intervals derived from the counts (3,809 of 4,000 correct, 2,000/2,000 classes,
Wilson / Hanley-McNeil like the rest of the table).

### 1.1 What deploying the vanilla model costs (Lineage A comparison)

`intern_fused_ground` (LR-fused, **not deployed**) re-run 2026-11-16 22:52–23:43 on the same
52,266 samples, epoch 3, `best_auc` 0.993647:

| Split | vanilla (deployed) | LR-fused (not deployed) | Δ |
| --- | --- | --- | --- |
| ID accuracy / AUC | 0.9338 / 0.9872 | **0.9410 / 0.9891** | +0.72 pp / +0.0019 |
| CM accuracy / AUC | **0.9908 / 0.9995** | 0.9889 / 0.9993 | −0.19 pp / −0.0002 |
| CF accuracy / AUC | 0.7813 / 0.9327 | **0.8256 / 0.9482** | +4.43 pp / +0.0155 |
| CD accuracy / AUC | **0.7777 / 0.8626** | 0.7698 / 0.8533 | −0.79 pp / −0.0093 |
| Macro accuracy / F1 / AUC | 0.8709 / 0.8524 / 0.9455 | **0.8813 / 0.8720 / 0.9475** | +1.04 pp / +0.0196 / +0.0020 |

**The trade is explicit and must be stated, not smoothed over:** shipping the vanilla checkpoint
costs **1.04 pp macro accuracy and 4.43 pp on cross-forgery**, and buys 0.79 pp on cross-domain
and 0.19 pp on cross-model. It is the better choice only if the deployment constraint that drives
it — portability to the unmodified upstream base — is worth ~1 pt of mean accuracy. Say that
sentence in the paper; do not present vanilla as the stronger model.

**Both models fail CF the same way, at different severity.** Vanilla precision 0.9958 / recall
**0.5650** (misses 43 % of cross-forgery edits); LR-fused 0.9947 / 0.6547 (misses 35 %). CD is a
different failure shape on both (vanilla 0.8192 precision / 0.7127 recall; fused 0.7770 / 0.7569).
Precision is never the problem on CF — the model is conservative, not confused.

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

Source: `verify_adapter_parity.py`, `test_adapter_loader.py` (21 tests). Measured on the **deployed**
package (`runs/intern_v2_ground/adapters` on `./models/InternVL3-2B`), which is what the tool
defaults to since 2026-11-16.

| Claim | Number | Status |
| --- | --- | --- |
| Adapter package reproduces the checkpoint | **18/18 identical verdicts, max \|Δp_fake\| = 0.000e+00** (`images=18 verdict mismatches=0 tol=1e-04`, vanilla run 2026-11-16 00:19) | **MEASURED** |
| Package contents | llm_adapter 392 tensors / 73,859,072 params; vision_adapter 192 / 6,291,456; custom_heads 59 / 13,257,034 | **MEASURED** |
| The wrong base is refused | `check_base_model` (pure): different base raises, same base accepted, absent provenance tolerated — plus `_check_base` on a real temp dir, and 9 path-identity tests | **VERIFIED** |
| Dropped / unknown adapter keys are refused | `check_adapter_keys` (pure): dropped key, unknown key, mismatch in both directions, and non-set iterables | **VERIFIED** |
| The real load path uses those checks | wiring tests with a stubbed model and stubbed peft/safetensors: `_attach_one` calls `check_adapter_keys` exactly once with the file keys, the live keys and the subdir; `attach_adapters` delegates to `check_base_model` with the resolved provenance, and skips it when no base is requested | **VERIFIED** |

Safe sentence: "the 93.4 M-parameter adapter package is verdict-identical to the 4.07 GB deployed
checkpoint on the comparison set" (18 single-face images). The Lineage A package passed the same
check earlier the same day, so the lossless-deployment property holds for both lineages. Do not
say "bit-identical weights" — the claim is about verdicts and probabilities, which is what was
compared.

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

### 1.2 Where CD loses its points — deployed vanilla model

`eval_by_generator.py` run on `runs/intern_v2_ground/checkpoints/best.pth` (2026-11-16 00:23–00:38,
~15 min). It reproduces the §1 CD row exactly — 8,741 / 6,727 predictions, 0.7777 acc, 0.8626 AUC —
which is the check that the grouping is right.

| Generator | N | Acc [95 % CI] | Precision | Recall | F1 | AUC [CI] |
| --- | --- | --- | --- | --- | --- | --- |
| FFIW | 6,832 | 0.8145 [0.8052, 0.8236] | 0.883 | 0.725 | 0.7963 | 0.9010 [0.8935, 0.9086] |
| deepfacelab | 3,094 | 0.7027 [0.6863, 0.7185] | **0.631** | **0.975** | 0.7664 | 0.8554 [0.8420, 0.8689] |
| dreamina | 952 | 0.9674 [0.9542, 0.9770] | 0.996 | 0.939 | 0.9665 | 0.9995 [0.9980, 1.0000] |
| gpt4o | 630 | 0.8556 [0.8259, 0.8809] | 0.996 | 0.714 | 0.8318 | 0.9899 [0.9820, 0.9978] |
| hailuo | 1,000 | 0.8730 [0.8509, 0.8922] | 1.000 | 0.746 | 0.8545 | 0.9971 [0.9937, 1.0000] |
| InfiniteYou-CD | 2,960 | 0.6615 [0.6442, 0.6783] | 0.994 | **0.325** | 0.4898 | 0.9013 [0.8899, 0.9127] |
| **SPLIT TOTAL** | 15,468 | **0.7777 [0.7711, 0.7842]** | 0.819 | 0.713 | 0.7622 | **0.8626 [0.8567, 0.8685]** |

| Generator | mean p_fake on REAL | mean p_fake on FAKE |
| --- | --- | --- |
| FFIW | 0.1584 | 0.7014 |
| deepfacelab | **0.5599** | 0.9243 |
| dreamina | 0.0533 | 0.9008 |
| gpt4o | 0.0532 | 0.6989 |
| hailuo | 0.0519 | 0.7249 |
| InfiniteYou-CD | 0.0518 | **0.3497** |

**The same two failure modes as Lineage A, both worse.** deepfacelab still over-triggers on real
faces (mean p_fake 0.5599; precision 0.631 at recall 0.975 — it flags 2,391 of 3,094 faces).
InfiniteYou-CD is still the silent miss, and markedly worse than on the fused model: **recall
0.325** (fused: 0.432), mean p_fake on fakes 0.3497. A third of the CD loss is one generator where
the vanilla model simply does not see the forgery.

Ranking per generator is stable between the two lineages (dreamina ≈ 0.97 both, FFIW 0.81 both,
deepfacelab 0.64 vs 0.70, InfiniteYou 0.71 vs 0.66), so the *diagnosis* transfers even though the
absolute numbers do not. That is a defensible sentence for the paper; a single CD average is not.

### 1.3 The same breakdown on Lineage A (LR-fused, comparison model)

Same harness, `runs/intern_fused_ground/checkpoints/best.pth` (2026-11-16, epoch 3). Split total
**0.7698 acc / 0.8533 AUC** (N = 15,468) — identical to the §1.1 comparison row, which is the
cross-check that the grouping is right. Kept because the per-generator *ranking* is what transfers
between lineages while the absolute numbers do not.

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
crop path scores *below* the whole-image path for the deployed model too (best doc AUC 0.8984 at
margin 0.45 vs 0.9090 whole-image). The crop path exists to see faces the model otherwise cannot;
it is competitive, not superior.

## 6. Calibration: margin, threshold, passthrough floor

Sources: `face_calibration.py`, `FACE_PIPELINE_CALIBRATION.md`, 400 FFIW frames (200/200,
seed 1234, one frame per video). **Re-measured on the deployed vanilla adapters 2026-11-16**
(`runs/calibration/ffiw_report_vanilla.json`, cache `ffiw_cache_vanilla.jsonl` — a separate cache
file, because the cache stores scores and would otherwise silently serve Lineage A probabilities).
438 faces, 0-6 per frame, face fraction 0.0007-0.7107.

| Claim | Number | Status |
| --- | --- | --- |
| Margin 0.30 is **no worse** than tight crops | doc AUC 0.8944 [0.8633, 0.9207] vs 0.8812 [0.8471, 0.9104] (+1.32 pt, **intervals overlap**); acc 0.8050 vs 0.7575 | **MEASURED** |
| The size gate excludes materially harder faces (vanilla) | < 96 px AUC 0.7596 [0.525, 0.959] (40 faces) vs >= 96 px 0.9065 [0.877, 0.931] (398) — a 14.7 pt gap. On Lineage A the same split was 0.8626 vs 0.8926, i.e. nearly flat | **MEASURED** |
| Padding mode has no measurable effect | `none` 0.9722 [0.846, 1.000] (15 faces), `reflect` 0.8875 [0.754, 0.983] (31), `stretch` 0.8974 [0.863, 0.927] (392) at margin 0.30 | **MEASURED** |
| Margin 0.45 looks better, for the wrong reason | 0.8984 doc AUC, but 99 % of faces are already >2× upscaled at margin 0 and 423/438 crops degrade to `stretch` | **MEASURED** |
| `face_threshold` is not a tuned number | doc accuracy spans 0.750-0.820 across T = 0.30-0.75; the best (0.820 at T = 0.39-0.40) is +0.5 pp over T = 0.50 and inside 1 SE, and face accuracy is flat (0.799-0.829) | **MEASURED** |
| Passthrough floor sits in a stable basin | face fractions 0.02-0.20 give identical results (AUC 0.90855, acc 0.8225, 92 % passthrough); even 0.30 moves only to 0.9067 / 0.820 | **MEASURED** |
| The floor changes real verdicts when it matters | `chick.jpg` whole-image 0.8832 vs crop 0.0526; `om.jpg` 0.7874 vs 0.0564 (vanilla scores) | **MEASURED** |

**DO NOT CLAIM** that FFIW validates the multi-face path: it is 256 px and 92 % single-face, so
it routes passthrough. The margin sweep bypasses routing to score crops, which is why the margin
question is answerable at all — say that explicitly.

**DO NOT CLAIM** a calibrated `face_threshold` improvement. The sweep's spread is inside the
error bar (n = 400, 1 SE ≈ 1.5 pt); the value is confirmed, not optimised.

**DO NOT CLAIM** that margin 0.30 is *significantly* better than tight crops or that mirrored
padding is harmful — those intervals overlap. 0.30 is chosen for non-inferiority plus training-frame
agreement.

**The gate is model-specific — do not generalise the Lineage A finding.** For the deployed vanilla
model the 96 px gate excludes faces that score **14.7 pt** worse (0.7596 vs 0.9065 AUC), so the gate
is doing real work; on Lineage A the same comparison was 0.8626 vs 0.8926, which is why an earlier
version of this ledger said the gate was not validated. Both statements are true of their own
checkpoint; quote the one for the model you are describing.

## 7. Latency

Source: `face_timing.py`, `FACE_PIPELINE_CALIBRATION.md` §5. Medians of 15 calls for
detect/classify, mean of 2 generations per face for reasoning, `torch.cuda.synchronize()` around
every call.

| Claim | Number | Status |
| --- | --- | --- |
Measured on the **deployed vanilla** adapters 2026-11-16 (`runs/timing/vanilla_timing*.log`); the
Lineage A figures are kept in parentheses for comparison.

| Claim | Number | Status |
| --- | --- | --- |
| Detect | 15-23 ms across a 256² to 8500² input (vanilla); Lineage A measured 13.6-21.7 ms | **MEASURED** |
| Classify | 167 ms (1 face), 516-525 ms (4 faces) — ~0.12-0.13 s/face (Lineage A: 151.4 / 518.9 ms) | **MEASURED** |
| Reason | **15.7-20.9 s per face** in this session (2 faces 19.7-20.6 s; 4 faces 16.0-20.9 s); ~98-99 % of total wall time | **MEASURED** |
| 1-face end to end, `top1` | **15.9 s** (classify 0.17 s + one generation 15.7 s) | **MEASURED** |
| 4-face end to end | `top1` **21.5 s** vs `all` **72.6 s** — ratio **3.38×** (Lineage A: 2.82-2.84×) | **MEASURED** |
| Screening without reasoning | **0.20 s** (1 face) to **0.55 s** (4 faces) | **MEASURED** |

The reason times above are a **single-session** reading. The same rig produced a ~4× spread between
sessions on Lineage A (14.9-117.9 s/face), so quote the **ratio** (`all` is 3.38× `top1` at 4 faces)
as the stable quantity and the absolute seconds as "measured in one session on one A5000".
| Lowering `max_new_tokens` does not save time | 512 vs 256: 25.3 vs 24.5 s; 512 vs 192: 18.5 s but the slowest face's answer was truncated to `unknown`; single-face identical at all three caps | **MEASURED** |

**DO NOT CLAIM** these as throughput figures: single machine, unlocked clocks, one 4-face image
(a synthetic montage), one reasoning repeat per face for the vanilla run, and ~1 in 15 classify
calls lands 3-4 s outside the median.

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
python verify_adapter_parity.py        # 18/18 identical verdicts, Δ = 0 (vanilla defaults)
python eval_metrics.py --log runs/intern_v2_ground/logs/eval_best.log        # §1 (deployed)
python eval_by_generator.py --checkpoint runs/intern_v2_ground/checkpoints/best.pth \
    --split cd --out runs/timing/eval_by_generator_cd_vanilla.json   # ~15 min, CD only
python face_calibration.py --adapter_dir ./runs/intern_v2_ground/adapters \
    --cache runs/calibration/ffiw_cache_vanilla.jsonl --n_per_class 200      # ~3 min
python face_timing.py --adapter_dir ./runs/intern_v2_ground/adapters \
    --skip_reason --repeats_fast 15    # ~90 s, detect/classify only
# Lineage A comparison: same commands with ./models/internvl3_2b_lr_fused,
#   runs/intern_fused_ground/*, runs/timing/epoch3/eval_best.log and a separate cache file
```
