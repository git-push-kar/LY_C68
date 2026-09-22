# Deepfake Detection + Reasoning — Full Domain Report v1

**Date:** 2026-09-22 (covers everything from project start to today + future plan)
**Project root:** `C:\Users\Admin\Desktop\ly project c 68\deepfake_project`
**Goal:** image deepfake classifier that also *explains itself* in structured natural language.
**Hardware:** NVIDIA RTX A5000, 25.8 GB VRAM, CUDA.
**Envs:** `veritas` = training/eval/inference env (`miniforge3/envs/veritas`); base `miniforge3` python does NOT have `peft` — always use the veritas python. `requirements.txt`: torch 2.6+cu126, transformers 4.37.2, peft 0.7.1, accelerate, timm, sklearn, matplotlib.
**Prior report:** `REPORT_exp2_to_exp3.md` (still valid, focused on the exp2→exp3 corrective). THIS file supersedes it as the single complete context.

---

## 0. TL;DR for a newcomer

1. We detect fake faces with a frozen InternVL3-2B vision+LLM backbone, LoRA adapters (rank 64), a 2-layer classification head, and a 2-token visual prefix projector. Only ~3.4–3.8% of params train.
2. Training data = HydraFake (48,320 unique train images). Every image trains the classifier; only the ~35k images with rich reasoning traces train the explainer (joint loss `cls + 0.3·lm`).
3. History: RUN1 (from-scratch ViT + GRU, dead) → RUN2/exp2 (CLIP→InternVL3, 11 messy epochs, good classifier / broken reasoning) → exp3 corrective (clean data + frozen heads + fresh schedule, 5 clean epochs).
4. Current best = exp3 epoch 3 (`best.pth`): test MEAN acc 0.809 / F1 0.774 / AUC 0.905; CM ~0.98, ID ~0.78, CF ~0.76, CD ~0.71.
5. Known weaknesses: face-edit fakes (StarGANv2-style) can fool it confidently; reasoning text is fluent but mostly templated, and drifts from the classifier on hard fakes; low-res reals get low confidence.
6. Next: ep6 training (`config.yaml` already set: resume ep005, `epochs_joint: 6`), eval `best.pth` on all 4 splits, then paper experiments (grounding tests, ablations).

---

## 1. Data — HydraFake

### 1.1 Layout (on disk)

```
datasets/hydrafake/
  train/  val/  test/{id,cm,cf,cd}/<Generator>/...
  jsons/train/{all.json, sft_36k.json, mipo_3k.json, pgrpo_8k.json, fake/, real/}
  jsons/val/  jsons/test/{id,cm,cf,cd}/*.json
```

- Test JSONs (23 files): `id` = FaceForensics++, facevid2vid, Hallo2, Midjourney, StyleGAN; `cm` = AdobeFirefly, Flux11Pro, HART, Infinity, MAGI, StarryAI; `cf` = CodeFormer, FaceAdapter, ICLight, InfiniteYou, PuLID, StarGANv2; `cd` = deepfacelab, dreamina, FFIW, gpt4o, hailuo, InfiniteYou-CD.
- Test sizes (from eval logs): ID 12,819 (R/F 5909/6910), CM 11,249 (5625/5624), CF 12,730 (6364/6366), CD 15,468 (7735/7733). Total ≈ 52,266.
- Val = 4,000 (2000/2000).
- Train: 48,320 unique images (`all.json`).

### 1.2 The two JSON dialects (root of most past pain)

- **Bare** (72% of raw entries): `<answer>real|fake</answer>` only (~23 chars). Sources: `all.json`, 26 nested per-generator files, `pgrpo_8k.json`.
- **Rich** (28%): full `<fast>…</fast><planning>/<reasoning>/<reflection>/<conclusion>…<answer>X</answer>`. Sources: `sft_36k.json` (36,750) + `mipo_3k.json` (3,480 pairs, reserved for future DPO).
- Same image appears in several files with DIFFERENT targets. Without dedupe: 144,903 entries for 48,320 images.

### 1.3 How we use them TODAY (`dataset.py`)

- `_extract_reasoning` returns the assistant message **verbatim** (tags + answer kept). `_sanitize_reasoning` guarantees non-garbage. EOS (`<|im_end|>`) appended. Targets longer than `lm_seq_len=512` are **dropped, never truncated** (truncation would teach stop-without-answer).
- Keep-richest merge per image: tagged trace beats bare answer. Counts: **35,367 usable / Missing 0 / Duplicates merged 96,583 / No-trace skipped 11,277 / Too-long skipped 1,676**.
- Separated pools (`build_separated_loaders`): CLS pool = `all.json` (48,320, bare OK); SFT pool = `sft_36k.json` (~35k usable, rich required); PREF pool = `mipo_3k` (loaded, inactive). `pgrpo_8k` never loaded.
- `_resolve_path`: JSON paths like `hydrafake/test/FFIW/…` map to `<dataset_root>/test/<cd|cf|cm|id>/FFIW/…` via `_TEST_GENERATOR_MAP` (`dataset.py:40-64,105-115`).
- Test/val use `require_reasoning=False` + minimal fallback targets so eval stays cls-only and full-size.

**One-line version:** every image trains the classifier; ~73% (rich, ≤512 tok) additionally train the explainer.

---

## 2. Model evolution — RUN1 → RUN2 → current

|  | RUN1 (`RUN1/`) | RUN2 (`RUN2/`) | Current (`model.py`) |
|---|---|---|---|
| Backbone | From-scratch ViT (`transformer_blocks.py`) | Pretrained CLIP ViT (768-d CLS) | **InternVL3-2B** (InternViT 448px + Qwen2 LLM), bf16, frozen |
| Adaptation | Full training | From-scratch projection + transformer cls head | **LoRA r=64/α=128/drop 0.05** on `q/k/v/o/gate/up/down_proj` (392 tensors, 73.86M params) |
| Classifier | MLP on CLS | Transformer-based head | **2-layer head on CLS + mean-patch concat** (`ClassificationHead`) |
| Reasoning | GRU decoder | GRU decoder | **The LLM itself** via 2-token visual prefix (`CustomProjector`: CLS token + patch-mean → 2 prefix tokens) |
| Extra | — | — | `DomainRouter` (kept for ckpt compat, excluded from compute) |
| Status | Abandoned (too weak) | Stepping stone to exp2 | **Active. Total 2.17B, trainable 82.4M (3.79%)** |

Key fixes along the way: LoRA applied BEFORE freezing (else zero trainable params — hard-asserted at init); LM labels masked by `attention_mask` (not token==0); deterministic `predict(do_sample=False)` for forensics; cp1252 `✓`→`[OK]`.

Forward modes: `model(pv)` = cls-only (val/eval/inference classifier); `model(pv, labels, reasoning_tokens, mask)` = joint (train). Inference generation: `model.predict(pv, tokenizer, max_new_tokens=512)` greedy + `use_cache`, output cut after last `</answer>` (`trim_after_answer`).

---

## 3. Training history

### 3.1 exp2 — `runs/intern_exp2` (the baseline that worked, then rotted)

- Config (archived, `config.yaml:9-28`): batch 8 / accum 4 (eff 32), `epochs_cls 3 + epochs_joint 2`, `lm_seq_len 192`, lr 5e-5, LoRA 64/128, `lm_weight 0.3`, no freezes.
- Reality: stretched to **11 epochs via 20+ resumes** (`epochs_joint 2→5→7→10`), scheduler rebuilt each time (`total_steps 90560` for a 5-epoch plan). Trusted checkpoint **`ep010.pth`** (~5.1 GB). Discarded `ep011.pth` (LaTeX/CJK drift, worse reasoning) and `best.pth` (213 MB truncated/corrupt).
- Result: strong classifier, incoherent reasoner (see §4).

### 3.2 What went wrong (P1–P11, detail in `REPORT_exp2_to_exp3.md`)

P1 mixed bare/rich targets (72/28). P2 tag-stripping (`" ".join(TAG_RE.findall)`, answer dropped) vs inference expecting tags. P3 no EOS → rambling to token cap. P4 dupes (train 144,903/48,320; val 8000/4000; 3× epoch time). P5 silent black-image fallback. P6 `GradScaler` BF16 crash. P7 stochastic sampling (0.8/0.9). P8 log dir outside repo + no trim. P9 patchwork scheduler/optimizer. P10 AUC-only `best` picked ep11 despite acc drop. P11 cp1252 crash.

### 3.3 exp3 corrective — `runs/intern_exp3_corrective` (current line)

| Phase | Init | Epochs | LR | Frozen | Outcome |
|---|---|---|---|---|---|
| A: frozen corrective | `--weights_from …/exp2/ep010.pth` (model-only, critical-key guard) | cls 0 / joint 2 | 1e-5 | cls_head (4T) + projector (12T) | ep001, protected classifier |
| B: unfrozen continuation | `--weights_from …/exp3/best.pth` | 0 / 2 | 5e-6 | none | new ep1 (unfrozen) |
| C1: resume | `--resume ep001.pth` | 0 / 2 | 5e-6 | none | ep002 |
| C2: 3 more | `--resume ep002.pth` | 0 / 5 | 5e-6 | none | ep003–ep005 |
| **Next (set, not run)** | `--resume ep005.pth` | 0 / **6** | 5e-6 | none | **ep006** |

- Common: batch 4 / accum 8 (eff 32, same as exp2), `lm_seq_len 512`, `lm_weight 0.3`, joint Option A (8,765 forward pairs/epoch = 35,060 SFT + 35,060 CLS images; 1,096 optimizer updates; ~72.6% CLS coverage/epoch), fresh cosine schedule w/ 5% warmup, patience 5.
- `epochs.csv` (train): ep1 0.834/0.859 acc, va_auc 0.9386 → ep3 peak va_auc **0.9419** (`is_best=1`) → ep4 0.9416, ep5 0.9412 (plateau, no collapse).
- Fixes shipped: GradScaler removed (pure `loss.backward()` + clip 1.0); tokenization in dataset; per-step/per-epoch CSVs + `curves.png`; atomic ckpt saves (`.tmp` + `os.replace`); central `config.yaml` (CLI always overrides).

---

## 4. Current condition (2026-09-22)

### 4.1 Checkpoints (`runs/intern_exp3_corrective/checkpoints/`)

`ep001`–`ep005.pth` (~4.72 GB each) + `best.pth` (~4.36 GB) = **epoch 3** (val AUC 0.9419). All load with `missing=0, unexpected=0`.

### 4.2 Test evals (all 5 epochs complete, `logs/eval_ep00{1-5}.log`)

| Ckpt | ID acc/F1/AUC | CM acc/F1/AUC | CF acc/F1/AUC | CD acc/F1/AUC | MEAN |
|---|---|---|---|---|---|
| ep1 | .7716/.7433/.9019 | .9794/.9795/.9977 | .7679/.7092/.9113 | .7091/.6516/.7965 | .8070/.7709/.9018 |
| ep2 | .7774/.7520/.9049 | .9808/.9809/.9979 | .7636/.7000/.9160 | .7135/.6622/.7951 | .8088/.7738/.9035 |
| ep3 ⭐ | .7810/.7571/.9071 | .9808/.9808/.9979 | .7601/.6931/.9168 | .7141/.6646/.7966 | **.8090/.7739/.9046** |
| ep4 | .7792/.7545/.9062 | .9809/.9809/.9979 | .7610/.6952/.9147 | .7124/.6598/.7974 | .8084/.7726/.9040 |
| ep5 | .7788/.7536/.9058 | .9808/.9809/.9979 | .7623/.6976/.9138 | .7124/.6594/.7975 | .8086/.7728/.9037 |

Read: plateaued after ep3 (ΔMEAN ep3→ep5 ≈ −0.0004). CM saturated. CD weakest. `best.pth` (ep3) NOT yet 4-split-evaluated — only used for inference so far.

### 4.3 Inference spot-checks (all on `best.pth`)

- 10-image scored set (5 real + 5 fake, diverse generators): **classifier 9/10** (reals 5/5; fakes 4/5 — StarGANv2 face-edit miss at 97% REAL), **reasoning 7/10** (2 classifier/reasoner drifts on deepfacelab + gpt4o).
- Custom `test_images`: 1.jpg FAKE 77.7%, 2.jpg FAKE 84.0%, 3.jpg REAL 64.6%, om.jpg FAKE 88% (drift — story said real), omi.jpg FAKE 75.6% (full agreement, planning+reflection present).
- Reasoning quality verdict: **fluent but mostly templated.** Proof: 306_680.png (real) and img_116.jpg (fake) share near-identical paragraphs; all reals recite symmetrical→skin→pores→lighting with zero image-specific detail. Only StyleGAN/Flux11Pro/custom 1–2.jpg traces name concrete cues. Treat `<answer>` as decoration, classifier as verdict.

### 4.4 What's good / what's bad

Good: SOTA-ish CM generalization; no train/val collapse; stable joint training (cls ~0.43, lm ~0.62); clean data pipeline with counts; reproducible CLI+config; atomic ckpts; full eval coverage ep1–5.
Bad: CD/CF ceiling (~0.71–0.76 acc); confident miss on subtle face-edits; reasoning ungrounded + drifts on hard fakes; low-res reals → coin-flip confidence (FFIW 56%); ep4–5 show diminishing returns; `best.pth` 4-split eval still missing; reasoning never quantitatively evaluated.

---

## 5. Tooling (how to run everything)

- `config.yaml` = central config; **CLI args always win** (`train.py:516-533`, `eval.py:146-156`, `inference.py:206-211`). Active block: resume ep005, joint 6, lr 5e-6, batch 4/accum 8, seq 512.
- `python train.py` (veritas python) — trains per config; logs `runs/<exp>/logs/{train.log,steps.csv,epochs.csv,curves.png}`, ckpts `checkpoints/epNNN.pth`.
- `python eval.py --checkpoint … --batch_size 8 --num_workers 8` — 4-split table → `logs/eval_<ckpt>.log`. ~40 min/full ckpt on A5000.
- `python inference.py --checkpoint … --images … --log_dir ./runs/intern_exp3_corrective/logs` — classifier + greedy reasoning per image. NOTE: default `log_dir` is stale (`runs/intern_exp2/logs`) — always pass `--log_dir`. Images may be repo-relative or absolute.
- `run_all_eval.bat` (cmd.exe only): now evals **ep4, ep5, best** (ep1–3 skipped, logs complete), guards missing ckpts, anchors `cd /d %~dp0`, then `python train.py` (ep6). No `pause`.
- Resume math: `total = epochs_cls + epochs_joint`; `start = ckpt_epoch + 1`. So joint=6 from ep005 → exactly ep006.

---

## 6. Future

1. **Immediate:** run ep6 (`python train.py` as-is); 4-split-eval `best.pth` AND ep006; compare vs ep3 plateau.
2. **Paper experiments:** (a) reasoning-grounding test — occlude/shuffle regions, measure text change + phrase-uniqueness over ~50 samples; (b) ablations — `lm_weight ∈ {0, 0.12, 0.3}`, frozen vs unfrozen heads, seq 192 vs 512; (c) error analysis on CF/CD misses (StarGANv2 family); (d) calibration plot (confidence vs accuracy — FFIW/gpt4o cases suggest overconfidence).
3. **Next modeling:** DPO on `mipo_3k` prefs (loader ready, inactive); unfreeze vision late-stage probe; reasoning-answer consistency loss (penalize drift); quantitative reasoning eval set (human/GPT-judge rubric: specificity, localization, correctness).
4. **Paper skeleton:** Abstract → Intro (why explainable detection) → Related (HydraFake, InternVL, LoRA-forensics) → Data (dialects, keep-richest) → Method (prefix-LoRA joint) → What-broke (P1–P11) → Experiments (tables §4.2 + ablations) → Error/Reasoning analysis (§4.3–4.4) → Limitations → Future → Repro appendix (§5).
5. **Stop rule:** if ep6 MEAN − ep3 < +0.002, freeze the line at ep3/best and pivot to paper + DPO.

---

## 7. File map (where everything lives)

`train.py` (pipeline) · `dataset.py` (pools, keep-richest, resolve) · `model.py` (prefix-LoRA) · `eval.py` (4-split) · `inference.py` (cls+generate) · `config.yaml` (active: resume ep005→ep6) · `run_all_eval.bat` (eval 4/5/best + train) · `runs/intern_exp2/` (baseline) · `runs/intern_exp3_corrective/{checkpoints,logs}/` (all ckpts + eval/inference/train logs) · `RUN1/ RUN2/` (archived code) · `graphs/` (clean figs + maker) · `reports/REPORT_exp2_to_exp3.md` (prior focused report) · `test_images/` (customs incl. om/omi) · `requirements.txt` + `RUN1|RUN2|datasets/environment.yml`.

## 8. Glossary of runs (don't confuse)

- RUN1/exp1 = from-scratch ViT+GRU (dead). RUN2/exp2 = CLIP-era code dir, but `runs/intern_exp2` = the 11-epoch InternVL3 baseline. exp3 = corrective line (ep001 from exp2-ep010 frozen; then unfrozen ep1→ep5; best=ep3; ep6 pending). "New ep1" in old notes = phase-B restart, NOT phase-A ep001 — same filename, different weights (phase-B overwrote `ep001.pth`).
