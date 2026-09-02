# Deepfake Detection + Reasoning — Corrective Report Since `exp2`

**Project:** `C:\Users\Admin\Desktop\ly project c 68\deepfake_project`  
**Base model:** `InternVL3-2B` (InternViT 448px + Qwen2 LLM + LoRA rank 64 / alpha 128)  
**Baseline:** `runs/intern_exp2` — 11 epochs on `datasets/hydrafake` (InternViT frozen, LoRA on LLM)  
**Trusted checkpoint:** `runs/intern_exp2/checkpoints/ep010.pth` (5,166,616,259 bytes, ZIP_OK)  
**Discarded:** `ep011.pth` (worse reasoning, LaTeX/CJK drift), `best.pth` (213 MB truncated/corrupt vs ~5.1 GB)  
**Corrective run:** `runs/intern_exp3_corrective` via `--weights_from ep010.pth`  
**Date:** 2025-08-24+

---

## 1. Executive Summary

`exp2` trained successfully for 10 epochs, then degraded. Root cause was **three conflicting reasoning-target formats mixed in the same training set** (72% bare answers vs 28% rich tagged traces) plus **no EOS**, **duplicate loading**, and a **patchwork scheduler** rebuilt across 20+ resumes. Corrective fine-tuning keeps **all ep10 knowledge** (frozen vision + frozen classifier) and retrains **only LoRA** on a single coherent tagged+EOS target format. Sequence length was raised **192 → 512** to preserve full justifications without truncation (cost: 0.76s → 2.77s/step). All CLI paths verified reachable.

---

## 2. Problems Found Since `exp2`

| # | Problem | File:Line | Evidence |
|---|---------|-----------|----------|
| **P1** | **Mixed / conflicting training targets** | `datasets/hydrafake/jsons/train` | `all.json` (48,320) + 26 nested per-generator (48,320) + `pgrpo_8k.json` (8,033) = **bare** `<answer> X </answer>` (23 chars) → 104,673 (72%) vs `sft_36k.json` (36,750) + `mipo_3k.json` (3,480) = **rich** `<fast>…</fast><reasoning>…</reasoning><conclusion>…</conclusion><answer> X </answer>` → 40,230 (28%). Without dedupe, 144,903 entries for 48,320 unique images mixed both modes. |
| **P2** | **Train / inference format mismatch** | `dataset.py:71` vs `inference.py:81` | `_extract_reasoning` did `" ".join(TAG_RE.findall)` — **tags stripped, `<answer>` dropped** — while `inference.py` parses `r"<(fast|planning|reasoning|reflection|conclusion)>"` + `<answer>` and falls back to `[Raw LM output]` + `answer=unknown`. |
| **P3** | **No EOS — model never learns when to stop** | `dataset.py:219` (old) | Tokenized `reasoning` directly, never appended `tokenizer.eos_token` (`<|im_end|>` id 151645) → generation always ran to `max_new_tokens` and rambled. |
| **P4** | **Duplicate sample loading** | `dataset.py:165` `glob("**/*.json")` | `train` 144,903 entries / 48,320 unique; `val` 8,000 / 4,000 unique (`all+real+fake`). `steps/epoch 18112` = `144,903/8`. 3× epoch time, skewed sampling. |
| **P5** | **Silent black-image fallback** | `dataset.py:300` | Missing file → `Image.new(RGB,0)` with label kept; doubled `train\fake\fake\EFG` (since fixed on disk) would have trained on black images invisibly. |
| **P6** | **GradScaler BF16 crash** | `train.py:47,243` | `GradScaler` is FP16-only → `RuntimeError: _amp_foreach_non_finite_check_and_unscale_cuda not implemented for BFloat16` at `scaler.unscale_` (reproduced at Step 0). |
| **P7** | **Stochastic inference** | `model.py:338` | `predict(do_sample=True, temp 0.8, top_p 0.9, rep 1.3)` — only path `inference.py:210 model.predict()` uses — unstable forensic outputs. |
| **P8** | **Wrong log dir + no trim** | `inference.py:62` | `default=r"C:\deepfake_project\runs/..."` → today's logs landed outside repo; no `trim_after_answer` → 512-token ramble. |
| **P9** | **Scheduler / optimizer patchwork** | `runs/intern_exp2/logs/train.log:10` | `total_steps 90560` (5 epochs) for 11-epoch run; 20+ `Resumed from epoch X` rebuilding cosine with `epochs_joint 2→5→7→10`. State untrustworthy. |
| **P10** | **Checkpoint selection AUC-only** | `train.py:525` | `cur_auc > best_auc` → `best.pth` = ep011 (AUC 0.946) despite `VaAcc .858→.818` drop and visibly worse reasoning. |
| **P11** | **Windows cp1252 crash** | `model.py:228` | `print("… ✓")` → `UnicodeEncodeError: charmap can't encode \u2713`. |

---

## 3. Changes Made

### 3.1 `dataset.py` — Reasoning Target Overhaul

**Before:**
```python
def _extract_reasoning(messages):
    parts = _TAG_RE.findall(content)
    if parts: return " ".join(p.strip() for p in parts)  # tags + answer gone
    return _ANSWER_RE.sub("", content).strip()
```

**After (`L71-L86`):**
```python
def _extract_reasoning(messages):
    """Return FULL assistant response verbatim, preserving all tags AND <answer>."""
    for msg in messages:
        if msg.get("role") != "assistant": continue
        return msg.get("content","").strip()
    return ""
```

**`_load` (L173-L289) — two-phase keep-richest merge:**

1.  Phase 1 — `rich = 1 if _TAG_RE.search(raw) else 0`; `candidates: normcase(path) → (rich, resolved, label, raw, ftype)`; later `rich` wins (tagged `sft`/`mipo` beats bare `all`/nested/`pgrpo`). Counts `Missing` (skipped, not appended), `Duplicates(merged)`.
2.  Phase 2 — `if not rich and split=="train": skip` (11,277 dropped); `val`/`test` instead fall back to `raw=""` → sanitized minimal tagged target so `validate()`/`eval.py` stay 4,000 / 52,266. **EOS appended:** `target_text = reasoning + tokenizer.eos_token`. **Drop > `max_text_len`:** `n_tok = len(tokenizer(target_text)["input_ids"]); if >512: skip` — never truncates mid-answer (which would teach stop-without-answer).
3.  `max_text_len` `192 → 512` (`L153, L329`), `build_dataloaders` default `L329`.

**Result (smoke verified):** `35,367 usable / 48,320 unique / Missing 0 / Duplicates 96,583 / No-trace 11,277 / Too-long 1,676 / Fallbacks 0`; **100% contain `<answer>` + `<fast>`**, `mean 259 tok, 0 truncated (last-token == eos)`.

### 3.2 `train.py` — Corrective Fine-Tuning Mechanics

*   Header `L10-L11` comment updated `192 → 512` (95.5% intact, rest dropped not truncated).
*   Header one-liner added `L34-35` (see §6).
*   `L47` `GradScaler` import removed → `from torch.amp import autocast` only.
*   `L343` `lm_seq_len` default `192→512`.
*   New args `L356-L365`: `--weights_from` (model-only, critical-key guard `cls_head./custom_projector./lora_`), `--freeze_cls_head`, `--freeze_projector`.
*   `train_epoch` `L209` signature `scaler` removed; `L243-253` replaced `scaler.scale/unscale/step/update` with `loss.backward()` + `grads = [p for p in parameters if p.grad is not None]; clip_grad_norm(grads,1.0); optimizer.step(); scheduler.step()`.
*   `main` `L417-L453`: after `DeepfakeReasoningModel(...).to(device)` → `if args.weights_from: load ckpt["model"] strict=False, check critical prefixes` → freeze loop (`cls_head.` 4 tensors + `custom_projector` 12 tensors → **16 frozen**) → logs `Trainable 73,990,788 (3.41%)`. Optimizer then built **after** freezes: `LoRA=392, custom=4` (dormant `domain_router`), so `custom` never gets grads. Scheduler freshly built `total_steps = (0+2)*8841 = 17682` (clean, not patchwork).

### 3.3 `model.py` — Deterministic Inference

*   `L228` `✓` → `[OK]` (cp1252 fix).
*   `L339-L382` `predict(do_sample=False, temperature=1.0, top_p=1.0, repetition_penalty=1.05, use_cache=True)` — was hardcoded `True/0.8/0.9/1.3`. Docstring notes greedy is required for forensics. Existing `predict()` callers (only `inference.py`) now deterministic without code change.

### 3.4 `inference.py` — Auditable Output

*   `L62` `log_dir` hardcoded `C:\deepfake_project` → `default=None` + `L162-165` resolves `None → script_dir/runs/intern_exp2/logs`.
*   New `L97 trim_after_answer()` — cuts everything after last `</answer>` (fixes pre-EOS checkpoints that ramble to cap).
*   `L228` `raw_text = trim_after_answer(result["reasoning"][0])`.

### 3.5 `eval.py`

*   Unchanged by design — `validate()` is `model(pv)` cls-only, so `lm_seq_len` default 192 is irrelevant (kept for compat). Benefits from `val` fallback fix (still 4,000 samples).

---

## 4. Sequence Length 192 → 512 — Why and Cost

**Measured on fixed keep-richest 35,367 targets (with EOS):** `p50 236 / mean 272 / p90 457 / p95 507 / p99 576 / max 746`.  
At old cap: `>256` = 14,798 **(40% lose `<answer>`)**, `>384` = 6,240 (17%). At new 512: **0 truncated**, only 1,676 (4.5% of 37,043) **dropped cleanly** (never truncated mid-answer).

**Training time:** `192→512` is 2.67× tokens + `512²/192² = 7.1×` attention → observed `0.76s → 2.77s/step` (3.7×). Old `18112×0.76s = 3.8h/epoch`; new `8841×2.77s = 6.8h/epoch` (fewer steps but each does 2.67× useful tokens). 2 epochs ≈ 13.6h total. **Intentional** — 196/256 would re-introduce truncated-no-answer teaching.

---

## 5. Will Resuming from `ep010` Still Get Better Results? — Yes, Protected

| Component | Fate |
|---|---|
| **Vision** (InternViT) | Frozen since day 1 — untouched |
| **Classifier** (`cls_head` 4 tensors + `custom_projector` 12) | **Frozen** → `grad check OK: cls_head=0 projector=0` — mathematically cannot degrade |
| **Projector** | Frozen → LLM prefix semantics stable |
| **LoRA** (73.8M) | Adapts at low `lr 1e-5` (vs old 5e-5) to corrected tagged+EOS format — prior domain knowledge retained |
| **Base LLM** | Frozen bf16 — untouched |
| **Optimizer/Scheduler** | **Fresh** (not restored) — old patchwork discarded; `total_steps 8841` for corrective is clean |

`VaAcc/VaAUC` will stay flat by design (~.86/.94) — pick final checkpoint by `LM` loss + inference spot-check, not `Best`.

---

## 6. All CLI Directories — Verified Reachable

```
datasets/hydrafake, /jsons, /jsons/train|val|test|id|cm|cf|cd,
/train|val|test, models/InternVL3-2B,
runs/intern_exp2/checkpoints/ep010.pth (5,166,616,259 bytes) → True
```
Random train samples: `datasets/hydrafake\train\fake\EFG\Dall-E1\img_644.png` (exists True, tok len 226, has tags), etc., all resolved via `_resolve_path` with `TEST_GENERATOR_MAP`.

---

## 7. Verification Completed

* `py_compile` all 4 files `SYNTAX OK`
* Smoke corrective: `35,367 usable Missing 0 Duplicates 96,583 No-trace 11,277 Too-long 1,676`, `ep010 missing=0`, `VAL 4000 (2000/2000)`, `grad cls_head=0 projector=0 lora_with_grad=392`
* BF16 crash reproduced and fixed; cp1252 crash fixed
* `best.pth` verified corrupt (213 MB truncated zip) vs ep checkpoints ZIP_OK

---

## 8. One-Line Corrective Command

```bat
python train.py --dataset_root "C:\Users\Admin\Desktop\ly project c 68\deepfake_project\datasets\hydrafake" --json_root "C:\Users\Admin\Desktop\ly project c 68\deepfake_project\datasets\hydrafake\jsons" --output_dir ./runs/intern_exp3_corrective --weights_from ./runs/intern_exp2/checkpoints/ep010.pth --epochs_cls 0 --epochs_joint 2 --lr 1e-5 --lm_loss_weight 0.3 --lm_seq_len 512 --batch_size 4 --grad_accum 8 --num_workers 8 --freeze_cls_head --freeze_projector
```

To continue after epoch 1: replace `--weights_from .../ep010.pth` with `--resume ./runs/intern_exp3_corrective/checkpoints/ep001.pth` (keep all other flags, same `output_dir`).
