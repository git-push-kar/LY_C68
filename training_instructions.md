# Training Instructions & Strategy Guide: DeepfakeReasoningModel v2

**Target Hardware**: NVIDIA RTX A5000 (24GB GDDR6 VRAM, ~111 TFLOPS dense BF16 tensor throughput)  
**Model Architecture**: InternVL3-2B (`InternViT-300M` vision encoder + `Qwen2.5-1.5B` causal LLM) + Dual LoRA + Attention Pooling + 2D FFT Frequency Branch + Native Visual Tokens + Self-Consistency Alignment.

---

## 1. Overview

`DeepfakeReasoningModel v2` implements a multi-stage curriculum for joint binary classification and structured forensic reasoning (`<fast>`, `<planning>`, `<reasoning>`, `<reflection>`, `<conclusion>`, `<answer>`):

```
+---------------------------------------------------------------------------------------------------+
| Stage 1: Classification Warm-up (3 epochs)                                                        |
| Updates: Vision LoRA (r=16) + AttentionPool + FrequencyBranch (2D FFT) + cls_head                 |
| Data: train/all.json (48,320 images, bare/rich allowed)                                           |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| Stage 2: Joint Classification + Reasoning + Consistency (2 epochs)                                |
| Updates: Vision LoRA + LLM LoRA (r=64) + AttentionPool + FrequencyBranch + cls_head               |
| Objective: L = L_cls(all.json) + w_lm * L_sft(sft_36k.json) + w_cons * L_consistency              |
| Curriculum: w_lm ramps linearly from 0.3 -> 0.6 across joint epochs                               |
| Grounding: Full native multi-token visual representation (e.g. 256 tokens) feeds LLM             |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| Stage 3: Direct Preference Optimization (DPO, Optional, 1-2 epochs)                               |
| Updates: LLM LoRA policy alignment                                                                |
| Data: train/mipo_3k.json (3,480 chosen/rejected forensic trace pairs)                              |
+---------------------------------------------------------------------------------------------------+
```

### Important Architectural Clarifications
1. **Vision LoRA does NOT eliminate the need for `cls_head`**:
   Adding LoRA to `InternViT` only makes the extracted feature representations richer and forgery-aware. A dedicated mapping layer (`cls_head`) is still strictly necessary to project the combined feature vector `[cls_token, attn_pooled_patches, freq_features]` (dimension $1024 \times 2 + 256 = 2304$) to 2 classification logits (`real` vs. `fake`).
2. **`lm_loss_weight` Curriculum is Fixed**:
   The linear ramp from `0.3` to `0.6` is a predetermined schedule indexed strictly by the joint epoch number. It is **not** conditionally triggered by or reactive to classification accuracy/loss plateaus.

---

## 2. Hardware Assumptions & RTX A5000 Guidance (24GB VRAM)

- **Target Device**: Single NVIDIA RTX A5000 (24GB VRAM).
- **Execution Precision**: `bfloat16` (`torch.amp.autocast('cuda', dtype=torch.bfloat16)`). Bfloat16 shares the dynamic range of FP32, completely eliminating underflow and the need for `GradScaler`.
- **Memory Scaling Factors**:
  - **Vision LoRA & Frequency Branch**: Add minimal parameter overhead (~3-5M parameters), but require activation caching during the backward pass.
  - **Native Multi-Token Visual Sequence**: Passing full visual tokens (e.g., 256 tokens) to the LLM increases the attention context from ~514 tokens to ~768 tokens ($256 \text{ vision} + 512 \text{ text}$). Because transformer self-attention memory scales quadratically with sequence length, per-sample activation memory during the LLM forward/backward pass is higher than in v1.
- **Gradient Checkpointing**:
  If VRAM pressure approaches >22GB or when running with batch size > 4, gradient checkpointing can be enabled on `backbone.language_model` to trade compute for memory.
- **Timing Disclaimers**:
  All timing figures in this guide are estimates based on dense BF16 tensor execution. Actual wall-clock time depends on data-loading throughput (`num_workers`), disk I/O (NVMe vs. HDD), image decode speed (libjpeg-turbo/Pillow), and CUDA driver overhead.

---

## 3. Recommended Run Order & CLI Examples

### 3.1 Environment Setup (Virtual Environment `venv`, No Conda)

On Linux / Remote RTX A5000 instance:
```bash
# 1. Create and activate venv
python3 -m venv venv
source venv/bin/activate

# 2. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows:
```cmd
:: Run setup script or manually:
python -m venv venv
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 3.2 Stage 1: Classification Warm-up (3 Epochs)
Warms up the Vision LoRA adapter, AttentionPool, Frequency Branch, and `cls_head` on the full 48k classification dataset. The LLM reasoning path is inactive.

```bash
python train.py \
    --dataset_root "/path/to/hydrafake" \
    --json_root "/path/to/hydrafake/jsons" \
    --model_path "./models/InternVL3-2B" \
    --tokenizer_path "./models/InternVL3-2B" \
    --output_dir "./runs/intern_v2" \
    --arch_version v2 \
    --batch_size 4 \
    --grad_accum 8 \
    --epochs_cls 3 \
    --epochs_joint 0 \
    --lr 5e-5 \
    --vision_lora_rank 16 \
    --vision_lora_alpha 32 \
    --num_workers 8
```

---

### 3.3 Stage 2: Full Curriculum (3 Cls + 2 Joint Epochs from Scratch)
Runs the entire end-to-end curriculum: 3 warm-up epochs followed by 2 joint epochs with linear `lm_loss_weight` ramping and self-consistency loss.

```bash
python train.py \
    --dataset_root "./deepfake_project/datasets/hydrafake" \
    --json_root "./deepfake_project/datasets/hydrafake/jsons" \
    --model_path "./models/InternVL3-2B" \
    --tokenizer_path "./models/InternVL3-2B" \
    --output_dir "./runs/intern_v2" \
    --arch_version v2 \
    --batch_size 4 \
    --grad_accum 8 \
    --epochs_cls 3 \
    --epochs_joint 2 \
    --lr 5e-5 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --vision_lora_rank 16 \
    --vision_lora_alpha 32 \
    --lm_loss_weight_start 0.3 \
    --lm_loss_weight_end 0.6 \
    --consistency_loss_weight 0.05 \
    --lm_seq_len 512 \
    --num_workers 8
```

---

### 3.4 Stage 3: Direct Preference Optimization (Optional, 1–2 Epochs)
Aligns the reasoning outputs using the 3,480 preference pairs in `mipo_3k.json`.

```bash
python train_dpo.py \
    --dataset_root "/path/to/hydrafake" \
    --json_root "/path/to/hydrafake/jsons" \
    --model_path "./models/InternVL3-2B" \
    --checkpoint "./runs/intern_v2/checkpoints/best.pth" \
    --output_dir "./runs/intern_v2_dpo" \
    --arch_version v2 \
    --batch_size 4 \
    --grad_accum 8 \
    --dpo_epochs 2 \
    --dpo_lr 1e-6 \
    --dpo_beta 0.1 \
    --lm_seq_len 512 \
    --num_workers 4
```

---

### 3.5 Evaluation & Multi-Domain Testing
Evaluates the checkpoint across all 4 benchmark splits (**ID**: In-Domain, **CM**: Cross-Model, **CF**: Cross-Forgery, **CD**: Cross-Domain):

```bash
python eval.py \
    --dataset_root "/path/to/hydrafake" \
    --json_root "/path/to/hydrafake/jsons" \
    --model_path "./models/InternVL3-2B" \
    --checkpoint "./runs/intern_v2/checkpoints/best.pth" \
    --arch_version v2 \
    --batch_size 8 \
    --num_workers 8
```

---

### 3.6 Inference on Unseen Images
Runs forensic reasoning and binary classification on test images:

```bash
python inference.py \
    --model_path "./models/InternVL3-2B" \
    --checkpoint "./runs/intern_v2/checkpoints/best.pth" \
    --arch_version v2 \
    --images test_images/1.jpg test_images/2.jpg test_images/3.jpg test_images/4.jpg \
    --max_new_tokens 512
```

---

## 4. Batch Size / Gradient Accumulation Guidance for 24GB VRAM

In v1, a configuration of `batch_size=8, grad_accum=4` was common.  
In v2, because **both the 2D FFT Frequency Branch (2.3)** and the **Native Multi-Token Visual Sequence (2.4)** increase per-sample memory footprint:

| Parameter | Recommended Starting Setting | Effective Batch Size | Peak VRAM on RTX A5000 (24GB) |
|---|---|---|---|
| **Stage 1 (Cls Warm-up)** | `batch_size=8`, `grad_accum=4` | 32 | ~12–14 GB |
| **Stage 2 (Joint Stage)** | `batch_size=4`, `grad_accum=8` | 32 | ~18–21 GB |
| **Stage 3 (DPO Stage)** | `batch_size=4`, `grad_accum=8` | 32 | ~20–22 GB (2 model instances: Policy + Reference) |

> [!TIP]
> If you observe VRAM spikes >22.5GB during the Joint stage on an RTX A5000, decrease `batch_size` to 2 and increase `grad_accum` to 16. Effective batch size remains 32 with zero change to gradient dynamics or learning rates.

---

## 5. Estimated Training Time on RTX A5000

| Stage | Steps/epoch (approx.) | Rough per-step time | Rough epoch time |
|---|---|---|---|
| **Stage 1: Cls warm-up** (vision fwd/bwd incl. Vision LoRA + freq branch, no LLM decode) | ~6,000–8,000 (batch-size dependent) | ~0.2–0.4s | ~25–45 min |
| **Stage 2: Joint stage** (two forwards/step: CLS batch + full LLM fwd/bwd on native visual sequence + up to 512 text tokens) | ~8,000–10,000 (SFT-pool dependent) | ~1.0–2.0s (longer than v1 due to native visual sequence) | ~2.5–4.5 hrs |
| **Stage 3: DPO stage** (optional, 3,480 preference pairs) | ~435 steps (at batch=4, accum=8) | ~1.0–2.0s (two forward passes per step: policy + reference) | under 1 hr for 1–2 epochs |

- **Total Estimated Wall-Clock Time (Full v2 Curriculum)**: Roughly **6–10 hours** (3 cls epochs + 2 joint epochs + optional DPO).
- *Note*: This is slightly higher than v1 (~5.5–8.5 hrs) specifically due to the richer, multi-token spatial visual representation fed to the LLM during the joint stage. Add a **20–30% buffer** for data-loading stalls, validation passes, and atomic checkpoint saves.

---

## 6. Monitoring & Diagnostic Guidance

1. **Time & Step Progress Tracking**:
   - **Per-step logging**: Every 200 steps, the console logs progress percentage, speed in `s/step`, elapsed time, and estimated remaining `ETA`:
     `Ep 1 | Step  200/8765 (  2.3%) | 0.32s/step | Elapsed 1m 04s | ETA 45m 12s | Loss 0.6421 | Cls 0.3120 | LM 1.1003`
   - **Epoch completion summary**: At epoch completion, the total duration and average step speed are printed:
     `Stage 2 Joint Epoch 1 completed: 8765/8765 pairs | Duration: 2h 45m 12s | Avg Speed: 1.130s/step`
   - **Epoch Summary Table**: The table logs exact steps and duration per epoch:
     ```
       Ep    Stage       Steps       Time   TrLoss   TrAcc   VaLoss   VaAcc   VaAUC   LM_W  Best
        1 cls_only   6040/6040    28m 14s   0.3412  0.8650   0.3120  0.8810  0.9412   0.00     *
        4    joint   8765/8765   2h 45m   0.2840  0.9120   0.2610  0.9240  0.9650   0.30     *
     ```
2. **Log Files & Metrics**:
   - `logs/train.log`: Full timestamped console log with per-step speeds and ETAs.
   - `logs/steps.csv`: Step-by-step recording of `epoch`, `step`, `global_step`, `step_time_s`, `loss`, `cls_loss`, `lm_loss`, and `consistency`.
   - `logs/epochs.csv`: Epoch summary with `steps`, `time_sec`, `time_str`, train/val losses, accuracy, F1, validation AUC, and scheduled `lm_loss_weight`.
   - `logs/curves.png`: Training curve plots generated automatically after every epoch.
3. **Monitoring Consistency Loss**:
   - Watch the `consistency` metric in `steps.csv` and `epochs.csv`.
   - **Healthy Trend**: `consistency_loss` should decline alongside `cls_loss` and `lm_loss` (typical range: `0.05` to `0.30`).
   - **Divergence Warning**: If `consistency_loss` remains high (>1.5) while `cls_loss` drops (<0.2), the classifier and LLM answer predictions are systematically disagreeing. Check if reasoning prompts are generating unparsed answers or if label mapping is inverted.
4. **Collapse Detection**:
   - If the model begins predicting only class 0 or only class 1, a `COLLAPSE WARNING` is logged.
   - The balanced sampler (`WeightedRandomSampler`) and `label_smoothing=0.1` prevent trivial majority-class collapse.
5. **Per-Generator-Family Diagnostics**:
   - During validation at the end of every epoch, `validate()` automatically outputs a breakdown across generator families (`cd`, `cf`, `cm`, `id`). Inspect these numbers before adding custom class weighting or focal loss adjustments.

---

## 7. Retraining vs. Resuming Reference Matrix

When planning experiments or modifying components, consult this canonical matrix:

| Code / Architectural Change | Retrain from Scratch? | Can Warm-Start from Previous Checkpoint? | Notes |
|---|---|---|---|
| **2.1 Vision LoRA** | **Yes** for the new adapter | LLM LoRA + `cls_head` can warm-start; vision LoRA starts fresh | Old checkpoints lack `vision_model.lora_*` tensors. |
| **2.2 Attention Pooling** | **Yes** for `cls_head` + pooling | Backbone LoRA reusable as init | Modifies patch aggregation module. |
| **2.3 Frequency Branch** | **Yes** (new module & input width) | Everything else warm-startable | `cls_head` input width changes from $2048 \to 2304$. |
| **2.4 Native Visual Tokens** | **Yes** for reasoning/LM path | Classification path largely independent | Sequence length and spatial representation change. |
| **2.5 Self-Consistency Loss** | **No** | **Yes** (can resume directly) | Additive loss term, zero structural weight changes. |
| **2.6 LM-Weight Curriculum** | **No** | **Yes** (can resume directly) | Optimization schedule adjustment only. |
| **2.7 DPO Preference Stage** | **No** | **Yes** (runs after Stage 2) | Additive alignment stage on top of joint `best.pth`. |
| **2.8 Focal Loss / Diagnostics** | **No** | **Yes** (can resume directly) | Loss function / logging change only. |

> [!IMPORTANT]
> Because changes 2.1 through 2.4 simultaneously modify both the classification-feature path and the reasoning-input path, **a full retrain from scratch (Stage 1 Cls Warm-up $\to$ Stage 2 Joint)** is strongly recommended for v2 to prevent inconsistent, partially-adapted representations.


python train.py --resume "./runs/intern_v2/checkpoints/ep001.pth"
