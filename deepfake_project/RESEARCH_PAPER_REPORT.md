# DeepfakeReasoningModel v2: Grounded Dual-Branch Multimodal Forensics with Self-Consistent Vision-Language Reasoning

**Technical Report — Complete Approach and Latest Results**

*Model: InternVL3-2B (InternViT-300M + Qwen2.5-1.5B) with dual LoRA, 2D-FFT frequency branch, attention pooling, telemetry grounding and bidirectional self-consistency.*

---

## Abstract

Deepfake detectors based on discriminative vision backbones reach high in-domain accuracy but behave as black boxes, transfer poorly to unseen generators, and provide no auditable evidence. Off-the-shelf multimodal LLMs can describe an image, but on forensic questions they rationalize post hoc: they invent plausible artifacts, or repeat a verdict supplied in the prompt, without measuring anything.

This report documents **DeepfakeReasoningModel v2**, a 2.18B-parameter framework that couples a quantitative classifier branch with a grounded reasoning branch inside one model. The classifier fuses three evidence streams: the InternViT `[CLS]` token, a learned attention-pooled summary over all 1,024 patch tokens, and a 256-dimensional embedding of the image's 2D FFT log-magnitude spectrum (input vector 2,304-d). The reasoning branch receives 256 native visual tokens plus a structured *forensic telemetry* prefix containing the verdict, the spectral HF/LF ratio and the top-8 attention hotspots mapped to facial regions. A bidirectional KL term aligns the classifier distribution with the language model's `<answer>` token distribution.

Training follows a four-stage curriculum (classifier warm-up → joint classification + reasoning → optional DPO → grounded reasoning fine-tuning), instantiated over two foundation lineages that differ only in their starting weights: an LR-fused InternVL3-2B and the vanilla InternVL3-2B.

**Verified results.** On the 4,000-sample balanced validation set, the grounded LR-fused model reaches **96.00% accuracy / AUC 0.9936** after three epochs; the grounded vanilla-base continuation reaches **95.23% / AUC 0.9903**. On the 52,266-sample HydraFake test benchmark, the best mean AUC across evaluated checkpoints is **0.9479**, with **cross-model AUC 0.9995** in that checkpoint, a best-observed **cross-forgery AUC of 0.9510**, and cross-domain AUC 0.8680; cross-forgery and cross-domain accuracy remain the weak points (76–84%). The newest grounded vanilla-base model is now fully benchmarked: it has the **best in-domain accuracy (93.38%, AUC 0.9872)** but trails its ungrounded predecessor on cross-forgery (78.13% vs 81.51%), giving **87.09% mean accuracy / AUC 0.9455**. Grounding is therefore not a free accuracy gain.

A throughput study run on 21–23 October 2026 measured the grounded stage at **2.74–2.98 s/step** with batch 4 × accumulation 8 (effective 32). Doubling the micro-batch to 8 (effective 64) did **not** raise throughput (6.20 s/step; 1.29 vs 1.34–1.46 samples/s) and halved the optimizer updates, degrading one-epoch validation accuracy from ~87.5% to 84.95%.

---

## 1. Problem Formulation and Contributions

### 1.1 Why pure discriminators are insufficient

A conventional detector minimizes

$$\min_\theta \; \frac{1}{N}\sum_{i=1}^{N}\mathcal{L}_{\text{CE}}\big(f_\theta(x_i), y_i\big), \qquad y_i \in \{0,1\},$$

and reports a probability. Three problems follow:

1. **Domain shift.** Spatial filters latch onto generator-specific traces (blend seams, specific upsampling grids), so accuracy collapses on unseen generators or heavy compression.
2. **No evidence.** A score of 0.92 is not an auditable finding; forensic use needs localization and a physical explanation.
3. **Pooling collapse.** Global average pooling dilutes small manipulated regions across ~1,000 unperturbed background patches.

### 1.2 Why a naive multimodal LLM is insufficient

Prompting a vision-language model with *"is this real or fake, explain why"* fails in three reproducible ways:

1. **Perceptual blindness.** The visual encoder and its 0.5× pixel-shuffle projection discard much of the high-frequency detail where generative traces live.
2. **Sycophancy.** When a hypothesis is present in the prompt, the model tends to confirm it and invent supporting artifacts.
3. **Reasoning–decision disconnect.** The generated trace can describe authentic properties and then emit the opposite `<answer>` tag.

### 1.3 Contributions

- **Dual-branch architecture in one model:** a high-capacity classifier (`[CLS]` + attention-pooled patches + FFT embedding) and an autoregressive reasoning branch sharing one vision encoder.
- **Quantitative telemetry conditioning:** spectral ratio and attention hotspots are computed from the classifier's own features and injected as a text prefix, so the LLM cites measurements rather than guessing.
- **Bidirectional self-consistency:** symmetric KL between classifier probabilities and the LLM's `real`/`fake` answer logits, making the two branches mutually accountable during training.
- **Two-lineage controlled comparison:** identical architecture, data, losses and schedule on an LR-fused versus vanilla foundation, isolating the effect of logical-reasoning pre-adaptation.
- **Measured engineering:** patch-token reuse, batched telemetry extraction, conditional attention-weight extraction and gradient checkpointing make a 2.18B model trainable on a single 24 GB RTX A5000, with step times documented per stage.

---

## 2. Approach

### 2.1 Backbone and token layout

| Component | Specification |
| :--- | :--- |
| Vision encoder | InternViT-300M, 448×448 input, 14×14 patches → 32×32 = **1,024 patch tokens** + 1 `[CLS]`, hidden 1,024 |
| Language model | Qwen2.5-1.5B, 28 layers, hidden **1,536**, vocab **151,674**, RoPE |
| Total parameters | **2,182,364,038** |
| Trainable parameters | **93,406,598 (4.28%)** with the classifier head active; 92,816,004 with `cls_head` frozen |

### 2.2 Dual LoRA and parameter budget

Two independent PEFT adapters are injected before the non-LoRA weights are frozen:

| Adapter | Target modules | Rank / α / dropout | Parameters |
| :--- | :--- | :--- | :---: |
| LLM LoRA | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | 64 / 128 / 0.05 | 73,859,072 |
| Vision LoRA | `qkv, proj, fc1, fc2` | 16 / 32 / 0.05 | 6,291,456 |
| `cls_head` | 2,304→256→2 MLP | — | 590,594 |
| `attn_pool` | 8-head cross-attention + LayerNorm | — | 4,201,472 |
| `freq_branch` | 4-stage CNN | — | 455,680 |
| `custom_projector` | 2-token fallback projector, 1,024→1,536→1,536 (×2) | — | 7,876,608 |
| `domain_router` | 1,024→128→4 compatibility stub | — | 131,716 |

Two notes on scope. `custom_projector` (two 1,024→1,536→1,536 MLPs) is only used as a fallback when the backbone exposes no `mlp1` projector; `domain_router` is a checkpoint-compatibility stub that is not called in the forward pass. Both are carried in checkpoints and in the adapter package, and their presence must not be read as active architectural components. The 4-tensor `cls_head` count is confirmed by the Stage-4 freeze log.

### 2.3 Frequency branch (2D FFT)

For each image, `dataset.compute_fft_features` resizes to 224×224, converts to `[0,1]`, applies a per-channel 2D FFT, shifts the zero frequency to the centre, takes the log-magnitude, and standardizes each channel:

$$F(u,v)=\sum_{x}\sum_{y} I(x,y)\,e^{-j2\pi(ux/H+vy/W)}, \qquad \tilde S_c(u,v)=\frac{\log\!\big(1+|F^{c}_{\text{shift}}(u,v)|\big)-\mu_c}{\sigma_c+\varepsilon}.$$

$\tilde S\in\mathbb{R}^{3\times224\times224}$ passes through four stride-2 convolutions with BatchNorm and GELU (3→32→64→128→256; 224→112→56→28→14), then adaptive average pooling, a linear layer and LayerNorm:

$$v_{\text{freq}}\in\mathbb{R}^{256}.$$

Generative pipelines leave spectral signatures — attenuation of high-frequency detail in diffusion outputs, periodic harmonics from transposed-convolution upsampling, resampling ridges after compression — that this branch can represent even when the RGB branch cannot see them.

### 2.4 Attention pooling

Uniform mean pooling over 1,024 patches is replaced by a single learned query that cross-attends over all patch tokens through an 8-head `nn.MultiheadAttention`:

$$\alpha=\operatorname{softmax}\!\left(\frac{q K^{\top}}{\sqrt{D}}\right)\in\mathbb{R}^{1\times1024}, \qquad h=\operatorname{LayerNorm}\!\left(\sum_{i}\alpha_i V_i\right)\in\mathbb{R}^{1024}.$$

The attention distribution $\alpha$ is not only a pooling weight; it is the localization signal exported as telemetry. Returned weights are the head-averaged attention, and they are requested only on the grounded joint path (`need_weights=True`), so classifier-only steps keep the fused fast path.

### 2.5 Hybrid classification head

$$v_{\text{fused}}=\big[\,v_{\text{cls}}\;\|\;v_{\text{pool}}\;\|\;v_{\text{freq}}\,\big]\in\mathbb{R}^{1024+1024+256=2304},$$

$$z=\operatorname{Linear}_{256\to2}\Big(\operatorname{Dropout}_{0.3}\big(\operatorname{GELU}(\operatorname{Linear}_{2304\to256}(v_{\text{fused}}))\big)\Big), \qquad P_{\text{cls}}=\operatorname{softmax}(z).$$

The hidden width is 256 (not 512), and dropout is applied before the output projection. The head is deliberately shallow: its inputs are already 2,304 discriminative dimensions, and a wide head was not needed to reach the reported validation AUC.

### 2.6 Native visual tokens for the reasoning branch

The 1,024 patch tokens are downsampled by InternVL's 0.5× pixel shuffle (1,024→256 tokens, 1,024→4,096 channels) and projected by `mlp1` (LayerNorm → 4,096→1,536 → GELU → 1,536→1,536):

$$V_{\text{vis}}\in\mathbb{R}^{256\times1536}.$$

In training the vision encoder runs once and the patch tokens are reused for both branches; at inference the backbone's `extract_feature` path is used. This removes the duplicate ViT pass that a naive implementation performs for the classifier and the LLM.

### 2.7 Grounded telemetry conditioning

Training and inference share one prompt builder (`model.build_grounding_prompt`) so the token structure cannot drift:

```text
[Forensic Telemetry]
- Statistical read-out (verify against what you see, do not blindly repeat): FAKE (dataset label)
- Frequency measurement: HF/LF log-magnitude energy ratio=-0.23 (lower ratio = smoother image with less high-frequency detail)
- Attention hotspots: Nose & Cheek region, Eyes & Facial midline

Weigh the telemetry numbers against the visual evidence, then explain using
<fast>, <planning>, <reasoning>, <reflection> and <conclusion> tags only.
Finish with exactly <answer>real</answer> or <answer>fake</answer> and nothing after it.
```

The LM input is

$$\mathbf{X}=\big[\,V_{\text{vis}}\;\|\;T_{\text{tele}}\;\|\;T_{\text{reasoning}}\,\big],$$

with labels $-100$ on both the visual and telemetry positions, so the model is conditioned on the measurement but never trained to generate it. The telemetry block is capped at 160 tokens; the reasoning target at 512 tokens.

**Verdict source.** During training the verdict word is the ground-truth label with confidence string `"dataset label"`. During inference it is the classifier's argmax and its softmax confidence. This is intentional — it teaches the LLM to explain a decision rather than to make one — but it creates a train/inference mismatch whenever the classifier is wrong, discussed in §8. `inference.py --force_verdict {REAL,FAKE}` exists to audit that failure mode by deliberately flipping the telemetry verdict.

**Hotspot mapping.** Top-8 attention indices are mapped to a 32×32 grid and then to six regions: `col<5` or `col≥27` → peripheral/ear-hair seams (checked first); `row<8` forehead/hairline; `8≤row<14` eyes/midline; `14≤row<20` nose/cheek; `20≤row<26` mouth/jaw; otherwise chin/neck. The two most-hit regions are emitted.

### 2.8 Spectral ratio

Using cached radial masks on the standardized log-magnitude map, with $H=W=224$ and centre $(c_y,c_x)$:

$$\Omega_{\text{low}}=\{\,\lVert(u,v)-(c_y,c_x)\rVert<0.15H\,\},\qquad \Omega_{\text{high}}=\{\,\lVert(u,v)-(c_y,c_x)\rVert>0.35H\,\},$$

$$R=\frac{\operatorname{mean}_{(u,v)\in\Omega_{\text{high}}}\tilde S(u,v)}{\operatorname{mean}_{(u,v)\in\Omega_{\text{low}}}\tilde S(u,v)+10^{-6}}.$$

Because $\tilde S$ is zero-mean per channel, $R$ is typically **negative** (observed ≈ −0.21 to −0.28 on the smoke-test faces); it is a relative spectral-slope read-out, not an energy fraction in $[0,1]$. The pipeline does not threshold it — the number is passed to the LLM as evidence. No hard threshold (for example "$R<0.65$ means diffusion") is implemented, and none has been calibrated.

### 2.9 Self-consistency loss

Let $t_{\text{ans}}$ be the index of the `real`/`fake` token inside the reasoning target and $N_{\text{prefix}}=N_{\text{vis}}+N_{\text{tele}}$. The LM predicts that token at position $N_{\text{prefix}}+t_{\text{ans}}-1$. Two logits are selected, `"real"` (id 7951) and `"fake"` (id 30570):

$$z_{\text{lm}}=[\,z[7951],\,z[30570]\,],\qquad P_{\text{lm}}=\operatorname{softmax}(z_{\text{lm}}),$$

$$\mathcal{L}_{\text{cons}}=\tfrac12\Big[D_{\mathrm{KL}}\big(P_{\text{lm}}\,\|\,P_{\text{cls}}^{\text{detach}}\big)+D_{\mathrm{KL}}\big(P_{\text{cls}}\,\|\,P_{\text{lm}}^{\text{detach}}\big)\Big].$$

The term is computed only for samples whose answer position is valid, and targets are detached so the classifier is not dragged by the LM's vocabulary bias.

### 2.10 Total objective

$$\mathcal{L}=\mathcal{L}_{\text{cls}}+\lambda_{\text{lm}}(e)\,\mathcal{L}_{\text{lm}}+\lambda_{\text{cons}}\,\mathcal{L}_{\text{cons}},$$

with $\mathcal{L}_{\text{cls}}$ = cross-entropy with label smoothing 0.1 (a focal-loss option exists but is off in every reported run), $\lambda_{\text{cons}}=0.05$, and $\lambda_{\text{lm}}$ ramped linearly from 0.30 to 0.60 across joint epochs.

---

## 3. Training Curriculum

```
Stage 0  Foundation preparation
         InternVL3-2B + LoRA on a logical-reasoning dataset → merged into
         internvl3_2b_lr_fused   (Lineage A only; weights fixed before this work)
              │
Stage 1  Classifier warm-up (3 epochs, 12,080 steps/epoch)
         Train: vision LoRA, attn_pool, freq_branch, cls_head.  LLM idle.
              │
Stage 2  Joint classification + reasoning (3 epochs, 8,765 paired steps/epoch)
         Train: all LoRA + custom heads.  Loss: cls + λ_lm·lm + 0.05·cons
              │
      ┌───────┴────────┐
      │                │
Stage 3 DPO        Stage 4 Grounded SFT
(2 epochs)         (3 epochs, cls_head frozen)
```

### 3.1 Stage 0 — foundations

Two starting points are used. **Lineage A** starts from `models/internvl3_2b_lr_fused`, a checkpoint in which an LoRA adapter trained on a logical-reasoning corpus was merged into the base weights,

$$W_{\text{fused}}=W_0+\frac{\alpha}{r}B_{\text{LR}}A_{\text{LR}},$$

and then saved as a standalone Hugging Face checkpoint. **Lineage B** starts from vanilla `models/InternVL3-2B`. The Stage-0 merge itself is external to this repository: no LR training logs, data manifest or merge script are archived here, so that step is taken as given and is not independently reproducible from this report.

### 3.2 Stage 1 — classifier warm-up

- **Active:** vision LoRA (r=16), `attn_pool`, `freq_branch`, `cls_head`.
- **Frozen:** the entire Qwen2.5-1.5B stack; no LM forward or backward pass.
- **Data:** `all.json`, 48,320 images, 12,080 steps/epoch at batch 4.
- **Measured:** ≈0.69 s/step, ≈2 h 20 m per epoch.
- **Purpose:** produce a stable classifier and a meaningful attention/hotspot map before the LM is engaged.

### 3.3 Stage 2 — joint training

Each step pairs one classification batch (`all.json`) with one reasoning batch (`sft_36k.json`), giving 8,765 paired steps per epoch and 1,096 optimizer updates at batch 4 × accumulation 8. The LM loss weight is scheduled by joint-epoch index:

$$\lambda_{\text{lm}}(e)=\lambda_{\text{start}}+\frac{e-e_{\text{cls}}-1}{\max(1,e_{\text{joint}}-1)}\big(\lambda_{\text{end}}-\lambda_{\text{start}}\big),\qquad \lambda_{\text{start}}=0.30,\ \lambda_{\text{end}}=0.60.$$

Reasoning targets use the six-tag template `<fast> … </fast> <planning> … </planning> <reasoning> … </reasoning> <reflection> … </reflection> <conclusion> … </conclusion> <answer>real|fake</answer>`.

### 3.4 Stage 3 — Direct Preference Optimization

3,480 chosen/rejected trace pairs (`mipo_3k.json`), 870 steps/epoch (batch 4 × accumulation 8), $\beta=0.1$, AdamW at $1\times10^{-6}$, two epochs:

$$\mathcal{L}_{\text{DPO}}=-\mathbb{E}\Big[\log\sigma\Big(\beta\log\tfrac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}-\beta\log\tfrac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\Big)\Big].$$

Measured: 1 h 41 m/epoch (LR-fused, 3 h 22 m total) and 1 h 10 m/epoch (vanilla, 2 h 20 m total). Reward accuracy rose to 96.61% (LR-fused) and 91.03% (vanilla).

### 3.5 Stage 4 — grounded reasoning fine-tuning

- **Init:** `intern_fused/ep006.pth` (Lineage A) or `intern_v2/ep006.pth` (Lineage B); weights only, fresh optimizer and schedule.
- **Freeze:** `cls_head` (4 tensors) to protect the classifier at its peak.
- **Train:** LLM LoRA + vision LoRA + `attn_pool`/`freq_branch`/`custom_projector`/`domain_router` (43 custom tensors + 584 LoRA tensors).
- **Grounding:** on; `ground_flip_prob = 0.0`.
- **Epochs:** 3, 8,765 steps each, 1,096 updates.

### 3.6 Optimizer and schedule (all stages)

AdamW; LoRA parameters at $5\times10^{-5}$; custom heads at $1\times10^{-4}$ (2× the LoRA rate); weight decay 0.01; cosine decay to zero after a 5% linear warm-up; gradient-norm clipping at 1.0; bf16 autocast; gradient checkpointing on the LM with `enable_input_require_grads()`. Validation runs every epoch; `best.pth` is selected by validation AUC.

### 3.7 The two lineages

| | **Lineage A — LR-fused** | **Lineage B — vanilla** |
| :--- | :--- | :--- |
| Foundation | `internvl3_2b_lr_fused` | `InternVL3-2B` |
| Stage 1–2 output | `runs/intern_fused` → `ep006.pth` (val AUC 0.9915) | `runs/intern_v2` → `ep006.pth` (val AUC 0.9848) |
| Post-stage α | DPO → `runs/intern_fused_dpo` | DPO → `runs/intern_v2_dpo` |
| Post-stage β | Grounded → `runs/intern_fused_ground` (**deployed checkpoint**) | Grounded → `runs/intern_v2_ground` (**latest trained**) |
| Stage-4 wall time | 8 h 27 m – 14 h 02 m per epoch (host load variance) | 6 h 41 m – 7 h 15 m per epoch (21 h 11 m total) |

The lineages are identical through epoch 6 and diverge afterwards, which is what makes the comparison informative: any difference at Stage 4 is attributable to the foundation weights (and to host-level variance, which is not negligible — see §8).

---

## 4. Data and Evaluation Protocol

| Pool | Samples | Notes |
| :--- | ---: | :--- |
| Classification train (`all.json`) | 48,320 | 48,320 unique images; bare or rich labels allowed |
| Reasoning SFT (`sft_36k.json`) | 35,063 | 36,750 unique images; 1,687 skipped for length |
| Preference pairs (`mipo_3k.json`) | 3,480 | chosen/rejected traces |
| Validation | 4,000 | balanced 2,000 real / 2,000 fake |
| Test | 52,266 | held-out; split below |

Test composition (HydraFake): **ID** 12,819 (FaceForensics++, Hallo2, Midjourney, StyleGAN, facevid2vid); **CM** 11,249 (Adobe Firefly, Flux 1.1 Pro, HART, Infinity, MAGI, StarryAI); **CF** 12,730 (CodeFormer, FaceAdapter, IC-Light, InfiniteYou, PuLID, StarGAN v2); **CD** 15,468 (FFIW, DeepFaceLab, Dreamina, GPT-4o, Hailuo, InfiniteYou). The JSON files list 53,272 entries; the loader merges 1,006 duplicates to the 52,266 evaluated samples. Splits are evaluated with `eval.py`, threshold 0.5, and optionally `--calibrate` for per-split Youden/F1 threshold search. Means are **macro-averages over the four splits**, not sample-weighted. Hardware: one NVIDIA RTX A5000 (24 GB), bf16.

---

## 5. Results

### 5.1 Validation dynamics — LR-fused grounded (Lineage A)

From `runs/intern_fused_ground/logs/epochs.csv` (epoch 2 was re-run after a restart; the second row is authoritative):

| Epoch | $\lambda_{\text{lm}}$ | Train cls | Train LM | Cons | Train acc | Val loss | Val acc | Val AUC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.30 | 0.3117 | 0.9222 | 0.0424 | 93.18% | 0.2934 | 95.05% | 0.992431 |
| 2 | 0.45 | 0.2909 | 0.7811 | 0.0246 | 94.53% | 0.2759 | 95.80% | 0.993507 |
| 3 | 0.60 | 0.2786 | 0.7517 | 0.0200 | 95.25% | **0.2736** | **96.00%** | **0.993647** |

Total reported loss rises (0.590 → 0.731) even as every quality signal improves, because the scheduled LM weight doubles and the LM term is larger in magnitude. The deployed `best.pth` is the epoch-3 checkpoint (checkpoint metadata: `epoch=3`, `best_auc=0.993647`).

### 5.2 Validation dynamics — grounded vanilla base (Lineage B)

From `runs/intern_v2_ground/logs/epochs.csv`:

| Epoch | $\lambda_{\text{lm}}$ | Train cls | Train LM | Cons | Train acc | Val loss | Val acc | Val AUC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.30 | 0.3094 | 0.7284 | 0.0122 | 93.16% | 0.3353 | 93.08% | 0.986410 |
| 2 | 0.45 | 0.2860 | 0.6975 | 0.0078 | 94.71% | 0.2887 | 95.10% | 0.989919 |
| 3 | 0.60 | 0.2761 | 0.6782 | 0.0064 | 95.37% | 0.2853 | **95.23%** | **0.990270** |

The shape is identical to Lineage A: classifier loss and consistency loss fall monotonically while the scheduled total rises. The ~0.8 pp accuracy / 0.0034 AUC gap to Lineage A is consistent with the LR pre-adaptation, but with one run per lineage and a 4,000-sample validation set it is not statistically established.

### 5.3 Held-out benchmark (52,266 samples)

Verified from archived `eval_*.log` files. Each cell is **accuracy / AUC**; the last column is macro-mean accuracy / F1 / AUC. The `intern_v2_ground` row is the newest measurement (31 Oct 2026, 00:23–01:14): checkpoint `best.pth` loaded as epoch 3 with missing=0 / unexpected=0, 52,266 samples at batch 8 in bf16, **51 min 20 s** wall time on the A5000.

| Checkpoint (stage) | ID 12,819 | CM 11,249 | CF 12,730 | CD 15,468 | Mean |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `intern_v2` `ep005.pth` — S2, ungrounded vanilla | 92.42 / 0.9823 | **99.17 / 0.9995** | 81.51 / 0.9418 | 78.11 / 0.8680 | 87.80 / 0.8659 / **0.9479** |
| `intern_v2_ground` best — S4, grounded vanilla | **93.38 / 0.9872** | 99.08 / 0.9995 | 78.13 / 0.9327 | 77.77 / 0.8626 | 87.09 / 0.8524 / 0.9455 |
| `intern_fused_ground` best — S4, LR-fused | 92.13 / 0.9881 | 98.88 / 0.9994 | **82.66 / 0.9507** | 76.60 / 0.8480 | 87.57 / 0.8655 / 0.9465 |
| `intern_fused_dpo` best — S3, LR-fused | 92.08 / 0.9852 | 98.48 / 0.9993 | 80.42 / 0.9510 | 75.96 / 0.8503 | 86.74 / 0.8519 / 0.9465 |
| `intern_v2_dpo` best — S3, vanilla | 89.73 / 0.9788 | 98.91 / 0.9996 | 76.17 / 0.9235 | 76.62 / 0.8675 | 85.36 / 0.8268 / 0.9423 |

Reading the table:

- **Cross-model detection is essentially solved** on this benchmark: 98.5–99.2% accuracy and AUC 0.9993–0.9996 for every checkpoint, with no sign that reasoning or DPO disturbs it.
- **Cross-forgery** is the regime where the design pays off for the LR-fused lineage: the grounded LR-fused model has the best accuracy of the five (82.66%), +1.2 pp over the vanilla Stage-2 model and +2.2 pp over its LR-fused DPO sibling; its AUC (0.9507) is within 0.0003 of the best observed (0.9510). On the vanilla lineage, however, grounding moves the other way (81.51% → 78.13%).
- **Grounding trades in-domain for cross-forgery on the vanilla base.** Relative to its Stage-2 predecessor, `intern_v2_ground` gains 0.96 pp on ID (93.38 vs 92.42) but loses 3.38 pp on CF (78.13 vs 81.51) and 0.71 pp on the macro mean (87.09 vs 87.80); mean AUC falls from 0.9479 to 0.9455. The same trade is not visible in validation, where the grounded model is 0.4 pp higher — the 4,000-sample balanced set does not represent the CF/CD distributions.
- **Cross-domain** is the weakest regime (75.96–78.11% accuracy, AUC 0.848–0.868) and does not improve with grounding on either lineage; compression and in-the-wild post-processing remain the dominant failure source.
- **DPO does not improve classification.** For the vanilla lineage, the DPO checkpoint is below its Stage-2 counterpart on ID (89.73 vs 92.42) and CF (76.17 vs 81.51); DPO's purpose is trace quality, which this benchmark does not measure.

The `intern_v2` row is the checkpoint the archived evaluation actually loaded: `checkpoints/ep005.pth` (metadata `epoch=5`, `best_auc=0.989061`). The directory's current `best.pth` was later overwritten by a second run (epoch 4, AUC 0.991403) and has no 52k evaluation; it is therefore not labelled "best" in this table.

A metrics block in `config.yaml` records slightly different accuracies (ID 92.01, CM 99.16, CF 80.93, CD 77.85, mean 87.49) plus calibrated CF/CD numbers. Those values are not reproducible from the archived logs, while the AUCs match the `intern_v2` log exactly; the log-derived table above is therefore the reference, and the config block should be treated as a stale/derived summary.

### 5.4 Threshold calibration

Calibration has been run for the grounded vanilla checkpoint and both DPO checkpoints (`--calibrate`). Per-split optima and resulting accuracy:

| Checkpoint | ID T* / acc | CM T* / acc | CF T* / acc | CD T* / acc | Calibrated mean acc / F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `intern_v2_ground` | 0.14 / 95.00% | 0.22 / 99.34% | 0.06 / 86.27% | 0.09 / 77.39% | 89.50% / 89.62% |
| `intern_fused_dpo` | 0.11 / 93.92% | 0.16 / 99.07% | 0.06 / 88.85% | 0.09 / 76.56% | **89.60% / 90.27%** |
| `intern_v2_dpo` | 0.09 / 94.07% | 0.13 / 99.42% | 0.06 / 85.76% | 0.07 / 78.16% | 89.35% / 89.50% |

The optimal thresholds sit far below 0.5, and calibration buys most on CF (+8.4 pp for the LR-fused DPO model, +9.6 pp for the vanilla DPO model, +8.1 pp for the grounded vanilla model). This is a **post-hoc diagnostic**, not a deployable result: the thresholds are tuned and evaluated on the same split. The fused grounded checkpoint has no archived calibration run.

### 5.5 Latest-model throughput and batch-size study (21–23 Oct 2026)

Four `gpu_util_demo` runs probed the grounded stage's cost profile. All four start from random heads (`epochs_cls=0`, no `weights_from` line in their logs), train one grounded epoch, and save a checkpoint; they are throughput probes, **not** part of the final model.

| Run | Micro-batch × accum | Effective batch | Steps | s/step | Epoch wall time | Updates | Val acc | Val AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `…21_162100` | 4 × 8 | 32 | — | — | aborted before step 1 | — | — | — |
| `…21_162502` | 4 × 8 | 32 | 8,765 | 2.98 | 7 h 14 m 43 s | 1,096 | 87.55% | 0.9627 |
| `…22_053241` | 4 × 8 | 32 | 8,765 | 2.74 | 6 h 40 m 50 s | 1,096 | 87.13% | 0.9623 |
| `…23_053405` | 8 × 8 | 64 | 4,382 | 6.20 | 7 h 32 m 58 s | 548 | 84.95% | 0.9431 |

Derived throughput: **1.34, 1.46 and 1.29 samples/s** respectively. Two identical 4×8 runs differ by ~8% in wall time, which sets the noise floor. Against that floor, batch 8 gives no throughput gain, doubles the per-step latency, halves the optimizer updates, and loses ~2.2–2.6 pp validation accuracy after one epoch — the expected consequence of halving updates rather than of the larger batch per se. The current 4 × 8 configuration is therefore retained. The first launch (`162100`) terminated before its first logged step with no exception in stderr; it is recorded here for completeness only.

### 5.6 Qualitative behaviour

**Correct authentic case** (`intern_fused_ground` best, `test_images/real/161022.jpg`, classifier REAL 95.3%):

```text
  Classifier verdict: REAL  (conf 95.3%)
  Anomaly Hotspots  : Nose & Cheek region, Eyes & Facial midline
  Frequency Domain  : HF/LF log-magnitude energy ratio=-0.23

  [Quick take]  At first glance, the image appears authentic with no obvious signs of manipulation.
  [Reasoning]   … the facial structure is proportionate and symmetrical, with consistent skin tone
                and natural expression … the edges and textures appear smooth and realistic without
                any artificial artifacts … lighting and shadows are physically plausible …
  [Conclusion]  … the image does not display any significant signs of being fake and is most likely authentic.
```

**Failure case** (`test_images/real/160932.png`, classifier FAKE 94.9%). The trace still invents evidence for the wrong verdict — the sycophancy failure the grounding is meant to reduce:

```text
  Classifier verdict: FAKE  (conf 94.9%)
  [Reasoning]  … edges around the hairline and jawline appear unnaturally sharp … a noticeable
               blending boundary where the face meets the background … skin texture … lacks the
               fine-grained detail expected …
  [Conclusion] … strongly indicate that the image is not a genuine photograph but rather an AI-generated one.
```

On the 18-image real-only smoke set, the LR-fused grounded model labels 14 REAL / 4 FAKE (77.8% on authentic images). This is a diagnostic on 18 hand-picked files, not a balanced accuracy estimate. The latest vanilla-base model (`intern_v2_ground`) has a 19 Oct inference log that covers only **two** of the 18 images before the run stopped; one (161022) is correct at 94.8% confidence and the other (160932) is a false positive at 95.1%. No accuracy is claimed from that smoke run; the model's benchmark accuracy is reported in §5.3. That false positive is consistent with the benchmark's FAKE-leaning score bias: on cross-forgery the grounded model predicts 9,118 fakes against 6,364 ground-truth fakes, and calibration at T*≈0.06–0.14 recovers 8.1 pp there (§5.4).

Tag coverage differs between checkpoints: the LR-fused model emits `<fast>`, `<planning>`, `<reasoning>`, `<reflection>`, `<conclusion>` on most of its 18 samples, while both logged vanilla-grounded samples emit only fast/reasoning/conclusion. Both keep the telemetry structure, but the richer trace is a qualitative advantage of the LR-fused lineage.

---

## 6. Engineering and Compute

| Mechanism | Implementation | Effect |
| :--- | :--- | :--- |
| Patch-token reuse | One vision pass; classifier and native visual tokens both consume the same patch tokens (`_get_native_visual_tokens_from_patches`) | Removes a second ViT forward per joint step |
| Conditional attention weights | `need_weights=True` only when grounded telemetry is needed; otherwise fused attention path | Keeps classifier-only steps on the fast path |
| Batched telemetry | `batch_describe_frequency` / `batch_regions_from_topk`; one CPU transfer per batch; radial masks cached | Replaces per-sample `.item()`/`.tolist()` syncs |
| Gradient checkpointing | LM checkpointing + `enable_input_require_grads()` | Fits 2.18B model + 256 visual tokens + telemetry in 24 GB |
| bf16 autocast | Whole forward/backward in bfloat16, no GradScaler | Stable training without loss scaling |
| Checkpoint portability | Whole-model `best.pth` → `extract_adapters.py` (LLM adapter, vision adapter, custom heads) with round-trip verification | 179 MB deployable package instead of ~4.4 GB |

Measured step times on the RTX A5000 (eager attention — see limitation 4 in §8):

| Stage | Steps/epoch | s/step | Epoch wall time |
| :--- | :---: | :---: | :---: |
| Stage 1 classifier warm-up | 12,080 | ≈0.69 | ≈2 h 20 m |
| Stage 2 joint, ungrounded | 8,765 | 2.32–2.46 | 5 h 39 m – 5 h 59 m |
| Stage 4 grounded, vanilla | 8,765 | 2.84–2.90 | 6 h 54 m – 7 h 03 m |
| Stage 4 grounded, LR-fused | 8,765 | 3.47–5.77 | 8 h 27 m – 14 h 02 m |
| DPO | 870 | 4.81–6.98 | 1 h 10 m – 1 h 41 m |

The fused-base Stage-4 spread is host-load variance, not a property of the recipe: the same configuration produced 8 h 27 m and 14 h 02 m epochs on different days. The one-epoch probes in §5.5 (2.74–2.98 s/step at batch 4) run on fresh heads and are not directly comparable to the 2.84–2.90 s/step of the resumed `intern_v2_ground` run.

---

## 7. What the System Produces

`inference.py` returns, per image:

- `verdict` and `confidence` from the classifier — **authoritative**; the LLM answer is never allowed to override it;
- `telemetry`: hotspot regions and the spectral ratio actually fed to the LLM;
- `reasoning`: parsed `<fast>`, `<planning>`, `<reasoning>`, `<reflection>`, `<conclusion>` blocks;
- an optional audit mode that flips the telemetry verdict (`--force_verdict`) to test whether the reasoning parrots the prompt.

Decoding defaults to **greedy** (`do_sample=False`) with a `</answer>` stopping criterion. The `--temperature`/`--top_p` flags are accepted but inert unless `--do_sample` is passed, so the defaults do not produce nucleus sampling. Repetition penalty defaults to 1.05.

---

## 8. Limitations

1. **Grounding does not improve the aggregate benchmark.** The grounded vanilla model is fully benchmarked (87.09% / 0.9455), but the ungrounded Stage-2 checkpoint remains higher on mean accuracy (87.80%) and AUC (0.9479); the grounded model's +0.96 pp ID gain is offset by a −3.38 pp CF loss, and validation does not predict that trade. One evaluation per checkpoint, no confidence intervals.
2. **No baseline comparison.** The report contains no external detector or vanilla-LLM baseline under the same protocol, so "state-of-the-art" is not supported and is not claimed.
3. **Single run per configuration, 4,000-sample validation.** Lineage differences (~0.8 pp) and epoch-to-epoch deltas are within plausible run-to-run variation; no confidence intervals or seeds are recorded (the training script does not set a seed).
4. **Eager attention in every recorded run.** `attn_implementation="sdpa"` is rejected by the InternVL chat wrapper and silently falls back to `eager`; FlashAttention-2 is not installed. All timings above therefore reflect eager attention, not a fused-SDPA kernel.
5. **Train/inference verdict mismatch.** Grounding is trained on ground-truth verdicts but applied to classifier predictions. When the classifier is wrong, the LLM receives a wrong verdict and tends to rationalise it (observed in the 160932 failure case). The `--force_verdict` audit exists but no audit results are archived.
6. **Telemetry is descriptive, not calibrated.** The spectral ratio has no validated threshold or direction per generator family; values are zero-mean standardized means and are typically negative.
7. **Reasoning is not quantitatively evaluated.** No faithfulness, hallucination-rate or human-preference metric is computed; §5.6 is illustrative.
8. **Weak regimes persist.** CD accuracy stays at 76–78% and CF F1 at 0.76–0.79; threshold calibration helps CF but is not deployable as computed (tuned and evaluated on the same split).
9. **Dead weight in the checkpoint.** `domain_router` is never called and `custom_projector` is only a fallback, yet both are saved, exported and partly trained; they add ~8 M parameters and confound parameter accounting.
10. **Config drift.** `config.yaml` carries a metrics block that disagrees with the archived logs; only log-derived numbers are reported here.

---

## 9. Reproducibility and Artifacts

### 9.1 Training

```bash
# Lineage A: classifier warm-up + joint training (LR-fused base)
python -u train.py --config config.yaml \
    --model_path ./models/internvl3_2b_lr_fused \
    --tokenizer_path ./models/internvl3_2b_lr_fused \
    --batch_size 4 --grad_accum 8 \
    --epochs_cls 3 --epochs_joint 3 \
    --output_dir ./runs/intern_fused

# Lineage B: same recipe on the vanilla base
python -u train.py --config config.yaml \
    --model_path ./models/InternVL3-2B \
    --tokenizer_path ./models/InternVL3-2B \
    --batch_size 4 --grad_accum 8 \
    --epochs_cls 3 --epochs_joint 3 \
    --output_dir ./runs/intern_v2

# Stage 3: DPO (either lineage; 3,480 pairs, beta 0.1, lr 1e-6)
python -u train_dpo.py --config config.yaml \
    --model_path ./models/internvl3_2b_lr_fused \
    --checkpoint ./runs/intern_fused/checkpoints/best.pth \
    --dpo_epochs 2 --output_dir ./runs/intern_fused_dpo

# Stage 4: grounded fine-tuning, classifier frozen (Lineage A)
python -u train.py --config config.yaml \
    --weights_from ./runs/intern_fused/checkpoints/ep006.pth \
    --epochs_cls 0 --epochs_joint 3 --freeze_cls_head \
    --output_dir ./runs/intern_fused_ground

# Stage 4 on the vanilla lineage (latest trained model)
#   equivalent one-liner: run_intern_v2_ground.bat
python -u train.py --config config_intern_v2_ground.yaml \
    --model_path ./models/InternVL3-2B \
    --tokenizer_path ./models/InternVL3-2B \
    --weights_from ./runs/intern_v2/checkpoints/ep006.pth \
    --output_dir ./runs/intern_v2_ground \
    --epochs_cls 0 --epochs_joint 3 --freeze_cls_head
```

### 9.2 Evaluation

```bash
python eval.py --dataset_root ./datasets/hydrafake \
    --json_root ./datasets/hydrafake/jsons \
    --model_path ./models/internvl3_2b_lr_fused \
    --checkpoint ./runs/intern_fused_ground/checkpoints/best.pth \
    --batch_size 8 --num_workers 8            # add --calibrate for T* search
```

### 9.3 Deployable artifacts

`best.pth` is a whole-model state dict (backbone + 584 LoRA tensors + 59 custom-head tensors, ~4.36 GB), not an adapter file. `extract_adapters.py` produces the porting unit:

| Artifact | Contents | Size / params |
| :--- | :--- | :--- |
| `runs/intern_fused_ground/adapters/` | LLM adapter (392 tensors, 73,859,072), vision adapter (192 tensors, 6,291,456), custom heads (59 tensors, 13,256,070 params + 964 BatchNorm buffers) | ~179 MB, attach to `models/internvl3_2b_lr_fused` |
| `runs/intern_v2_ground/adapters/` | Same structure, extracted against `models/InternVL3-2B` | ~179 MB, attach to the **vanilla** base |

Neither package reproduces classifier scores without its matching base and the custom heads; the adapters must travel together. Adapter files passed exact safetensors round-trip and clean-base attach checks (tensor-for-tensor); the 59 custom-head tensors passed round-trip but are loaded via `strict=False`, which the extractor does not exercise end to end. Neither `eval.py` nor `inference.py` accepts an adapter directory, so adapter-only deployment requires a loader not included in the repository. Full artifact details, hashes and the DPO package's missing-heads caveat are documented in [RESEARCH_PAPER_REPORT_VANILLA_ADAPTER_ONLY.md](RESEARCH_PAPER_REPORT_VANILLA_ADAPTER_ONLY.md).

```bash
python extract_adapters.py \
    --checkpoint ./runs/intern_v2_ground/checkpoints/best.pth \
    --output_dir ./runs/intern_v2_ground/adapters \
    --base_model ./models/InternVL3-2B
```

### 9.4 Inference

```bash
python inference.py --config config.yaml \
    --model_path ./models/internvl3_2b_lr_fused \
    --checkpoint ./runs/intern_fused_ground/checkpoints/best.pth \
    --grounded_reasoning \
    --images test_images/real/*
```

### 9.5 Evidence map

| Claim | Source |
| :--- | :--- |
| Architecture, losses, telemetry format | `model.py`, `train.py`, `dataset.py` |
| Parameter counts | run logs (`launch_stdout.log`, `…/logs/train.log`), `extract_adapters.py` report |
| Validation dynamics | `runs/intern_fused_ground/logs/epochs.csv`, `runs/intern_v2_ground/logs/epochs.csv` |
| Benchmark table | `runs/intern_v2/logs/eval_best.log`, `runs/intern_v2_ground/logs/eval_best.log`, `runs/intern_fused_ground/logs/eval_best.log`, `runs/intern_fused_dpo/logs/eval_dpo_best.log`, `runs/intern_v2_dpo/logs/eval_dpo_best.log` |
| Calibration | `runs/intern_v2_ground/logs/eval_best.log` and the two `eval_dpo_best.log` files |
| Throughput / batch study | `runs/gpu_util_demo_20261021_162100`, `…162502`, `…20261022_053241`, `…20261023_053405` |
| Qualitative examples | `runs/inference_logs/inference_best_20261014_210419.log`, `…20261019_203120.log` |
| DPO details | `runs/*_dpo/logs/dpo_epochs.csv`, `dpo_train.log` |

---

## 10. Conclusion

The approach is sound and the evidence is consistent across two independent lineages: fusing a frequency branch and attention pooling into a classifier, then conditioning a LoRA-tuned LLM on that classifier's measurements, produces a model that is both accurate on in-domain and cross-model data (AUC 0.99+ / 0.9995) and able to explain itself with specific spectral and spatial evidence. The grounded stage costs ≈7 h per epoch for three epochs on one A5000 and does not harm classification; the LR-fused foundation adds a small edge over the vanilla base (0.8 pp validation accuracy, consistent in direction but not yet replicated).

The results also delimit the claim. Cross-forgery and cross-domain performance (AUC 0.95 / 0.86, accuracy 83% / 78%) remain the bottleneck, and the newest grounded model is fully benchmarked but **not better overall**: it buys 0.96 pp in-domain at the cost of 3.38 pp cross-forgery. Reasoning quality is still supported only by qualitative evidence. The next steps are therefore a faithfulness/hallucination evaluation, a telemetry calibration per generator family, and a targeted investigation of why grounding hurts cross-forgery, rather than more architecture.

---

*Compiled from repository source code, training logs, evaluation logs and adapter extraction reports. All numbers in §5 are traceable to the files listed in §9.5; values that could not be traced are called out and excluded.*
