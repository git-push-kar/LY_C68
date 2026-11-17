# RP build manifest

Files required to rebuild `DEEP_RP_JOURNAL.md` / `.html`, with the role each plays in the manuscript. Paths are relative to `deepfake_project/`; `build_rp_journal.py` prints any input it cannot find and records it in the Missing inputs section below.

| # | File | Present | Provides | Used for |
| --- | --- | :---: | --- | --- |
| 1 | `runs/timing/vanilla_calib/eval_calib.log` | yes | deployed checkpoint, four-split evaluation + calibration (2026-11-17) | Table 2 deployed row, Table 3 calibration, Fig. 4, Fig. 5 |
| 2 | `runs/intern_v2/logs/eval_best.log` | yes | Stage-2 parent checkpoint (intern_v2 ep005), archived 52,266-sample evaluation | Table 2 Stage-2 row |
| 3 | `runs/timing/epoch3/eval_best.log` | yes | LR-fused comparison checkpoint, four-split evaluation (2026-11-16 22:52-23:43) | Table 2 fused row |
| 4 | `runs/timing/epoch3_calib/eval_best.log` | yes | LR-fused comparison, evaluation + calibration (2026-11-17 01:30-02:50) | Table 3 fused row |
| 5 | `runs/intern_fused_dpo/logs/eval_dpo_best.log` | yes | LR-fused DPO checkpoint, four-split evaluation | Table 2 DPO row (fused) |
| 6 | `runs/intern_v2_dpo/logs/eval_dpo_best.log` | yes | vanilla DPO checkpoint, four-split evaluation | Table 2 DPO row (vanilla) |
| 7 | `runs/intern_v2_ground/logs/epochs.csv` | yes | deployed run, per-epoch training/validation history | Table 1, Fig. 3, Fig. 11 |
| 8 | `runs/intern_v2/logs/epochs.csv` | yes | Stage-2 (six-epoch) run, per-epoch history: throughput for cls and joint stages | Fig. 11 |
| 9 | `runs/intern_fused_ground/logs/epochs.csv` | yes | LR-fused run, per-epoch history (comparison) | Fig. 3 panel b |
| 10 | `runs/calibration/ffiw_report_vanilla.json` | yes | per-face calibration on FFIW: margin sweep, size gate, padding modes, threshold sweeps | Table 5, Fig. 7, Fig. 8 |
| 11 | `runs/timing/eval_by_generator_cd_vanilla.json` | yes | cross-domain accuracy per generator, deployed checkpoint | Table 3, Fig. 5 |
| 12 | `runs/timing/vanilla_face_timing_t512.json` | yes | end-to-end latency, 512-token cap, 1-face / 4-face / FFIW 4-face | Table 6, Fig. 7 |
| 13 | `runs/timing/vanilla_face_timing_t256.json` | yes | token-cap probe at 256 new tokens | Fig. 7 panel b |
| 14 | `runs/timing/vanilla_face_timing_t192.json` | yes | token-cap probe at 192 new tokens (answer degradation) | Fig. 7 panel b |
| 15 | `runs/samples/samples_vanilla.json` | yes | two worked examples: per-face verdicts, telemetry, aggregate document verdict | Table 7, Fig. 9 |
| 16 | `runs/samples/ann_vanilla/omi_annotated.jpg` | yes | annotated single-face example (REAL) | Fig. 9 |
| 17 | `runs/samples/ann_vanilla/3-people_annotated.jpg` | yes | annotated three-face example (FAKE) | Fig. 9 |
| 18 | `runs/samples/context/multiple-faces_face0_margins.png` | yes | crop-context sweep: margin 0.00 -> 1.00 around one face | Fig. 9 |
| 19 | `runs/intern_v2_ground/logs/train.log` | yes | Stage-4 launch record: freeze list, grounding flag, init checkpoint | Section 3, Table 1 |
| 20 | `runs/intern_v2_ground/checkpoints/best.pth` | yes | deployed weights (epoch 3, best_auc 0.990270) | all deployment numbers |
| 21 | `runs/intern_v2/checkpoints/ep006.pth` | yes | Stage-4 initialisation checkpoint | Section 3 |
| 22 | `reports/PAPER_CLAIMS.md` | yes | claim ledger: what may and may not be asserted, with artifact pointers | Section 8, Section 9 |
| 23 | `verify_stage_history.py` | yes | checkpoint weight-diff probe behind the stage history | Section 3 |

## Build

```bash
cd deepfake_project
python build_rp_journal.py
```

Outputs: `DEEP_RP_JOURNAL.md`, `DEEP_RP_JOURNAL.html` (images inlined), `reports/journal/figures/fig1..fig11.png`, and this manifest.

## Large inputs (not in git)

The weights are tracked by Git LFS pattern but are not committed: 8.5 GB exceeds the free LFS quota. `checkpoints_manifest.json` is the contract instead — restore the files to the paths below and check them with `python verify_checkpoints.py`.

| File | SHA-256 | Bytes | Role |
| --- | --- | ---: | --- |
| `runs/intern_v2_ground/checkpoints/best.pth` | `35c72be1b88f133f…` | 4365579738 | deployed checkpoint: all Table 2/3 numbers, calibration, latency, qualitative examples |
| `runs/intern_v2/checkpoints/ep006.pth` | `b93a71d60dbca329…` | 4707708703 | Stage-4 initialisation checkpoint; also the provenance behind section 3 and the Stage-2 comparison row |
| `runs/intern_v2_ground/adapters/llm_adapter/adapter_model.safetensors` | `ec210956b1273cc5…` | 147770856 | shipped LLM LoRA adapter (392 tensors, r=64) |
| `runs/intern_v2_ground/adapters/vision_adapter/adapter_model.safetensors` | `fbf3b50455f02bf5…` | 12607408 | shipped vision LoRA adapter (192 tensors, r=16) |
| `runs/intern_v2_ground/adapters/custom_heads/custom_heads.safetensors` | `55c225843f8abd42…` | 26519764 | shipped custom heads: cls_head, attn_pool, freq_branch, custom_projector, domain_router (59 tensors) |

Base model: `models/InternVL3-2B` — upstream base the adapters attach to (not part of this repository).

All listed inputs were present at build time.
