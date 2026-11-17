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

All listed inputs were present at build time.
