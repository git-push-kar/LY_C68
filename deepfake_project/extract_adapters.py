"""
extract_adapters.py
===================
Extract standalone PEFT LoRA adapters (LLM + vision) PLUS the full-finetune
custom heads from a full-model `.pth` checkpoint so the run can be redeployed
elsewhere.

Final-model status (verified 2026-10-14):
  runs/intern_fused_ground/checkpoints/best.pth is a WHOLE-MODEL checkpoint
  (1328 tensors, ~4.36 GB: frozen fused backbone + 584 LoRA tensors
  [392 LLM r=64 + 192 vision r=16] + 59 custom-head tensors), NOT an
  adapter-only file. It was post-trained from
  runs/intern_fused/checkpoints/ep006.pth with:
    epochs_cls=0, epochs_joint=3, --freeze_cls_head (cls_head frozen),
    grounding on, base ./models/internvl3_2b_lr_fused.
  Optimizer groups at train time: LoRA=584 tensors, Custom=43 tensors
  (attn_pool + freq_branch + custom_projector + domain_router; cls_head
  excluded by the freeze).

Because adapters alone do NOT reproduce the classifier/reasoning behaviour,
this script now extracts BOTH:
  <out>/llm_adapter/          — LLM LoRA (attach to base.language_model)
  <out>/vision_adapter/       — Vision LoRA (attach to base.vision_model)
  <out>/custom_heads/         — cls_head + attn_pool + freq_branch +
                                custom_projector + domain_router
                                (load onto a fresh DeepfakeReasoningModel
                                via load_state_dict(strict=False))

Adapter attach pattern (needs the SAME fused base the run trained from):

    from peft import PeftModel
    base = AutoModel.from_pretrained("./models/internvl3_2b_lr_fused", trust_remote_code=True, torch_dtype=torch.bfloat16)
    llm  = PeftModel.from_pretrained(base.language_model, "<out>/llm_adapter")
    vit  = PeftModel.from_pretrained(base.vision_model,   "<out>/vision_adapter")
    # then load <out>/custom_heads/custom_heads.safetensors into
    # cls_head / attn_pool / freq_branch / custom_projector / domain_router

Usage (from deepfake_project/):
    python extract_adapters.py \
        --checkpoint ./runs/intern_fused_ground/checkpoints/best.pth \
        --output_dir ./runs/intern_fused_ground/adapters \
        --base_model ./models/internvl3_2b_lr_fused
"""

import argparse
import json
import os
import re
import sys

import torch
from safetensors.torch import load_file as safetensors_load, save_file as safetensors_save

try:
    from peft import LoraConfig, TaskType
except ImportError:
    sys.exit("ERROR: peft is required (pip install -r requirements.txt)")

# Must match the injection in model.py (DeepfakeReasoningModel steps 2-3).
LLM_DEFAULTS = {
    "r": 64, "lora_alpha": 128, "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
}
VISION_DEFAULTS = {
    "r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
    "target_modules": ["qkv", "proj", "fc1", "fc2"],
}

PARTS = {
    # adapter subdir : (state-dict prefix in .pth, is_causal_lm)
    "llm_adapter": ("backbone.language_model.", True),
    "vision_adapter": ("backbone.vision_model.", False),
}

# Full-finetune custom heads (NOT LoRA — must ship alongside the adapters
# for the intern_fused_ground final model to reproduce its scores).
CUSTOM_PREFIXES = ("cls_head.", "attn_pool.", "freq_branch.",
                   "custom_projector.", "domain_router.")

PEFT_INFIX = ".base_model.model."


def parse_args():
    p = argparse.ArgumentParser(description="Extract standalone PEFT adapters + custom heads from a full .pth checkpoint")
    p.add_argument("--checkpoint", default="./runs/intern_fused_ground/checkpoints/best.pth")
    p.add_argument("--output_dir", default="./runs/intern_fused_ground/adapters")
    p.add_argument("--base_model", default="./models/internvl3_2b_lr_fused",
                   help="Recorded as base_model_name_or_path in adapter_config.json (must be the base the run trained from)")
    p.add_argument("--llm_r", type=int, default=LLM_DEFAULTS["r"])
    p.add_argument("--llm_alpha", type=int, default=LLM_DEFAULTS["lora_alpha"])
    p.add_argument("--llm_dropout", type=float, default=LLM_DEFAULTS["lora_dropout"])
    p.add_argument("--vision_r", type=int, default=VISION_DEFAULTS["r"])
    p.add_argument("--vision_alpha", type=int, default=VISION_DEFAULTS["lora_alpha"])
    p.add_argument("--vision_dropout", type=float, default=VISION_DEFAULTS["lora_dropout"])
    p.add_argument("--skip_functional_check", action="store_true",
                   help="Skip rebuilding the model to strict-load the extracted adapters")
    p.add_argument("--no_custom_heads", action="store_true",
                   help="Skip extracting the full-finetune custom heads (adapters only)")
    return p.parse_args()


def strip_peft_prefix(key: str) -> str:
    """backbone.language_model.base_model.model.layers.0...lora_A.weight
    -> layers.0...lora_A.weight (adapter key space, adapter-name infix kept).
    Also handles keys already rooted at base_model.model.* (e.g. from a
    PeftModel wrapping the sub-module directly)."""
    if key.startswith("base_model.model."):
        return key[len("base_model.model."):]
    if PEFT_INFIX in key:
        return key.split(PEFT_INFIX, 1)[1]
    return key


def to_adapter_key(full_key: str) -> str:
    """Map a full-model state-dict key to the standard PEFT adapter key space:
    strip only our wrapper prefix (``backbone.language_model.`` /
    ``backbone.vision_model.``), keeping PEFT's own ``base_model.model.``
    prefix, and drop the adapter-name infix (``.default.``) to match the
    peft 0.7.1 file format produced by get_peft_model_state_dict."""
    k = full_key
    for prefix in ("backbone.language_model.", "backbone.vision_model."):
        if k.startswith(prefix):
            k = k[len(prefix):]
            break
    k, n = re.subn(r"(\.lora_[AB])\.[^.]+\.", r"\1.", k)
    assert n == 1, f"unexpected LoRA key shape (adapter-name infix): {full_key}"
    return k


def adapter_names_in(sd_keys) -> set:
    """Adapter names referenced by full-model keys (…lora_A.<name>.weight)."""
    return {m.group(1) for k in sd_keys
            if (m := re.search(r"\.lora_[AB]\.([^.]+)\.", k))}


def derive_target_modules(adapter_sd: dict) -> list:
    """Infer PEFT target module names from tensor keys (….<mod>.lora_A/B.weight)."""
    mods = set()
    for k in adapter_sd:
        if ".lora_" in k:
            mods.add(k.split(".lora_")[0].rsplit(".", 1)[-1])
    return sorted(mods)


def check_adapter(name: str, sd: dict) -> None:
    a_keys = {k for k in sd if ".lora_A." in k}
    b_keys = {k for k in sd if ".lora_B." in k}
    a_roots = {k.replace(".lora_A.", ".lora_X.") for k in a_keys}
    b_roots = {k.replace(".lora_B.", ".lora_X.") for k in b_keys}
    assert a_keys, f"{name}: no lora_A tensors found"
    assert a_roots == b_roots, (
        f"{name}: A/B pairing mismatch: "
        f"only-A={len(a_roots - b_roots)} only-B={len(b_roots - a_roots)}")
    dtypes = {str(v.dtype) for v in sd.values()}
    assert len(dtypes) == 1, f"{name}: mixed dtypes {dtypes}"
    for k, v in sd.items():
        assert v.dim() == 2, f"{name}: non-matrix tensor {k}: {tuple(v.shape)}"
    n_params = sum(v.numel() for v in sd.values())
    print(f"  [{name}] tensors={len(sd)} params={n_params:,} dtype={dtypes.pop()} "
          f"targets={derive_target_modules(sd)}")


def main():
    args = parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    print(f"  epoch={ckpt.get('epoch', '?') if isinstance(ckpt, dict) else '?'} "
          f"total_tensors={len(sd)}")
    print(f"  sample key: {next(iter(sd))}")

    os.makedirs(args.output_dir, exist_ok=True)
    report = {"source_checkpoint": os.path.abspath(args.checkpoint),
              "epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
              "adapters": {}}

    names = adapter_names_in(sd.keys())
    assert names == {"default"}, f"expected single 'default' adapter, found: {names}"
    print(f"  adapter name in checkpoint: 'default'")

    for subdir, (prefix, is_llm) in PARTS.items():
        part = {to_adapter_key(k): v.contiguous().cpu()
                for k, v in sd.items()
                if k.startswith(prefix) and ".lora_" in k}
        if not part:
            print(f"  [{subdir}] WARNING: no LoRA keys under '{prefix}' — skipped")
            continue

        print(f"Extracting {subdir} ...")
        check_adapter(subdir, part)

        if is_llm:
            cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.llm_r,
                             lora_alpha=args.llm_alpha, lora_dropout=args.llm_dropout,
                             target_modules=LLM_DEFAULTS["target_modules"], bias="none")
        else:
            cfg = LoraConfig(r=args.vision_r, lora_alpha=args.vision_alpha,
                             lora_dropout=args.vision_dropout,
                             target_modules=VISION_DEFAULTS["target_modules"], bias="none")
        # Cross-check recorded targets against what the tensors show.
        seen = derive_target_modules(part)
        cfg.target_modules = sorted(set(cfg.target_modules) | set(seen))
        cfg.base_model_name_or_path = args.base_model

        out_dir = os.path.join(args.output_dir, subdir)
        os.makedirs(out_dir, exist_ok=True)
        w_path = os.path.join(out_dir, "adapter_model.safetensors")
        c_path = os.path.join(out_dir, "adapter_config.json")
        safetensors_save(part, w_path)
        with open(c_path, "w", encoding="utf-8") as f:
            json.dump(cfg.to_dict(), f, indent=2)
        print(f"  wrote {w_path} ({os.path.getsize(w_path) / 1e6:.1f} MB) + adapter_config.json")

        # Round-trip integrity: reload and compare exactly.
        rt = safetensors_load(w_path, device="cpu")
        assert set(rt) == set(part), f"{subdir}: key mismatch after reload"
        for k in part:
            assert torch.equal(rt[k], part[k]), f"{subdir}: tensor mismatch after reload: {k}"
        print(f"  round-trip integrity OK ({len(rt)} tensors identical)")

        report["adapters"][subdir] = {
            "tensors": len(part),
            "params": sum(v.numel() for v in part.values()),
            "dtype": str(next(iter(part.values())).dtype),
            "target_modules": cfg.target_modules,
            "r": cfg.r, "lora_alpha": cfg.lora_alpha,
        }

    # ── Custom heads (full-finetune, required for deployment) ──────────
    # intern_fused_ground post-training updated attn_pool / freq_branch /
    # custom_projector / domain_router (cls_head frozen but still needed at
    # inference). Adapters alone do not reproduce classifier scores.
    if not args.no_custom_heads:
        custom = {k: v.contiguous().cpu() for k, v in sd.items()
                  if k.startswith(CUSTOM_PREFIXES)}
        if not custom:
            print("  [custom_heads] WARNING: no custom-head keys found — skipped")
        else:
            n_params = sum(v.numel() for v in custom.values())
            print(f"Extracting custom_heads ... tensors={len(custom)} params={n_params:,}")
            out_dir = os.path.join(args.output_dir, "custom_heads")
            os.makedirs(out_dir, exist_ok=True)
            w_path = os.path.join(out_dir, "custom_heads.safetensors")
            safetensors_save(custom, w_path)
            rt = safetensors_load(w_path, device="cpu")
            assert set(rt) == set(custom), "custom_heads: key mismatch after reload"
            for k in custom:
                assert torch.equal(rt[k], custom[k]), f"custom_heads: tensor mismatch: {k}"
            print(f"  wrote {w_path} ({os.path.getsize(w_path) / 1e6:.1f} MB) + round-trip OK")
            report["custom_heads"] = {
                "tensors": len(custom),
                "params": n_params,
                "modules": sorted({k.split(".")[0] for k in custom}),
                "file": "custom_heads/custom_heads.safetensors",
                "note": ("Load onto a fresh DeepfakeReasoningModel (same arch_version/arch) "
                         "via model.load_state_dict(..., strict=False) together with the "
                         "two PEFT adapters. cls_head was frozen during intern_fused_ground "
                         "post-training but is still required at inference."),
            }
    else:
        print("  [custom_heads] skipped (--no_custom_heads)")

    report["provenance"] = {
        "training": "intern_fused_ground joint post-training from runs/intern_fused/checkpoints/ep006.pth",
        "base_model": args.base_model,
        "frozen": "cls_head (4 tensors); trainable: LLM LoRA r=64 + vision LoRA r=16 + attn_pool/freq_branch/custom_projector/domain_router",
        "checkpoint_type": "whole-model state_dict (backbone + LoRA + custom heads), NOT adapter-only",
    }

    if not args.skip_functional_check and report["adapters"]:
        print("Functional check: loading clean base + attaching via PeftModel.from_pretrained ...")
        from transformers import AutoModel
        from peft import PeftModel, get_peft_model_state_dict
        backbone = AutoModel.from_pretrained(
            args.base_model, trust_remote_code=True, torch_dtype=torch.bfloat16)
        checks = {"llm_adapter": "language_model", "vision_adapter": "vision_model"}
        for subdir, part_name in checks.items():
            if subdir not in report["adapters"]:
                continue
            adapted = PeftModel.from_pretrained(
                getattr(backbone, part_name),
                os.path.join(args.output_dir, subdir))
            # Compare via the canonical PEFT state dict (adapter-name-free keys,
            # same key space as the extracted files).
            src = safetensors_load(
                os.path.join(args.output_dir, subdir, "adapter_model.safetensors"),
                device="cpu")
            got = {k: v.cpu() for k, v in get_peft_model_state_dict(adapted).items()}
            assert set(got) == set(src), (
                f"{subdir}: key mismatch after from_pretrained: "
                f"only-got={len(set(got) - set(src))} only-src={len(set(src) - set(got))}")
            for k in src:
                assert torch.equal(got[k].to(src[k].dtype), src[k]), \
                    f"{subdir}: tensor mismatch after from_pretrained: {k}"
            print(f"  [{subdir}] from_pretrained attach + tensor equality OK "
                  f"({len(got)} tensors)")
        print("Functional check passed: adapters attach to a clean base with exact weights.")

    r_path = os.path.join(args.output_dir, "extraction_report.json")
    with open(r_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {r_path}")
    print("Done.")


if __name__ == "__main__":
    main()
