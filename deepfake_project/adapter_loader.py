"""
adapter_loader.py
=================
Load DeepfakeReasoningModel either from the whole-model checkpoint or from the
extracted adapter package (``llm_adapter`` + ``vision_adapter`` + ``custom_heads``).

`safetensors`/PEFT loading in INTEGRATION_GUIDE.md section 5.3 was a snippet
that silently skipped keys it could not place (``next(..., None)``). A silently
skipped key means an adapter that is *partly* applied: the model loads, runs,
and reports verdicts that do not match the checkpoint it was extracted from.
This module makes that impossible -- it asserts exact key agreement in both
directions, checks the tensor counts against extraction_report.json, and
verifies the weights after attach.

Usage:
    from adapter_loader import load_model

    model, tokenizer, info = load_model(
        base_model="./models/internvl3_2b_lr_fused",
        adapter_dir="./runs/intern_fused_ground/adapters",
    )
    # or
    model, tokenizer, info = load_model(
        base_model="./models/internvl3_2b_lr_fused",
        checkpoint="./runs/intern_fused_ground/checkpoints/best.pth",
    )
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import torch
from safetensors.torch import load_file as safetensors_load

from model import DeepfakeReasoningModel

# (adapter subdir, backbone attribute holding the live PEFT wrapper)
ADAPTER_PARTS: Tuple[Tuple[str, str], ...] = (
    ("llm_adapter", "language_model"),
    ("vision_adapter", "vision_model"),
)

CUSTOM_HEADS_REL = os.path.join("custom_heads", "custom_heads.safetensors")
CUSTOM_PREFIXES: Tuple[str, ...] = (
    "cls_head.", "attn_pool.", "freq_branch.",
    "custom_projector.", "domain_router.",
)

# Expected tensor counts from runs/intern_fused_ground/adapters/extraction_report.json.
# Used only when the report is missing; otherwise the report is authoritative.
EXPECTED_FALLBACK = {"llm_adapter": 392, "vision_adapter": 192, "custom_heads": 59}


class AdapterLoadError(RuntimeError):
    """Raised when an adapter package does not match the model it is attached to."""


# ── Backbone ─────────────────────────────────────────────────────────────────

def load_backbone(base_model: str,
                  arch_version: str = "v2",
                  attn_implementation: str = "sdpa",
                  device: Optional[str] = None) -> DeepfakeReasoningModel:
    """Fresh DeepfakeReasoningModel with correctly-shaped LoRA scaffolding + heads."""
    model = DeepfakeReasoningModel(
        model_path=base_model,
        arch_version=arch_version,
        attn_implementation=attn_implementation,
        use_gradient_checkpointing=False,      # inference only
    )
    if device:
        model = model.to(device)
    model.eval()
    return model


def _load_report(adapter_dir: str) -> Dict[str, Any]:
    path = os.path.join(adapter_dir, "extraction_report.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _canonical_path(path: str) -> str:
    """Absolute, symlink-resolved, case-normalised form of a path reference."""
    expanded = os.path.expanduser(str(path))
    return os.path.normcase(os.path.realpath(os.path.abspath(expanded)))


def same_model_reference(a: str, b: str) -> bool:
    """True when two base-model references denote the same model directory.

    Compares *resolved* paths, so ``./models/base``, an absolute path to that
    same directory, a trailing-slash variant and a differently-cased Windows
    path all agree. When a reference does not exist on this machine (adapters
    extracted on another box), it falls back to comparing the final path
    component -- weaker, but still catches the vanilla-vs-fused mix-up that
    actually changes verdicts. Two different directories that both exist are
    never treated as equal.
    """
    ca, cb = _canonical_path(a), _canonical_path(b)
    if ca == cb:
        return True
    if os.path.isdir(ca) and os.path.isdir(cb):
        return False
    return os.path.basename(ca) == os.path.basename(cb)


def _check_base(adapter_dir: str, report: Dict[str, Any], base_model: str) -> None:
    """Adapters are only valid on the base they were trained on."""
    recorded = ((report.get("provenance") or {}).get("base_model")
                or _config_base_model(adapter_dir))
    if not recorded:
        return
    if not same_model_reference(recorded, base_model):
        raise AdapterLoadError(
            f"adapter_dir={adapter_dir!r} was extracted from base_model={recorded!r}, "
            f"but you asked for base_model={base_model!r}. Attaching these adapters to a "
            f"different base silently changes every verdict."
        )


def _config_base_model(adapter_dir: str) -> Optional[str]:
    for subdir, _ in ADAPTER_PARTS:
        cfg = os.path.join(adapter_dir, subdir, "adapter_config.json")
        if os.path.isfile(cfg):
            with open(cfg, encoding="utf-8") as f:
                return json.load(f).get("base_model_name_or_path")
    return None


# ── Adapters ─────────────────────────────────────────────────────────────────

def _attach_one(model: DeepfakeReasoningModel,
                adapter_dir: str,
                subdir: str,
                backbone_attr: str,
                expected_tensors: Optional[int]) -> Dict[str, Any]:
    path = os.path.join(adapter_dir, subdir, "adapter_model.safetensors")
    if not os.path.isfile(path):
        raise AdapterLoadError(f"missing adapter file: {path}")

    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    peft_module = getattr(model.backbone, backbone_attr, None)
    if peft_module is None:
        raise AdapterLoadError(
            f"backbone has no '{backbone_attr}'; cannot attach {subdir}. "
            f"Was the base built with vision_lora_rank=0?")

    file_sd = safetensors_load(path, device="cpu")
    live_keys = set(get_peft_model_state_dict(peft_module))
    file_keys = set(file_sd)

    missing_in_model = sorted(file_keys - live_keys)
    missing_in_file = sorted(live_keys - file_keys)
    if missing_in_model or missing_in_file:
        raise AdapterLoadError(
            f"{subdir}: adapter file and model do not agree.\n"
            f"  {len(missing_in_model)} key(s) in the file have no target in the model "
            f"(e.g. {missing_in_model[:3]})\n"
            f"  {len(missing_in_file)} model adapter parameter(s) have no value in the file "
            f"(e.g. {missing_in_file[:3]})\n"
            f"  This would produce a partially-adapted model, so the load is refused."
        )

    if expected_tensors is not None and len(file_sd) != expected_tensors:
        raise AdapterLoadError(
            f"{subdir}: expected {expected_tensors} tensors per extraction_report.json, "
            f"found {len(file_sd)}")

    set_peft_model_state_dict(peft_module, file_sd)

    # Prove the values actually landed, not just that the keys matched.
    after = get_peft_model_state_dict(peft_module)
    for key, expected in file_sd.items():
        got = after[key]
        if not torch.equal(got.cpu().to(expected.dtype), expected):
            raise AdapterLoadError(
                f"{subdir}: tensor did not load correctly: {key} "
                f"(|delta|max={float((got.float() - expected.float()).abs().max()):.3e})")

    n_params = sum(v.numel() for v in file_sd.values())
    return {
        "tensors": len(file_sd),
        "params": n_params,
        "dtype": str(next(iter(file_sd.values())).dtype),
    }


def _attach_custom_heads(model: DeepfakeReasoningModel,
                         adapter_dir: str,
                         expected_tensors: Optional[int]) -> Dict[str, Any]:
    path = os.path.join(adapter_dir, CUSTOM_HEADS_REL)
    if not os.path.isfile(path):
        raise AdapterLoadError(f"missing custom heads file: {path}")

    sd = safetensors_load(path, device="cpu")
    if expected_tensors is not None and len(sd) != expected_tensors:
        raise AdapterLoadError(
            f"custom_heads: expected {expected_tensors} tensors, found {len(sd)}")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing_custom = [k for k in missing if k.startswith(CUSTOM_PREFIXES)]
    if missing_custom:
        raise AdapterLoadError(
            f"custom_heads: {len(missing_custom)} tensor(s) had no home in the model "
            f"(e.g. {missing_custom[:3]}) -- classifier/pooling would stay randomly initialized")
    if unexpected:
        raise AdapterLoadError(
            f"custom_heads: {len(unexpected)} checkpoint tensor(s) are not part of the model "
            f"(e.g. {unexpected[:3]})")

    n_params = sum(v.numel() for v in sd.values())
    return {
        "tensors": len(sd),
        "params": n_params,
        "modules": sorted({k.split(".")[0] for k in sd}),
    }


def attach_adapters(model: DeepfakeReasoningModel,
                    adapter_dir: str,
                    base_model: Optional[str] = None) -> Dict[str, Any]:
    """Attach llm + vision LoRA and the custom heads. Raises on any mismatch."""
    report = _load_report(adapter_dir)
    if base_model:
        _check_base(adapter_dir, report, base_model)

    expected = {k: (v or {}).get("tensors") for k, v in (report.get("adapters") or {}).items()}
    heads_expected = (report.get("custom_heads") or {}).get("tensors")

    summary: Dict[str, Any] = {"adapter_dir": adapter_dir, "parts": {}}
    for subdir, attr in ADAPTER_PARTS:
        info = _attach_one(model, adapter_dir, subdir, attr, expected.get(subdir))
        summary["parts"][subdir] = info
        print(f"  [{subdir}] {info['tensors']} tensors, {info['params']:,} params, "
              f"{info['dtype']} -> exact key match")

    heads = _attach_custom_heads(
        model, adapter_dir,
        heads_expected if heads_expected is not None else EXPECTED_FALLBACK["custom_heads"])
    summary["parts"]["custom_heads"] = heads
    print(f"  [custom_heads] {heads['tensors']} tensors, {heads['params']:,} params "
          f"-> {', '.join(heads['modules'])}")

    model.eval()
    return summary


# ── Whole-model checkpoint (the parity reference) ────────────────────────────

def load_checkpoint(model: DeepfakeReasoningModel, checkpoint: str) -> Dict[str, Any]:
    """Load a whole-model .pth the way inference.py does (strict=False by design)."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()
    return {
        "checkpoint": checkpoint,
        "epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
        "tensors": len(state_dict),
        "missing": len(missing),
        "unexpected": len(unexpected),
    }


# ── Unified entry point ──────────────────────────────────────────────────────

def load_model(base_model: str = "./models/internvl3_2b_lr_fused",
               adapter_dir: Optional[str] = None,
               checkpoint: Optional[str] = None,
               arch_version: str = "v2",
               attn_implementation: str = "sdpa",
               device: Optional[str] = None
               ) -> Tuple[DeepfakeReasoningModel, Any, Dict[str, Any]]:
    """Load the detector from exactly one weight source. Returns (model, tokenizer, info)."""
    if bool(adapter_dir) == bool(checkpoint):
        raise AdapterLoadError(
            "pass exactly one of adapter_dir= or checkpoint= "
            f"(got adapter_dir={adapter_dir!r}, checkpoint={checkpoint!r})")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_backbone(base_model, arch_version=arch_version,
                          attn_implementation=attn_implementation, device=device)

    if adapter_dir:
        info = {"source": "adapters", **attach_adapters(model, adapter_dir, base_model=base_model)}
    else:
        info = {"source": "checkpoint", **load_checkpoint(model, checkpoint)}

    if device:
        model = model.to(device)
    return model, tokenizer, info
