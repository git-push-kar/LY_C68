"""
dataset.py
==========
HydraFake dataset loader.

Fixes applied (vs previous version):
  #4  Tokenization moved out of __getitem__ â†’ done at load time in _load().
      Previous: tokenizer called inside the DataLoader worker loop every step,
      creating a CPU bottleneck that kept the GPU idle between batches.
      Now: reasoning text is tokenized once at dataset construction. The
      DataLoader collates pre-computed int tensors â€” extremely fast.

  #7  Reasoning quality filter â€” samples with empty reasoning are still
      included (the image + label is valid) but get a minimal fallback
      text "<answer>real</answer>" or "<answer>fake</answer>" so the LM
      loss always has a clean target, never trains on empty sequences.
"""

import os
import json
import re
import glob
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer


_ANSWER_RE = re.compile(
    r"<answer>\s*(real|fake)\s*</answer>", re.IGNORECASE)
_TAG_RE = re.compile(
    r"<(?:planning|reasoning|reflection|conclusion|fast)>(.*?)"
    r"</(?:planning|reasoning|reflection|conclusion|fast)>",
    re.DOTALL | re.IGNORECASE,
)

_TEST_GENERATOR_MAP = {
    "deepfacelab":      "cd",
    "dreamina":         "cd",
    "FFIW":             "cd",
    "gpt4o":            "cd",
    "hailuo":           "cd",
    "InfiniteYou-CD":   "cd",
    "CodeFormer":       "cf",
    "FaceAdapter":      "cf",
    "ICLight":          "cf",
    "InfiniteYou":      "cf",
    "PuLID":            "cf",
    "StarGANv2":        "cf",
    "AdobeFirefly":     "cm",
    "Flux11Pro":        "cm",
    "HART":             "cm",
    "Infinity":         "cm",
    "MAGI":             "cm",
    "StarryAI":         "cm",
    "FaceForensics++":  "id",
    "facevid2vid":      "id",
    "Hallo2":           "id",
    "Midjourney":       "id",
    "StyleGAN":         "id",
}

# Minimum non-trivial reasoning length (characters).
# Fix #7: anything shorter than this is considered empty/garbage.
_MIN_REASONING_LEN = 20


def _extract_reasoning(messages: list) -> str:
    """
    Return the FULL assistant response verbatim, preserving the structured
    tags (<fast>/<planning>/<reasoning>/<reflection>/<conclusion>) AND the
    final <answer>real|fake</answer>.

    Previous behaviour stripped the tags and dropped the answer entirely,
    which created a train/inference format mismatch: inference.py parses for
    exactly these tags, and pgrpo-style entries taught a conflicting bare
    '<answer> X </answer>' mode.
    """
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        return msg.get("content", "").strip()
    return ""


def _sanitize_reasoning(text: str, label: int) -> str:
    """
    Fix #7: guarantee a non-empty, non-garbage reasoning string.

    If the extracted reasoning is too short (empty, whitespace, or a single
    punctuation mark), replace it with a minimal but valid target that still
    teaches the LM the correct answer token. This prevents the LM loss from
    training on empty or near-empty sequences which introduce noise.
    """
    if len(text.strip()) >= _MIN_REASONING_LEN:
        return text.strip()
    # Fallback: minimal valid target with correct label
    verdict = "fake" if label == 1 else "real"
    return f"<fast>The image appears {verdict}.</fast> <answer>{verdict}</answer>"


def _resolve_path(raw: str, root: str) -> str:
    parts = Path(raw).parts
    if parts[0].lower() == "hydrafake":
        parts = parts[1:]
    if len(parts) >= 2 and parts[0].lower() == "test":
        generator = parts[1]
        category  = _TEST_GENERATOR_MAP.get(generator)
        if category:
            parts = (parts[0], category) + parts[1:]
    rel = os.path.join(*parts)
    return os.path.join(root, rel)


def build_transforms(split: str = "train", image_size: int = 448):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if split == "train":
        return T.Compose([
            T.Resize((image_size + 32, image_size + 32)),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.1),
            T.ColorJitter(brightness=0.4, contrast=0.4,
                          saturation=0.2, hue=0.1),
            T.RandomGrayscale(p=0.1),
            T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.3),
            T.RandomApply([T.RandomRotation(degrees=10)], p=0.3),
            T.ToTensor(),
            T.Normalize(mean, std),
            T.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])


class HydraFakeDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        json_dir:     str,
        tokenizer,
        split:        str = "train",
        image_size:   int = 448,
        max_text_len: int = 512,
        allowed_files: list = None,
        require_reasoning: bool = None,
    ):
        self.dataset_root = dataset_root
        self.tokenizer    = tokenizer
        self.split        = split
        self.image_size   = image_size
        self.max_text_len = max_text_len
        # allowed_files: explicit list like ["sft_36k.json"] or ["all.json"];
        # None = glob **/*.json (legacy, for val/test). require_reasoning:
        # None = auto (train=True, val/test=False), True = drop bare, False = keep bare.
        self.allowed_files = allowed_files
        if require_reasoning is None:
            require_reasoning = (split == "train")
        self.require_reasoning = require_reasoning
        self.transform    = build_transforms(split, image_size)
        self.samples      = []
        self._load(json_dir)

    def _load(self, json_dir: str):
        # Explicit file list takes priority (for cls/sft/pref separation)
        if self.allowed_files is not None:
            files = []
            for fname in self.allowed_files:
                cand = os.path.join(json_dir, fname)
                if os.path.isfile(cand):
                    files.append(cand)
                else:
                    # also try recursive find for nested names like "sft_36k.json"
                    found = glob.glob(os.path.join(json_dir, "**", fname), recursive=True)
                    files.extend(found)
            files = sorted(set(files))
        else:
            files = glob.glob(
                os.path.join(json_dir, "**", "*.json"), recursive=True)
            if not files:
                files = glob.glob(os.path.join(json_dir, "*.json"))
        if not files:
            raise FileNotFoundError(
                f"No JSON files found under: {json_dir} (filter={self.allowed_files})")
        # pgrpo is always inactive — never load it even via glob
        files = [f for f in files if "pgrpo" not in os.path.basename(f).lower()]

        missing_count  = 0
        fallback_count = 0
        dupe_count     = 0
        no_trace_count = 0
        too_long_count = 0

        # Phase 1 â€” merge conflicting annotations per image.
        # The same image is listed across multiple JSON files with
        # DIFFERENT assistant targets: all.json / per-generator files /
        # pgrpo contain a bare '<answer> X </answer>', while sft_36k and
        # mipo carry the full structured trace. Keep the RICHEST copy
        # (tagged trace beats bare answer) per unique image.
        candidates = {}   # normcase path -> (rich, resolved, label, raw, ftype)

        for jf in sorted(files):
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
            entries = data if isinstance(data, list) else [data]

            for e in entries:
                imgs = e.get("images", [])
                if not imgs:
                    continue

                resolved = _resolve_path(imgs[0], self.dataset_root)
                label    = int(e.get("label", 0))

                # Skip entries whose image is missing on disk instead of
                # silently training on a black-image substitute.
                if not os.path.exists(resolved):
                    missing_count += 1
                    continue

                raw_reasoning = _extract_reasoning(e.get("messages", []))
                rich = 1 if _TAG_RE.search(raw_reasoning or "") else 0

                key = os.path.normcase(resolved)
                prev = candidates.get(key)
                if prev is None:
                    candidates[key] = (rich, resolved, label, raw_reasoning,
                                       e.get("type", "unknown"))
                else:
                    dupe_count += 1
                    if rich > prev[0]:
                        candidates[key] = (rich, resolved, label,
                                           raw_reasoning,
                                           e.get("type", "unknown"))

        # Phase 2 â€” build samples from the winning annotation only.
        # Images with NO tagged copy anywhere are skipped entirely: they
        # would teach the LM the degenerate 'bare answer' mode that caused
        # the reasoning-format collapse. Harmless for the classifier,
        # which is frozen during corrective fine-tuning.
        for rich, resolved, label, raw_reasoning, ftype in candidates.values():
            if not rich:
                if self.require_reasoning:
                    no_trace_count += 1
                    continue
                # Classification pool (all.json) or val/test bare: keep as
                # classification-only. Use zero attention so LM loss (if ever
                # computed) is fully masked, and no synthetic reasoning is invented.
                self.samples.append({
                    "image_path":   resolved,
                    "label":        label,
                    "reasoning_ids": torch.zeros(self.max_text_len, dtype=torch.long),
                    "attention_mask": torch.zeros(self.max_text_len, dtype=torch.long),
                    "reasoning_text": "",
                    "forgery_type": ftype,
                })
                continue

            # Fix #7: sanitize empty / garbage reasoning
            reasoning = _sanitize_reasoning(raw_reasoning, label)
            if reasoning != raw_reasoning.strip():
                fallback_count += 1

            # Fix #4: tokenize HERE at load time, not in __getitem__.
            #
            # EOS append: targets never contained an eos_token before, so
            # the LM never learned WHEN to stop generating. Appending it
            # makes '<answer>...</answer><eos>' the learned ending.
            target_text = reasoning + self.tokenizer.eos_token

            # Drop over-long targets instead of truncating: a truncated
            # trace ends mid-sentence with NO <answer> and NO eos, which
            # would teach the LM to stop without answering.
            n_tok = len(self.tokenizer(target_text)["input_ids"])
            if n_tok > self.max_text_len:
                too_long_count += 1
                continue

            enc = self.tokenizer(
                target_text,
                max_length=self.max_text_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            self.samples.append({
                "image_path":     resolved,
                "label":          label,
                "reasoning_ids":  enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "reasoning_text": reasoning,   # kept for logging/debug only
                "forgery_type":   ftype,
            })

        print(f"[{self.split}] {len(self.samples)} usable samples "
              f"from {len(files)} JSON file(s). "
              f"Unique images found: {len(candidates)}  "
              f"Missing(skipped): {missing_count}  "
              f"Duplicates(merged): {dupe_count}  "
              f"No-tagged-trace(skipped): {no_trace_count}  "
              f"Too-long(skipped): {too_long_count}  "
              f"Reasoning fallbacks: {fallback_count}")

        if missing_count > 0:
            print(f"  WARNING: skipped {missing_count} entries with "
                  f"missing images under root: {self.dataset_root}")
        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"[{self.split}] 0 usable images resolved from {json_dir}. "
                f"Check --dataset_root ({self.dataset_root}) and that the "
                f"on-disk folder layout matches the JSON paths.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        # Image loading is still here (can't pre-load images into RAM)
        try:
            img = Image.open(s["image_path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.image_size, self.image_size), 0)

        pixel_values = self.transform(img)

        return {
            "pixel_values":   pixel_values,
            "labels":         torch.tensor(s["label"], dtype=torch.long),
            # Fix #4: these are pre-computed tensors, not freshly tokenized
            "reasoning_ids":  s["reasoning_ids"],
            "attention_mask": s["attention_mask"],
            "reasoning_text": s["reasoning_text"],  # str, for debug logging
        }


def _balanced_sampler(dataset):
    labels        = [s["label"] for s in dataset.samples]
    class_counts  = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[torch.tensor(labels)]
    return WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True)


def build_dataloaders(
    dataset_root:   str,
    json_root:      str,
    tokenizer_name: str = "./models/InternVL3-2B",
    image_size:     int = 448,
    max_text_len:   int = 512,
    batch_size:     int = 8,
    num_workers:    int = 8,
):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    loaders = {}

    for split in ("train", "val", "test"):
        json_dir = os.path.join(json_root, split)
        if not os.path.isdir(json_dir):
            continue

        ds = HydraFakeDataset(
            dataset_root, json_dir, tokenizer,
            split, image_size, max_text_len,
        )

        if split == "train":
            loader = DataLoader(
                ds,
                batch_size=batch_size,
                sampler=_balanced_sampler(ds),
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
            )
        else:
            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
        loaders[split] = loader

    return loaders, tokenizer


# ── Separated supervision pools (Polyvalent, not Veritas) ───────────────────
# Classification = all.json full image/label pool (48,320 unique, bare allowed).
# SFT          = sft_36k.json rich traces only (36,750 → ~35k after >512 drop).
# Preference   = mipo_3k.json chosen/rejected pairs (3,480 pairs, 870 images ×4).
# pgrpo_8k.json is intentionally never loaded here.

class HydraFakePreferenceDataset(Dataset):
    """Preserves mipo_3k.json chosen/rejected pairs for later preference training.
    Each sample is one pair (image + chosen + rejected), not deduped to unique
    images — all 3,480 pairs are kept. No DPO loss is computed here; the
    dataset is only exposed for future stages."""
    def __init__(self, dataset_root: str, json_dir: str, tokenizer,
                 image_size: int = 448, max_text_len: int = 512):
        self.dataset_root = dataset_root
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_text_len = max_text_len
        self.transform = build_transforms("train", image_size)
        self.samples = []
        self._load(json_dir)

    def _load(self, json_dir: str):
        # Locate mipo file explicitly — do not glob all train JSONs
        candidates = glob.glob(os.path.join(json_dir, "**", "mipo_3k.json"), recursive=True)
        if not candidates:
            candidates = [os.path.join(json_dir, "mipo_3k.json")]
        files = [f for f in candidates if os.path.isfile(f)]
        if not files:
            raise FileNotFoundError(f"mipo_3k.json not found under: {json_dir}")

        kept = 0
        for jf in sorted(files):
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
            entries = data if isinstance(data, list) else [data]
            for e in entries:
                imgs = e.get("images", [])
                if not imgs:
                    continue
                resolved = _resolve_path(imgs[0], self.dataset_root)
                if not os.path.exists(resolved):
                    continue
                label = int(e.get("label", 0))
                # Chosen = messages[assistant], rejected = rejected_response
                chosen_raw = _extract_reasoning(e.get("messages", []))
                rejected_raw = (e.get("rejected_response") or "").strip()
                if not chosen_raw or not rejected_raw:
                    continue
                # Tokenize both with EOS
                chosen_enc = self.tokenizer(chosen_raw + self.tokenizer.eos_token,
                                            max_length=self.max_text_len, padding="max_length",
                                            truncation=True, return_tensors="pt")
                rejected_enc = self.tokenizer(rejected_raw + self.tokenizer.eos_token,
                                              max_length=self.max_text_len, padding="max_length",
                                              truncation=True, return_tensors="pt")
                self.samples.append({
                    "image_path": resolved,
                    "label": label,
                    "chosen_ids": chosen_enc["input_ids"].squeeze(0),
                    "chosen_mask": chosen_enc["attention_mask"].squeeze(0),
                    "rejected_ids": rejected_enc["input_ids"].squeeze(0),
                    "rejected_mask": rejected_enc["attention_mask"].squeeze(0),
                    "chosen_text": chosen_raw,
                    "rejected_text": rejected_raw,
                })
                kept += 1
        print(f"[pref] {len(self.samples)} preference pairs from {len(files)} file(s) (kept {kept})")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx: int):
        s = self.samples[idx]
        try:
            img = Image.open(s["image_path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.image_size, self.image_size), 0)
        pv = self.transform(img)
        return {
            "pixel_values": pv,
            "labels": torch.tensor(s["label"], dtype=torch.long),
            "chosen_ids": s["chosen_ids"],
            "chosen_mask": s["chosen_mask"],
            "rejected_ids": s["rejected_ids"],
            "rejected_mask": s["rejected_mask"],
        }


def build_classification_loader(dataset_root: str, json_root: str, tokenizer,
                                image_size: int = 448, max_text_len: int = 512,
                                batch_size: int = 8, num_workers: int = 8):
    """Classification pool: train/all.json only, bare allowed (48,320 unique)."""
    json_dir = os.path.join(json_root, "train")
    ds = HydraFakeDataset(dataset_root, json_dir, tokenizer, split="train",
                          image_size=image_size, max_text_len=max_text_len,
                          allowed_files=["all.json"], require_reasoning=False)
    return DataLoader(ds, batch_size=batch_size, sampler=_balanced_sampler(ds),
                      num_workers=num_workers, pin_memory=True, drop_last=True), ds


def build_sft_loader(dataset_root: str, json_root: str, tokenizer,
                     image_size: int = 448, max_text_len: int = 512,
                     batch_size: int = 8, num_workers: int = 8):
    """SFT reasoning pool: train/sft_36k.json only, rich required (~35k usable)."""
    json_dir = os.path.join(json_root, "train")
    ds = HydraFakeDataset(dataset_root, json_dir, tokenizer, split="train",
                          image_size=image_size, max_text_len=max_text_len,
                          allowed_files=["sft_36k.json"], require_reasoning=True)
    return DataLoader(ds, batch_size=batch_size, sampler=_balanced_sampler(ds),
                      num_workers=num_workers, pin_memory=True, drop_last=True), ds


def build_preference_loader(dataset_root: str, json_root: str, tokenizer,
                            image_size: int = 448, max_text_len: int = 512,
                            batch_size: int = 8, num_workers: int = 8):
    """Preference pool: train/mipo_3k.json chosen/rejected (3,480 pairs). Inactive in current joint training."""
    json_dir = os.path.join(json_root, "train")
    ds = HydraFakePreferenceDataset(dataset_root, json_dir, tokenizer, image_size, max_text_len)
    return DataLoader(ds, batch_size=batch_size, sampler=_balanced_sampler(ds),
                      num_workers=num_workers, pin_memory=True, drop_last=True), ds


def build_separated_loaders(dataset_root: str, json_root: str, tokenizer_name: str = "./models/InternVL3-2B",
                            image_size: int = 448, max_text_len: int = 512,
                            batch_size: int = 8, num_workers: int = 8):
    """Return dict with separated pools: cls_train (48k), sft_train (35k), pref (3,480 pairs, inactive), val, test."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    loaders = {}
    # Classification pool
    cls_loader, _ = build_classification_loader(dataset_root, json_root, tokenizer, image_size, max_text_len, batch_size, num_workers)
    loaders["cls_train"] = cls_loader
    # SFT pool
    sft_loader, _ = build_sft_loader(dataset_root, json_root, tokenizer, image_size, max_text_len, batch_size, num_workers)
    loaders["sft_train"] = sft_loader
    # Keep legacy "train" as sft for backward compat (train.py joint loss)
    loaders["train"] = sft_loader
    # Preference pool (exposed but not trained yet)
    try:
        pref_loader, _ = build_preference_loader(dataset_root, json_root, tokenizer, image_size, max_text_len, batch_size, num_workers)
        loaders["pref"] = pref_loader
    except FileNotFoundError:
        pass
    # Val/test via legacy (glob, cls-only fallback)
    for split in ("val", "test"):
        json_dir = os.path.join(json_root, split)
        if not os.path.isdir(json_dir):
            continue
        ds = HydraFakeDataset(dataset_root, json_dir, tokenizer, split, image_size, max_text_len)
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return loaders, tokenizer
