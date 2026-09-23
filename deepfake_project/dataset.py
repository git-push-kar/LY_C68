"""
dataset.py
==========
HydraFake dataset loader for DeepfakeReasoningModel v2.

Fixes & Enhancements in v2:
  #1  Frequency Features       — compute_fft_features(img) calculates log-magnitude 2D FFT spectrum.
  #2  Answer Token Position    — locates the token index of "real"/"fake" inside <answer> tags for consistency loss.
  #3  Tokenization at load     — done once at dataset initialization in _load(), zero CPU bottlenecks in worker loops.
  #4  Reasoning Quality Filter — sanitizes empty/short reasoning, ensures valid structured tags.
  #5  Separated Pools          — classification (all.json, 48k), SFT (sft_36k.json, 35k), Preference (mipo_3k, 3,480 pairs).
  #6  Forgery Type Tracking    — exposes per-family generator metadata (cd, cf, cm, id) for diagnostic evaluation.
"""

import os
import json
import re
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import numpy as np
import torchvision.transforms as T
from transformers import AutoTokenizer


_ANSWER_RE = re.compile(
    r"<answer>\s*(real|fake)\s*</answer>", re.IGNORECASE
)
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

_MIN_REASONING_LEN = 20


# ── Frequency-Domain Feature Extraction (2.3) ─────────────────────────────────

def compute_fft_features(img: Image.Image, target_size: int = 224) -> torch.Tensor:
    """
    Compute log-magnitude 2D FFT spectrum from an image.
    
    Generative artifacts (checkerboard artifacts, high-frequency attenuation,
    spectral peaks from upsampling) appear prominently in the Fourier domain.
    
    Args:
        img: PIL Image
        target_size: resolution for FFT (224x224 provides high spectral resolution with minimal memory)
    Returns:
        torch.Tensor of shape (3, target_size, target_size), float32, normalized
    """
    img_resized = img.resize((target_size, target_size), Image.BILINEAR)
    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    elif img_np.shape[2] == 4:
        img_np = img_np[:, :, :3]

    # Convert to (C, H, W)
    img_np = np.transpose(img_np, (2, 0, 1))

    # 2D FFT across spatial dimensions
    fft = np.fft.fft2(img_np, axes=(-2, -1))
    fft_shift = np.fft.fftshift(fft, axes=(-2, -1))
    mag = np.abs(fft_shift)
    log_mag = np.log1p(mag)

    # Channel-wise standard normalization
    mean = np.mean(log_mag, axis=(-2, -1), keepdims=True)
    std = np.std(log_mag, axis=(-2, -1), keepdims=True) + 1e-6
    norm_fft = (log_mag - mean) / std

    return torch.tensor(norm_fft, dtype=torch.float32)


# ── Answer Token Location Helper (2.5) ────────────────────────────────────────

def find_answer_token_pos(target_text: str, tokenizer, enc) -> int:
    """
    Locate the token index of the word 'real' or 'fake' inside <answer>...</answer>.
    Used to align the LM logit distribution with the classification head.
    
    Returns 0-based token index in enc['input_ids'], or -1 if not found.
    """
    m = _ANSWER_RE.search(target_text)
    if not m:
        return -1
    start_char = m.start(1)

    # Try fast tokenizer char_to_token mapping
    if hasattr(enc, "char_to_token"):
        try:
            pos = enc.char_to_token(start_char)
            if pos is not None:
                return pos
        except Exception:
            pass

    # Try offset mapping if available in enc
    if "offset_mapping" in enc:
        offsets = enc["offset_mapping"]
        if hasattr(offsets, "squeeze"):
            offsets = offsets.squeeze(0).tolist()
        for idx, (s, e) in enumerate(offsets):
            if s <= start_char < e:
                return idx

    # Fallback: tokenizing prefix
    try:
        prefix = target_text[:start_char]
        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        # Account for possible BOS
        full_with_special = tokenizer(target_text, add_special_tokens=True)["input_ids"]
        full_without_special = tokenizer(target_text, add_special_tokens=False)["input_ids"]
        has_bos = len(full_with_special) > len(full_without_special)
        return len(prefix_ids) + (1 if has_bos else 0)
    except Exception:
        return -1


# ── Reasoning Extraction & Sanitization ───────────────────────────────────────

def _extract_reasoning(messages: list) -> str:
    """Return the assistant response verbatim, preserving all structured tags."""
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        return msg.get("content", "").strip()
    return ""


def _sanitize_reasoning(text: str, label: int) -> str:
    """Guarantee non-empty reasoning string with valid fallback."""
    if len(text.strip()) >= _MIN_REASONING_LEN:
        return text.strip()
    verdict = "fake" if label == 1 else "real"
    return f"<fast>The image appears {verdict}.</fast> <answer>{verdict}</answer>"


def _resolve_path(raw: str, root: str) -> str:
    parts = Path(raw).parts
    if parts and parts[0].lower() == "hydrafake":
        parts = parts[1:]
    if len(parts) >= 2 and parts[0].lower() == "test":
        generator = parts[1]
        category = _TEST_GENERATOR_MAP.get(generator)
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


# ── HydraFake Dataset (v2) ───────────────────────────────────────────────────

class HydraFakeDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        json_dir:     str,
        tokenizer,
        split:        str = "train",
        image_size:   int = 448,
        max_text_len: int = 512,
        allowed_files: Optional[List[str]] = None,
        require_reasoning: Optional[bool] = None,
    ):
        self.dataset_root = dataset_root
        self.tokenizer = tokenizer
        self.split = split
        self.image_size = image_size
        self.max_text_len = max_text_len
        self.allowed_files = allowed_files
        if require_reasoning is None:
            require_reasoning = (split == "train")
        self.require_reasoning = require_reasoning
        self.transform = build_transforms(split, image_size)
        self.samples = []
        self._load(json_dir)

    def _load(self, json_dir: str):
        if self.allowed_files is not None:
            files = []
            for fname in self.allowed_files:
                cand = os.path.join(json_dir, fname)
                if os.path.isfile(cand):
                    files.append(cand)
                else:
                    found = glob.glob(os.path.join(json_dir, "**", fname), recursive=True)
                    files.extend(found)
            files = sorted(set(files))
        else:
            files = glob.glob(os.path.join(json_dir, "**", "*.json"), recursive=True)
            if not files:
                files = glob.glob(os.path.join(json_dir, "*.json"))
        if not files:
            raise FileNotFoundError(
                f"No JSON files found under: {json_dir} (filter={self.allowed_files})"
            )

        # pgrpo is always inactive — filter out
        files = [f for f in files if "pgrpo" not in os.path.basename(f).lower()]

        missing_count  = 0
        fallback_count = 0
        dupe_count     = 0
        no_trace_count = 0
        too_long_count = 0

        candidates = {}  # normcase path -> (rich, resolved, label, raw, ftype)

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

        for rich, resolved, label, raw_reasoning, ftype in candidates.values():
            if not rich:
                if self.require_reasoning:
                    no_trace_count += 1
                    continue
                self.samples.append({
                    "image_path":       resolved,
                    "label":            label,
                    "reasoning_ids":    torch.zeros(self.max_text_len, dtype=torch.long),
                    "attention_mask":   torch.zeros(self.max_text_len, dtype=torch.long),
                    "answer_token_pos": torch.tensor(-1, dtype=torch.long),
                    "reasoning_text":   "",
                    "forgery_type":     ftype,
                })
                continue

            reasoning = _sanitize_reasoning(raw_reasoning, label)
            if reasoning != raw_reasoning.strip():
                fallback_count += 1

            # Append EOS token so LM learns when to stop
            target_text = reasoning + (self.tokenizer.eos_token or "<|endoftext|>")

            # Drop over-long targets instead of truncating
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

            # Locate answer token position for self-consistency loss (2.5)
            ans_pos = find_answer_token_pos(target_text, self.tokenizer, enc)

            self.samples.append({
                "image_path":       resolved,
                "label":            label,
                "reasoning_ids":    enc["input_ids"].squeeze(0),
                "attention_mask":   enc["attention_mask"].squeeze(0),
                "answer_token_pos": torch.tensor(ans_pos, dtype=torch.long),
                "reasoning_text":   reasoning,
                "forgery_type":     ftype,
            })

        print(f"[{self.split}] {len(self.samples)} usable samples "
              f"from {len(files)} JSON file(s). "
              f"Unique images: {len(candidates)}  "
              f"Missing(skipped): {missing_count}  "
              f"Duplicates(merged): {dupe_count}  "
              f"No-tagged-trace(skipped): {no_trace_count}  "
              f"Too-long(skipped): {too_long_count}  "
              f"Reasoning fallbacks: {fallback_count}")

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"[{self.split}] 0 usable images resolved from {json_dir}. "
                f"Check --dataset_root ({self.dataset_root})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        try:
            img = Image.open(s["image_path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.image_size, self.image_size), 0)

        pixel_values  = self.transform(img)
        freq_features = compute_fft_features(img, target_size=224)

        return {
            "pixel_values":     pixel_values,
            "freq_features":    freq_features,
            "labels":           torch.tensor(s["label"], dtype=torch.long),
            "reasoning_ids":    s["reasoning_ids"],
            "attention_mask":   s["attention_mask"],
            "answer_token_pos": s["answer_token_pos"],
            "reasoning_text":   s["reasoning_text"],
            "forgery_type":     s["forgery_type"],
        }


# ── Preference Dataset (2.7: mipo_3k chosen/rejected pairs) ───────────────────

class HydraFakePreferenceDataset(Dataset):
    """
    Preserves mipo_3k.json chosen/rejected pairs for DPO preference optimization.
    Each sample is one pair (image + chosen + rejected), keeping all 3,480 pairs.
    """
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

                chosen_raw = _extract_reasoning(e.get("messages", []))
                rejected_raw = (e.get("rejected_response") or "").strip()
                if not chosen_raw or not rejected_raw:
                    continue

                eos = self.tokenizer.eos_token or "<|endoftext|>"
                chosen_enc = self.tokenizer(
                    chosen_raw + eos,
                    max_length=self.max_text_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                rejected_enc = self.tokenizer(
                    rejected_raw + eos,
                    max_length=self.max_text_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                self.samples.append({
                    "image_path":    resolved,
                    "label":         label,
                    "chosen_ids":    chosen_enc["input_ids"].squeeze(0),
                    "chosen_mask":   chosen_enc["attention_mask"].squeeze(0),
                    "rejected_ids":  rejected_enc["input_ids"].squeeze(0),
                    "rejected_mask": rejected_enc["attention_mask"].squeeze(0),
                    "chosen_text":   chosen_raw,
                    "rejected_text": rejected_raw,
                    "forgery_type":  e.get("type", "unknown"),
                })
                kept += 1

        print(f"[pref] {len(self.samples)} preference pairs loaded from {len(files)} file(s) (kept {kept})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        try:
            img = Image.open(s["image_path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.image_size, self.image_size), 0)

        pixel_values  = self.transform(img)
        freq_features = compute_fft_features(img, target_size=224)

        return {
            "pixel_values":  pixel_values,
            "freq_features": freq_features,
            "labels":        torch.tensor(s["label"], dtype=torch.long),
            "chosen_ids":    s["chosen_ids"],
            "chosen_mask":   s["chosen_mask"],
            "rejected_ids":  s["rejected_ids"],
            "rejected_mask": s["rejected_mask"],
            "forgery_type":  s["forgery_type"],
        }


# ── DataLoader Builders ───────────────────────────────────────────────────────

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


def build_classification_loader(
    dataset_root: str, json_root: str, tokenizer,
    image_size: int = 448, max_text_len: int = 512,
    batch_size: int = 8, num_workers: int = 8
):
    """Classification pool: train/all.json (48k unique, bare allowed)."""
    json_dir = os.path.join(json_root, "train")
    ds = HydraFakeDataset(
        dataset_root, json_dir, tokenizer, split="train",
        image_size=image_size, max_text_len=max_text_len,
        allowed_files=["all.json"], require_reasoning=False
    )
    return DataLoader(
        ds, batch_size=batch_size, sampler=_balanced_sampler(ds),
        num_workers=num_workers, pin_memory=True, drop_last=True
    ), ds


def build_sft_loader(
    dataset_root: str, json_root: str, tokenizer,
    image_size: int = 448, max_text_len: int = 512,
    batch_size: int = 8, num_workers: int = 8
):
    """SFT reasoning pool: train/sft_36k.json rich traces only (~35k usable)."""
    json_dir = os.path.join(json_root, "train")
    ds = HydraFakeDataset(
        dataset_root, json_dir, tokenizer, split="train",
        image_size=image_size, max_text_len=max_text_len,
        allowed_files=["sft_36k.json"], require_reasoning=True
    )
    return DataLoader(
        ds, batch_size=batch_size, sampler=_balanced_sampler(ds),
        num_workers=num_workers, pin_memory=True, drop_last=True
    ), ds


def build_preference_loader(
    dataset_root: str, json_root: str, tokenizer,
    image_size: int = 448, max_text_len: int = 512,
    batch_size: int = 8, num_workers: int = 8
):
    """Preference pool: train/mipo_3k.json (3,480 chosen/rejected pairs)."""
    json_dir = os.path.join(json_root, "train")
    ds = HydraFakePreferenceDataset(
        dataset_root, json_dir, tokenizer, image_size, max_text_len
    )
    return DataLoader(
        ds, batch_size=batch_size, sampler=_balanced_sampler(ds),
        num_workers=num_workers, pin_memory=True, drop_last=True
    ), ds


def build_separated_loaders(
    dataset_root:   str,
    json_root:      str,
    tokenizer_name: str = "./models/InternVL3-2B",
    image_size:     int = 448,
    max_text_len:   int = 512,
    batch_size:     int = 8,
    num_workers:    int = 8
):
    """
    Return separated loaders:
      - cls_train : 48k classification images
      - sft_train : 35k rich reasoning traces
      - pref      : 3,480 preference pairs (mipo_3k)
      - val / test
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    loaders = {}
    cls_loader, _ = build_classification_loader(
        dataset_root, json_root, tokenizer, image_size, max_text_len, batch_size, num_workers
    )
    loaders["cls_train"] = cls_loader

    sft_loader, _ = build_sft_loader(
        dataset_root, json_root, tokenizer, image_size, max_text_len, batch_size, num_workers
    )
    loaders["sft_train"] = sft_loader
    loaders["train"] = sft_loader  # for backward compatibility

    try:
        pref_loader, _ = build_preference_loader(
            dataset_root, json_root, tokenizer, image_size, max_text_len, batch_size, num_workers
        )
        loaders["pref"] = pref_loader
    except FileNotFoundError:
        pass

    for split in ("val", "test"):
        json_dir = os.path.join(json_root, split)
        if not os.path.isdir(json_dir):
            continue
        try:
            ds = HydraFakeDataset(
                dataset_root, json_dir, tokenizer, split, image_size, max_text_len
            )
        except FileNotFoundError as e:
            # e.g. jsons/val/* references hydrafake/val/* images that are not
            # present in this checkout -> skip validation instead of crashing
            # training (train.py guards on missing "val" loader).
            print(f"[{split}] skipped: {e}")
            continue
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

    return loaders, tokenizer
