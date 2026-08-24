"""
dataset.py
==========
HydraFake dataset loader.

Fixes applied (vs previous version):
  #4  Tokenization moved out of __getitem__ → done at load time in _load().
      Previous: tokenizer called inside the DataLoader worker loop every step,
      creating a CPU bottleneck that kept the GPU idle between batches.
      Now: reasoning text is tokenized once at dataset construction. The
      DataLoader collates pre-computed int tensors — extremely fast.

  #7  Reasoning quality filter — samples with empty reasoning are still
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
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        parts = _TAG_RE.findall(content)
        if parts:
            return " ".join(p.strip() for p in parts)
        return _ANSWER_RE.sub("", content).strip() or content.strip()
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
        max_text_len: int = 192,
    ):
        self.dataset_root = dataset_root
        self.tokenizer    = tokenizer
        self.split        = split
        self.image_size   = image_size
        self.max_text_len = max_text_len
        self.transform    = build_transforms(split, image_size)
        self.samples      = []
        self._load(json_dir)

    def _load(self, json_dir: str):
        files = glob.glob(
            os.path.join(json_dir, "**", "*.json"), recursive=True)
        if not files:
            files = glob.glob(os.path.join(json_dir, "*.json"))
        if not files:
            raise FileNotFoundError(
                f"No JSON files found under: {json_dir}")

        missing_count  = 0
        found_count    = 0
        fallback_count = 0
        dupe_count     = 0
        seen_paths     = set()

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

                # The same image is listed across multiple JSON files
                # (all.json + per-generator files + sft/mipo/pgrpo).
                # Keep only the first occurrence.
                key = os.path.normcase(resolved)
                if key in seen_paths:
                    dupe_count += 1
                    continue
                seen_paths.add(key)

                # Skip entries whose image is missing on disk instead of
                # silently training on a black-image substitute.
                if not os.path.exists(resolved):
                    missing_count += 1
                    continue
                found_count += 1

                raw_reasoning = _extract_reasoning(e.get("messages", []))

                # Fix #7: sanitize empty / garbage reasoning
                reasoning = _sanitize_reasoning(raw_reasoning, label)
                if reasoning != raw_reasoning.strip():
                    fallback_count += 1

                # Fix #4: tokenize HERE at load time, not in __getitem__.
                # This runs once per sample in the main process during dataset
                # construction. The DataLoader then just collates pre-computed
                # int tensors — no tokenizer CPU overhead per training step.
                enc = self.tokenizer(
                    reasoning,
                    max_length=self.max_text_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )

                self.samples.append({
                    "image_path":     resolved,
                    "label":          label,
                    "reasoning_ids":  enc["input_ids"].squeeze(0),       # (max_text_len,)
                    "attention_mask": enc["attention_mask"].squeeze(0),  # (max_text_len,)
                    "reasoning_text": reasoning,   # kept for logging/debug only
                    "forgery_type":   e.get("type", "unknown"),
                })

        print(f"[{self.split}] {len(self.samples)} samples "
              f"from {len(files)} JSON file(s). "
              f"Found: {found_count}  Missing(skipped): {missing_count}  "
              f"Duplicates(skipped): {dupe_count}  "
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
    max_text_len:   int = 192,
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
