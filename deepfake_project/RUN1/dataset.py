"""
dataset.py
==========
HydraFake dataset: parses JSON metadata, loads images, tokenizes reasoning text.

Usage:
    from dataset import HydraFakeDataset, build_dataloaders
"""

import os
import json
import re
import glob
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ANSWER_RE = re.compile(r"<answer>\s*(real|fake)\s*</answer>", re.IGNORECASE)
_TAG_RE = re.compile(
    r"<(?:planning|reasoning|reflection|conclusion|fast)>(.*?)"
    r"</(?:planning|reasoning|reflection|conclusion|fast)>",
    re.DOTALL | re.IGNORECASE,
)


def _extract_reasoning_text(messages: list[dict]) -> str:
    """
    Concatenate structured reasoning tags + answer from the assistant turn.
    Falls back to the raw assistant content if no tags are found.
    """
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        # Collect all tagged sections
        parts = _TAG_RE.findall(content)
        if parts:
            reasoning = " ".join(p.strip() for p in parts)
        else:
            # Fall back: strip the <answer> tag and keep remaining text
            reasoning = _ANSWER_RE.sub("", content).strip()
        if not reasoning:
            # Absolute fallback: keep whole content
            reasoning = content.strip()
        return reasoning
    return ""


def _resolve_image_path(raw_path: str, dataset_root: str) -> str:
    """
    Strips the leading 'hydrafake/' prefix and joins with dataset_root.

    Example:
        raw_path     = 'hydrafake/train/real/01508.png'
        dataset_root = '/data/hydrafake'
        returns      = '/data/hydrafake/train/real/01508.png'
    """
    # Remove the 'hydrafake/' prefix (first component)
    parts = Path(raw_path).parts
    if parts[0].lower() == "hydrafake":
        relative = os.path.join(*parts[1:])
    else:
        relative = raw_path
    return os.path.join(dataset_root, relative)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transforms(split: str = "train", image_size: int = 224):
    """
    Returns torchvision transform pipeline.
    Training split uses augmentation; val/test use deterministic resize + crop.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if split == "train":
        return T.Compose(
            [
                T.Resize((image_size + 32, image_size + 32)),
                T.RandomCrop(image_size),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                T.RandomGrayscale(p=0.05),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
                T.RandomErasing(p=0.1, scale=(0.02, 0.1)),
            ]
        )
    else:
        return T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HydraFakeDataset(Dataset):
    """
    PyTorch Dataset for HydraFake.

    Loads all JSON files under `json_dir`, resolves image paths relative
    to `dataset_root`, and tokenises the reasoning text.

    Args:
        dataset_root : str  – root directory of the hydrafake dataset.
        json_dir     : str  – directory containing *.json files.
        tokenizer    : HuggingFace tokenizer instance.
        split        : 'train' | 'val' | 'test'.
        image_size   : int  – side length for image resize.
        max_text_len : int  – max tokenised reasoning length.
    """

    def __init__(
        self,
        dataset_root: str,
        json_dir: str,
        tokenizer,
        split: str = "train",
        image_size: int = 224,
        max_text_len: int = 128,
    ):
        self.dataset_root = dataset_root
        self.tokenizer = tokenizer
        self.split = split
        self.image_size = image_size
        self.max_text_len = max_text_len
        self.transform = build_transforms(split, image_size)

        self.samples: list[dict] = []
        self._load_json_files(json_dir)

    # ------------------------------------------------------------------
    def _load_json_files(self, json_dir: str):
        pattern = os.path.join(json_dir, "**", "*.json")
        json_files = glob.glob(pattern, recursive=True)

        if not json_files:
            # Also try non-recursive one level deep
            json_files = glob.glob(os.path.join(json_dir, "*.json"))

        if not json_files:
            raise FileNotFoundError(
                f"No JSON files found under: {json_dir}\n"
                "Check that json_dir points to the correct sub-directory."
            )

        for jf in sorted(json_files):
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Support both a single dict and a list of dicts
            entries = data if isinstance(data, list) else [data]

            for entry in entries:
                img_paths = entry.get("images", [])
                if not img_paths:
                    continue

                # Use the first image in the list
                raw_path = img_paths[0]
                abs_path = _resolve_image_path(raw_path, self.dataset_root)

                label = int(entry.get("label", 0))
                reasoning = _extract_reasoning_text(entry.get("messages", []))
                forgery_type = entry.get("type", "unknown")

                self.samples.append(
                    {
                        "image_path": abs_path,
                        "label": label,
                        "reasoning": reasoning,
                        "forgery_type": forgery_type,
                    }
                )

        print(
            f"[{self.split}] Loaded {len(self.samples)} samples from {len(json_files)} JSON file(s)."
        )

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        # ── Load image ──────────────────────────────────────────────────
        try:
            img = Image.open(sample["image_path"]).convert("RGB")
        except Exception as e:
            # If file is missing return a blank tensor so collation doesn't crash
            print(f"[WARNING] Could not load {sample['image_path']}: {e}")
            img = Image.new("RGB", (self.image_size, self.image_size), color=0)

        image_tensor = self.transform(img)  # (3, H, W)

        # ── Tokenise reasoning ──────────────────────────────────────────
        encoding = self.tokenizer(
            sample["reasoning"],
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)     # (T,)
        attention_mask = encoding["attention_mask"].squeeze(0)  # (T,)

        return {
            "image": image_tensor,                       # (3, H, W)
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "reasoning_ids": input_ids,                  # (T,)
            "attention_mask": attention_mask,            # (T,)
            "image_path": sample["image_path"],
        }


# ---------------------------------------------------------------------------
# Balanced sampler helper
# ---------------------------------------------------------------------------

def make_balanced_sampler(dataset: HydraFakeDataset) -> WeightedRandomSampler:
    """
    Creates a WeightedRandomSampler so real/fake classes are seen equally.
    """
    labels = [s["label"] for s in dataset.samples]
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[torch.tensor(labels)]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    dataset_root: str,
    json_root: str,
    tokenizer_name: str = "bert-base-uncased",
    image_size: int = 224,
    max_text_len: int = 128,
    batch_size: int = 32,
    num_workers: int = 4,
    use_balanced_sampler: bool = True,
) -> dict[str, DataLoader]:
    """
    Builds DataLoaders for train, val, and test splits.

    Args:
        dataset_root         : root directory containing train/val/test folders.
        json_root            : root directory containing jsons/train|val|test.
        tokenizer_name       : HuggingFace model name for the tokenizer.
        image_size           : image resolution.
        max_text_len         : max reasoning token length.
        batch_size           : batch size.
        num_workers          : dataloader workers.
        use_balanced_sampler : balance real/fake for training.

    Returns:
        dict with keys 'train', 'val', 'test'.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    loaders = {}
    for split in ("train", "val", "test"):
        json_dir = os.path.join(json_root, split)
        if not os.path.isdir(json_dir):
            print(f"[INFO] No JSON directory for split '{split}': {json_dir}")
            continue

        ds = HydraFakeDataset(
            dataset_root=dataset_root,
            json_dir=json_dir,
            tokenizer=tokenizer,
            split=split,
            image_size=image_size,
            max_text_len=max_text_len,
        )

        if split == "train" and use_balanced_sampler:
            sampler = make_balanced_sampler(ds)
            loader = DataLoader(
                ds,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
            )
        else:
            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers,
                pin_memory=True,
                drop_last=(split == "train"),
            )

        loaders[split] = loader

    return loaders, tokenizer
