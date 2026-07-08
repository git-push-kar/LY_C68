"""
dataset.py
==========
HydraFake dataset — JSON parser, image loader, tokenizer, DataLoader factory.
Uses CLIP normalization stats.
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
_TAG_RE    = re.compile(
    r"<(?:planning|reasoning|reflection|conclusion|fast)>(.*?)"
    r"</(?:planning|reasoning|reflection|conclusion|fast)>",
    re.DOTALL | re.IGNORECASE,
)


def _extract_reasoning(messages: list) -> str:
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        parts   = _TAG_RE.findall(content)
        if parts:
            return " ".join(p.strip() for p in parts)
        return _ANSWER_RE.sub("", content).strip() or content.strip()
    return ""


def _resolve_path(raw: str, root: str) -> str:
    parts = Path(raw).parts
    rel   = os.path.join(*parts[1:]) \
            if parts[0].lower() == "hydrafake" else raw
    return os.path.join(root, rel)


def build_transforms(split: str = "train", image_size: int = 224):
    # CLIP normalization stats — must match what CLIP was pretrained with
    mean = [0.48145466, 0.4578275,  0.40821073]
    std  = [0.26862954, 0.26130258, 0.27577711]

    if split == "train":
        return T.Compose([
            T.Resize((image_size + 32, image_size + 32)),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2,
                          saturation=0.1, hue=0.05),
            T.RandomGrayscale(p=0.05),
            T.ToTensor(),
            T.Normalize(mean, std),
            T.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])


class HydraFakeDataset(Dataset):
    def __init__(self, dataset_root: str, json_dir: str, tokenizer,
                 split: str = "train", image_size: int = 224,
                 max_text_len: int = 128):
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

        for jf in sorted(files):
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
            entries = data if isinstance(data, list) else [data]
            for e in entries:
                imgs = e.get("images", [])
                if not imgs:
                    continue
                self.samples.append({
                    "image_path":   _resolve_path(
                                        imgs[0], self.dataset_root),
                    "label":        int(e.get("label", 0)),
                    "reasoning":    _extract_reasoning(
                                        e.get("messages", [])),
                    "forgery_type": e.get("type", "unknown"),
                })

        print(f"[{self.split}] {len(self.samples)} samples "
              f"from {len(files)} JSON file(s).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        try:
            img = Image.open(s["image_path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB",
                            (self.image_size, self.image_size), 0)

        image_tensor = self.transform(img)

        enc = self.tokenizer(
            s["reasoning"],
            max_length     = self.max_text_len,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )
        return {
            "image":          image_tensor,
            "label":          torch.tensor(s["label"], dtype=torch.long),
            "reasoning_ids":  enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


def _balanced_sampler(dataset: HydraFakeDataset) -> WeightedRandomSampler:
    labels        = [s["label"] for s in dataset.samples]
    class_counts  = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[torch.tensor(labels)]
    return WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True)


def build_dataloaders(
    dataset_root:   str,
    json_root:      str,
    tokenizer_name: str = "bert-base-uncased",
    image_size:     int = 224,
    max_text_len:   int = 128,
    batch_size:     int = 64,
    num_workers:    int = 8,
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    loaders   = {}

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
                batch_size  = batch_size,
                sampler     = _balanced_sampler(ds),
                num_workers = num_workers,
                pin_memory  = True,
                drop_last   = True,
            )
        else:
            loader = DataLoader(
                ds,
                batch_size  = batch_size,
                shuffle     = False,
                num_workers = num_workers,
                pin_memory  = True,
            )
        loaders[split] = loader

    return loaders, tokenizer