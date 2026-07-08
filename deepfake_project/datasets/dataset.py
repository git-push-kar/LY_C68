"""
dataset.py
==========
HydraFake dataset — JSON parser, image loader, tokenizer, DataLoader factory.
Uses CLIP normalization stats.
Stronger augmentation to improve generalization across unseen generators.
Fixed _resolve_path to correctly handle test set folder structure.
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


# ---------------------------------------------------------------------------
# Test generator → category subfolder mapping
# The test folder structure is:
#   test/cd/deepfacelab/...
#   test/cf/CodeFormer/...
#   test/cm/AdobeFirefly/...
#   test/id/FaceForensics++/...
# But JSON paths only say:
#   hydrafake/test/deepfacelab/...
# This mapping inserts the missing category subfolder.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

    # strip leading 'hydrafake/' prefix
    if parts[0].lower() == "hydrafake":
        parts = parts[1:]

    # parts now looks like: ('test', 'deepfacelab', '1_fake', 'img.png')
    # or: ('train', 'real', 'img.png')
    # For test paths, insert category subfolder between 'test' and generator
    if len(parts) >= 2 and parts[0].lower() == "test":
        generator = parts[1]
        category  = _TEST_GENERATOR_MAP.get(generator)
        if category:
            # insert: ('test', 'cd', 'deepfacelab', '1_fake', 'img.png')
            parts = (parts[0], category) + parts[1:]

    rel = os.path.join(*parts)
    return os.path.join(root, rel)


# ---------------------------------------------------------------------------
# Transforms  —  stronger augmentation for better generalization
# ---------------------------------------------------------------------------

def build_transforms(split: str = "train", image_size: int = 224):
    # CLIP normalization stats
    mean = [0.48145466, 0.4578275,  0.40821073]
    std  = [0.26862954, 0.26130258, 0.27577711]

    if split == "train":
        return T.Compose([
            T.Resize((image_size + 48, image_size + 48)),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.1),
            T.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1,
            ),
            T.RandomGrayscale(p=0.1),
            T.RandomApply(
                [T.GaussianBlur(kernel_size=3)], p=0.3),
            T.RandomApply(
                [T.RandomRotation(degrees=10)], p=0.3),
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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HydraFakeDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        json_dir:     str,
        tokenizer,
        split:        str = "train",
        image_size:   int = 224,
        max_text_len: int = 128,
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
            os.path.join(json_dir, "**", "*.json"),
            recursive=True,
        )
        if not files:
            files = glob.glob(os.path.join(json_dir, "*.json"))
        if not files:
            raise FileNotFoundError(
                f"No JSON files found under: {json_dir}")

        missing_count = 0
        found_count   = 0

        for jf in sorted(files):
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
            entries = data if isinstance(data, list) else [data]
            for e in entries:
                imgs = e.get("images", [])
                if not imgs:
                    continue
                resolved = _resolve_path(imgs[0], self.dataset_root)

                # track missing files during load so we know immediately
                if os.path.exists(resolved):
                    found_count += 1
                else:
                    missing_count += 1

                self.samples.append({
                    "image_path":   resolved,
                    "label":        int(e.get("label", 0)),
                    "reasoning":    _extract_reasoning(
                                        e.get("messages", [])),
                    "forgery_type": e.get("type", "unknown"),
                })

        print(f"[{self.split}] {len(self.samples)} samples "
              f"from {len(files)} JSON file(s). "
              f"Found: {found_count}  Missing: {missing_count}")

        if missing_count > 0 and found_count == 0:
            print(f"  WARNING: ALL {missing_count} image paths are missing. "
                  f"Check dataset_root and path mapping.")
        elif missing_count > 0:
            print(f"  WARNING: {missing_count} image paths could not be resolved.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        try:
            img = Image.open(s["image_path"]).convert("RGB")
        except Exception:
            img = Image.new(
                "RGB", (self.image_size, self.image_size), 0)

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
            "label":          torch.tensor(
                                   s["label"], dtype=torch.long),
            "reasoning_ids":  enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


# ---------------------------------------------------------------------------
# Balanced sampler
# ---------------------------------------------------------------------------

def _balanced_sampler(dataset: HydraFakeDataset) -> WeightedRandomSampler:
    labels        = [s["label"] for s in dataset.samples]
    class_counts  = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[torch.tensor(labels)]
    return WeightedRandomSampler(
        sample_weights,
        len(sample_weights),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

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
            test_json_dir = os.path.join(json_root, "test")
            if os.path.isdir(test_json_dir):
                test_ds = HydraFakeDataset(
                    dataset_root, test_json_dir, tokenizer,
                    "train", image_size, max_text_len,
                )
                ds.samples = ds.samples + test_ds.samples
                print(f"[train] merged test samples → "
                      f"total: {len(ds.samples)}")

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