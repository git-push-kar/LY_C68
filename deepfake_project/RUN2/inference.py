"""
inference.py
============
Run on a single image or folder after training.
No labels or dataset needed.

Usage:
    python inference.py \
        --checkpoint ./runs/exp1/checkpoints/best_model.pth \
        --image      test_image.png

    python inference.py \
        --checkpoint ./runs/exp1/checkpoints/best_model.pth \
        --image_dir  /path/to/folder
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer

from model import HydraFakeNet
from utils import load_checkpoint

LABEL = {0: "Real", 1: "Fake"}


def get_transform(image_size: int = 224):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(
            [0.48145466, 0.4578275,  0.40821073],
            [0.26862954, 0.26130258, 0.27577711],
        ),
    ])


@torch.no_grad()
def predict_image(model, image_path, transform, device, tokenizer):
    img    = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    bos_id = tokenizer.cls_token_id or 101
    eos_id = tokenizer.sep_token_id or 102
    result = model.predict(tensor, bos_id, eos_id, tokenizer)
    return {
        "image":      str(image_path),
        "prediction": LABEL[result["labels"][0]],
        "confidence": f"{result['confidences'][0] * 100:.1f}%",
        "reasoning":  result["reasoning"][0]
                      if isinstance(result["reasoning"][0], str)
                      else "No reasoning generated.",
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",       required=True)
    p.add_argument("--image",            default=None)
    p.add_argument("--image_dir",        default=None)
    p.add_argument("--tokenizer",        default="bert-base-uncased")
    p.add_argument("--image_size",       type=int, default=224)
    p.add_argument("--clip_model",
                   default="openai/clip-vit-base-patch32")
    p.add_argument("--model_dim",        type=int, default=512)
    p.add_argument("--cls_depth",        type=int, default=4)
    p.add_argument("--cls_heads",        type=int, default=8)
    p.add_argument("--cls_queries",      type=int, default=8)
    p.add_argument("--reasoning_hidden", type=int, default=512)
    p.add_argument("--reasoning_layers", type=int, default=2)
    return p.parse_args()


def main():
    args      = parse_args()
    device    = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    model = HydraFakeNet(
        clip_model_name   = args.clip_model,
        model_dim         = args.model_dim,
        cls_depth         = args.cls_depth,
        cls_heads         = args.cls_heads,
        cls_queries       = args.cls_queries,
        vocab_size        = tokenizer.vocab_size,
        reasoning_hidden  = args.reasoning_hidden,
        reasoning_layers  = args.reasoning_layers,
    ).to(device)

    load_checkpoint(args.checkpoint, model)
    model.eval()

    transform = get_transform(args.image_size)

    paths = []
    if args.image:
        paths.append(args.image)
    if args.image_dir:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        paths += [
            str(p) for p in Path(args.image_dir).rglob("*")
            if p.suffix.lower() in exts
        ]

    if not paths:
        print("No images provided. Use --image or --image_dir.")
        return

    for path in paths:
        r = predict_image(
            model, path, transform, device, tokenizer)
        print(f"\nImage      : {r['image']}")
        print(f"Prediction : {r['prediction']}  ({r['confidence']})")
        print(f"Reason     : {r['reasoning']}")
        print("─" * 60)


if __name__ == "__main__":
    main()