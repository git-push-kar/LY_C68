"""
inference.py
============
Run inference on a single image or a directory of images.

Usage:
    python inference.py \
        --checkpoint ./runs/exp1/checkpoints/best_model.pth \
        --image      test_image.png \
        [--image_dir /path/to/folder]

Output example:
    Prediction: Fake  (confidence: 92.3%)

    Reason:
    Lighting inconsistencies and unrealistic skin texture detected.
"""

import argparse
import os
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer

from model import HydraFakeNet
from utils import load_checkpoint

# ---------------------------------------------------------------------------
LABEL_NAMES = {0: "Real", 1: "Fake"}
# ---------------------------------------------------------------------------


def build_inference_transform(image_size: int = 224):
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def load_model(checkpoint_path: str, args, device: torch.device) -> HydraFakeNet:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = HydraFakeNet(
        image_size=args.image_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        vocab_size=tokenizer.vocab_size,
        reasoning_hidden_dim=args.embed_dim,
        max_reasoning_len=args.max_text_len,
    ).to(device)
    load_checkpoint(checkpoint_path, model)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def predict_single(
    model: HydraFakeNet,
    image_path: str,
    transform,
    device: torch.device,
    tokenizer,
    max_text_len: int = 128,
) -> dict:
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)  # (1, 3, H, W)

    bos_id = tokenizer.cls_token_id or 101
    eos_id = tokenizer.sep_token_id or 102

    result = model.predict(tensor, bos_id=bos_id, eos_id=eos_id, tokenizer=tokenizer)

    label_id = result["labels"][0]
    confidence = result["confidences"][0]
    reasoning = result["reasoning"][0] if isinstance(result["reasoning"][0], str) else ""

    return {
        "image": image_path,
        "label": LABEL_NAMES[label_id],
        "confidence": confidence,
        "reasoning": reasoning,
    }


def format_output(result: dict) -> str:
    lines = [
        f"\nImage     : {result['image']}",
        f"Prediction: {result['label']}  (confidence: {result['confidence']*100:.1f}%)",
        f"\nReason:",
        f"  {result['reasoning'] or 'No reasoning generated.'}",
    ]
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(description="HydraFake Inference")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", default=None, help="Path to a single image.")
    p.add_argument("--image_dir", default=None, help="Path to a directory of images.")
    p.add_argument("--tokenizer", default="bert-base-uncased")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--max_text_len", type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model(args.checkpoint, args, device)
    transform = build_inference_transform(args.image_size)

    # Collect image paths
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    if args.image_dir:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
        image_paths.extend(
            str(p) for p in Path(args.image_dir).rglob("*") if p.suffix.lower() in exts
        )

    if not image_paths:
        print("No images provided. Use --image or --image_dir.")
        return

    for path in image_paths:
        result = predict_single(model, path, transform, device, tokenizer, args.max_text_len)
        print(format_output(result))
        print("─" * 60)


if __name__ == "__main__":
    main()
