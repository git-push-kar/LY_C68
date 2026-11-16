"""
fetch_detector_weights.py
=========================
Reproducible fetch for the YOLO face detector, with size + SHA-256 pinning.

The multi-face pipeline routes images through this detector, so the exact
weights matter: a silently different checkpoint would change which faces are
cropped and therefore every downstream verdict. This script makes the artifact
reproducible and refuses to leave a mismatched file in place.

Source (AGPL-3.0, WIDERFace-derived fine-tune):
    repo : arnabdhar/YOLOv8-Face-Detection
    file : model.pt
    url  : https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/resolve/main/model.pt

The weights live under ``models/`` (gitignored), so they are fetched, never
committed. ``models/yolov8n-face.pt`` is the local name used by config.yaml.

Usage:
    python fetch_detector_weights.py            # download if absent, verify either way
    python fetch_detector_weights.py --check    # verify only, no network
    python fetch_detector_weights.py --force    # re-download even if present
"""

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request

SOURCE_REPO = "arnabdhar/YOLOv8-Face-Detection"
SOURCE_FILE = "model.pt"
SOURCE_URL = (
    "https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/resolve/main/model.pt"
)
SOURCE_LICENSE = "AGPL-3.0"

# Pinned artifact identity (verified 2026-11-16).
FILENAME = "yolov8n-face.pt"
SIZE_BYTES = 6_247_065
SHA256 = "37396ac6a9601ab9f5177e4231b09d81cf6f65a7f22db99ec3b36ab63f674e71"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path, label="weights"):
    """Return True only if size AND digest both match the pin."""
    if not os.path.isfile(path):
        print(f"[{label}] missing: {path}")
        return False
    size = os.path.getsize(path)
    if size != SIZE_BYTES:
        print(f"[{label}] size mismatch: {size} != {SIZE_BYTES} bytes")
        return False
    digest = sha256_of(path)
    if digest != SHA256:
        print(f"[{label}] sha256 mismatch:\n  got      {digest}\n  expected {SHA256}")
        return False
    print(f"[{label}] OK  {os.path.basename(path)}  {size} bytes  sha256 {digest}")
    return True


def download(dest):
    """Stream to a temp file, verify, then atomically move into place."""
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".part", dir=os.path.dirname(os.path.abspath(dest)))
    os.close(tmp_fd)
    try:
        print(f"Downloading {SOURCE_REPO}/{SOURCE_FILE}")
        print(f"  -> {SOURCE_URL}")
        with urllib.request.urlopen(SOURCE_URL, timeout=120) as resp, open(tmp_path, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e6:6.2f} / {total / 1e6:.2f} MB", end="", flush=True)
        print()
        if not verify(tmp_path, label="download"):
            raise SystemExit(
                "ERROR: downloaded file does not match the pinned size/SHA-256. "
                "The upstream artifact changed; inspect it and update the pin "
                "deliberately instead of using it blindly."
            )
        os.replace(tmp_path, dest)
        print(f"Wrote {dest}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main(argv=None):
    p = argparse.ArgumentParser(description="Fetch and verify the YOLO face detector weights")
    p.add_argument("--dest", default=os.path.join("models", FILENAME),
                   help=f"destination path (default: models/{FILENAME})")
    p.add_argument("--check", action="store_true",
                   help="verify the existing file only; never download")
    p.add_argument("--force", action="store_true",
                   help="re-download even if a valid file is present")
    args = p.parse_args(argv)

    print(f"Pinned artifact : {SOURCE_REPO}/{SOURCE_FILE} ({SOURCE_LICENSE})")
    print(f"Pin             : {SIZE_BYTES} bytes  sha256 {SHA256}")

    if args.check:
        return 0 if verify(args.dest) else 1

    if os.path.isfile(args.dest) and not args.force:
        if verify(args.dest):
            print("Already present and verified; nothing to do (use --force to refetch).")
            return 0
        print("Existing file is invalid; re-downloading.")

    download(args.dest)
    return 0 if verify(args.dest) else 1


if __name__ == "__main__":
    sys.exit(main())
