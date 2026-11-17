"""Verify (and optionally fetch) the weight files the paper depends on.

The two checkpoints are 8.5 GB together and the adapter set is another 180 MB, so
they are not regular git objects. `checkpoints_manifest.json` records what each
file is and its SHA-256; this script checks a local copy against that contract.

Usage
    python verify_checkpoints.py                       # verify what is on disk
    python verify_checkpoints.py --download-base URL   # fetch missing files
    python verify_checkpoints.py --json

URL layout expected by --download-base: the script appends the path recorded in
`restore_to` (e.g. .../runs/intern_v2_ground/checkpoints/best.pth). Use whatever
mirror you host the weights on — an object-store prefix, a release asset, a shared
drive exported over HTTP.

Exit codes: 0 all verified, 1 a hash/size mismatch, 2 a file is missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(ROOT, "checkpoints_manifest.json")
CHUNK = 1 << 24


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download(base: str, rel: str, dest: str) -> None:
    url = base.rstrip("/") + "/" + urllib.parse.quote(rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    print(f"  fetching {url}")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            block = resp.read(CHUNK)
            if not block:
                break
            out.write(block)
            done += len(block)
            if total:
                print(f"\r    {done / 1e9:.2f}/{total / 1e9:.2f} GB", end="", flush=True)
    print()
    os.replace(tmp, dest)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--download-base", default=None,
                    help="URL prefix; missing files are fetched from <base>/<restore_to>")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    results = []
    status = 0

    for entry in manifest["weights"]:
        dest = os.path.join(ROOT, entry["restore_to"])
        item = {"path": entry["restore_to"], "role": entry["role"]}
        if not os.path.exists(dest):
            if args.download_base:
                try:
                    download(args.download_base, entry["restore_to"], dest)
                except Exception as exc:                       # noqa: BLE001
                    item.update(state="missing", detail=f"download failed: {exc}")
                    results.append(item)
                    status = max(status, 2)
                    continue
            else:
                item.update(state="missing", detail="not on disk")
                results.append(item)
                status = max(status, 2)
                continue

        size = os.path.getsize(dest)
        item["bytes"] = size
        if entry.get("bytes") and size != entry["bytes"]:
            item.update(state="size-mismatch", expected=entry["bytes"])
            results.append(item)
            status = max(status, 1)
            continue

        digest = sha256(dest)
        item["sha256"] = digest
        if digest != entry["sha256"]:
            item.update(state="hash-mismatch", expected=entry["sha256"])
            results.append(item)
            status = max(status, 1)
            continue

        item.update(state="ok")
        results.append(item)

    if args.json:
        print(json.dumps({"status": status, "files": results}, indent=2))
    else:
        for r in results:
            mark = {"ok": "OK   ", "missing": "MISS ", "size-mismatch": "SIZE ",
                    "hash-mismatch": "HASH "}[r["state"]]
            print(f"{mark} {r['path']}")
            if r["state"] != "ok":
                print(f"       {r.get('detail') or 'expected ' + str(r.get('expected'))}")
                print(f"       role: {r['role']}")
        bad = sum(1 for r in results if r["state"] != "ok")
        print(f"\n{len(results) - bad}/{len(results)} verified"
              + ("" if not bad else f"; {bad} problem(s)"))
        if bad:
            print("See checkpoints_manifest.json for the expected hashes, or pass "
                  "--download-base to fetch them.")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
