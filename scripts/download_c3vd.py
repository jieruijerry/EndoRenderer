#!/usr/bin/env python3
"""Download C3VD onto HDD2 and symlink it into this repo.

Talks only to https://hf-mirror.com (official Google Drive / huggingface.co
time out on this machine). Source: BeileiCui/C3VD, ~47 GB, 22 sequences.
Already-downloaded files are skipped so the job can resume.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("/media/jerry/HDD2/datasets/c3vd")
HF_REPO = "BeileiCui/C3VD"
HF_ENDPOINT = "https://hf-mirror.com"
UA = "EndoRenderer-c3vd/1.0"


def _request(url: str, timeout: int = 120):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=timeout)


def parse_next_link(link_header: str) -> str | None:
    if not link_header:
        return None
    for part in link_header.split(","):
        if 'rel="next"' in part:
            m = re.search(r"<([^>]+)>", part)
            if m:
                return m.group(1)
    return None


def _rewrite_mirror(url: str) -> str:
    return url.replace("https://huggingface.co", HF_ENDPOINT).replace(
        "http://huggingface.co", HF_ENDPOINT
    )


def list_repo_files() -> list[str]:
    # One-shot siblings list avoids /tree pagination, whose Link header
    # often points at huggingface.co even when the first page came from the mirror.
    url = f"{HF_ENDPOINT}/api/datasets/{HF_REPO}"
    print(f"[list] {url}")
    with _request(url, timeout=60) as resp:
        payload = json.load(resp)
    files = [s["rfilename"] for s in payload.get("siblings") or [] if s.get("rfilename")]
    if files:
        print(f"[list] {len(files)} files from siblings")
        return files

    files = []
    url = (
        f"{HF_ENDPOINT}/api/datasets/{HF_REPO}/tree/main"
        "?recursive=true&expand=false&limit=1000"
    )
    print(f"[list] fallback {url}")
    while url:
        with _request(url, timeout=60) as resp:
            items = json.load(resp)
            nxt = parse_next_link(resp.headers.get("Link", ""))
        for item in items:
            if item.get("type") == "file":
                files.append(item["path"])
        print(f"[list] {len(files)} files so far")
        url = _rewrite_mirror(nxt) if nxt else None
    return files


def download_one(rel_path: str, dest_root: Path) -> str:
    dest = dest_root / rel_path
    if dest.exists() and dest.stat().st_size > 0:
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    url = f"{HF_ENDPOINT}/datasets/{HF_REPO}/resolve/main/{quote(rel_path)}"
    try:
        with _request(url) as resp, open(part, "wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        if part.stat().st_size == 0:
            part.unlink(missing_ok=True)
            return f"empty:{rel_path}"
        part.replace(dest)
        return "ok"
    except Exception as exc:
        part.unlink(missing_ok=True)
        return f"fail:{rel_path}:{exc}"


def symlink_into_repo(data_root: Path) -> None:
    link = REPO_ROOT / "data" / "c3vd"
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        if link.is_symlink() and os.path.realpath(link) == os.path.realpath(data_root):
            print(f"[skip] symlink {link} -> {data_root}")
            return
        if link.is_symlink() or link.is_file():
            link.unlink()
        else:
            raise RuntimeError(f"{link} exists and is not a symlink")
    os.symlink(data_root, link, target_is_directory=True)
    print(f"[link] {link} -> {data_root}")


def main():
    parser = argparse.ArgumentParser(description="Download C3VD onto HDD2")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--geometry-only",
        action="store_true",
        help="only meshes and poses (fast); skip RGB/depth",
    )
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    root: Path = args.root
    root.mkdir(parents=True, exist_ok=True)

    files = list_repo_files()
    if args.geometry_only:
        files = [
            p
            for p in files
            if p.endswith(("coverage_mesh.obj", "pose.txt", "README.md"))
        ]
        print(f"[mode] geometry only: {len(files)} files")
    else:
        print(f"[download] {len(files)} files from {HF_REPO} via {HF_ENDPOINT}")

    counts = {"skip": 0, "ok": 0, "fail": 0}
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(download_one, p, root): p for p in files}
        done = 0
        for fut in as_completed(futs):
            done += 1
            result = fut.result()
            key = result.split(":", 1)[0]
            counts[key] = counts.get(key, 0) + 1
            if key == "fail" or key == "empty":
                failures.append(result)
            if done % 50 == 0 or key != "skip":
                print(
                    f"[{done}/{len(files)}] ok={counts.get('ok', 0)} "
                    f"skip={counts.get('skip', 0)} fail={counts.get('fail', 0)} "
                    f"last={result}",
                    flush=True,
                )

    symlink_into_repo(root)
    if failures:
        print(f"{len(failures)} failures:", file=sys.stderr)
        for line in failures[:20]:
            print(" ", line, file=sys.stderr)
        sys.exit(1)
    print(
        f"done. ok={counts.get('ok', 0)} skip={counts.get('skip', 0)} "
        f"fail={counts.get('fail', 0)}"
    )


if __name__ == "__main__":
    main()
