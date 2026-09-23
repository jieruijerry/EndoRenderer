#!/usr/bin/env python3
"""Download official C3VD zips from the project-page Google Drive links."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path(__file__).with_name("c3vd_manifest.json")
DEFAULT_ROOT = Path("/media/jerry/HDD2/datasets/c3vd")


def load_manifest() -> dict:
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def _is_zip(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 64:
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            return zf.namelist() is not None
    except zipfile.BadZipFile:
        return False


def already_extracted(out_dir: Path) -> bool:
    marker = out_dir / ".extracted"
    if marker.exists():
        return True
    if out_dir.exists() and (out_dir / "pose.txt").exists() and (out_dir / "coverage_mesh.obj").exists():
        return True
    return False


def download_drive(file_id: str, dest: Path, attempts: int = 6) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        if dest.suffix == ".zip" and not _is_zip(dest):
            print(f"[purge] corrupt {dest.name}")
            dest.unlink()
        else:
            print(f"[skip] {dest.name} ({dest.stat().st_size / 1e9:.2f} GB)")
            return
    print(f"[download] {dest.name}  id={file_id}")
    last_err: Exception | None = None
    for i in range(1, attempts + 1):
        try:
            import gdown

            gdown.download(id=file_id, output=str(dest), quiet=False, fuzzy=True)
            if dest.exists() and dest.stat().st_size > 0:
                return
            raise RuntimeError(f"empty download: {dest.name}")
        except Exception as exc:
            last_err = exc
            dest.unlink(missing_ok=True)
            print(f"[retry {i}/{attempts}] {dest.name}: {exc}")
            time.sleep(min(4 * i, 30))
            if i == attempts:
                _curl_drive(file_id, dest)
                return
    if last_err:
        raise last_err


def _curl_drive(file_id: str, dest: Path) -> None:
    url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
    print(f"[curl] {dest.name}")
    subprocess.check_call(
        [
            "curl",
            "-L",
            "--http1.1",
            "--retry",
            "5",
            "--retry-delay",
            "5",
            "-A",
            "Mozilla/5.0",
            "-o",
            str(dest),
            url,
        ]
    )
    if not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(f"curl failed: {dest.name}")


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    if already_extracted(out_dir):
        print(f"[skip] extracted {out_dir.name}")
        return
    if not _is_zip(zip_path):
        print(f"[purge] not a zip: {zip_path.name}")
        zip_path.unlink(missing_ok=True)
        raise RuntimeError(f"corrupt archive {zip_path.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {zip_path.name} -> {out_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir.parent if _zip_has_toplevel(zf, out_dir.name) else out_dir)
    (out_dir / ".extracted").write_text(zip_path.name + "\n")


def _zip_has_toplevel(zf: zipfile.ZipFile, name: str) -> bool:
    return any(n == f"{name}/" or n.startswith(f"{name}/") for n in zf.namelist()[:8])


def symlink_into_repo(data_root: Path) -> None:
    link = REPO_ROOT / "data" / "c3vd"
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        if link.is_symlink() and os.path.realpath(link) == os.path.realpath(data_root):
            return
        if link.is_symlink() or link.is_file():
            link.unlink()
        else:
            raise RuntimeError(f"{link} exists and is not a symlink")
    os.symlink(data_root, link, target_is_directory=True)
    print(f"[link] {link} -> {data_root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--no-screening", action="store_true")
    parser.add_argument("--models-only", action="store_true")
    parser.add_argument("--skip", action="append", default=[], help="zip/obj filename to skip")
    args = parser.parse_args()

    root: Path = args.root
    official = root / "official"
    zips = root / "zips"
    official.mkdir(parents=True, exist_ok=True)
    zips.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    groups: list[tuple[str, list[dict]]] = [
        ("models", manifest["files"]["models"]),
        ("calibration", manifest["files"]["calibration"]),
    ]
    if not args.models_only:
        seqs = sorted(manifest["files"]["sequences"], key=lambda x: x.get("gb", 0))
        groups.append(("sequences", seqs))
        if not args.no_screening:
            groups.append(("screening", manifest["files"]["screening"]))

    for group, items in groups:
        for item in items:
            name = item["name"]
            if name in args.skip:
                print(f"[skip] requested {name}")
                continue
            dest = (official / "models" / name) if group == "models" else (zips / name)
            if group == "sequences" and already_extracted(official / name.replace(".zip", "")):
                print(f"[skip] {name} already extracted")
                continue
            download_drive(item["id"], dest)
            if name.endswith(".zip"):
                stem = name[:-4]
                extract_dir = official / ("calibration" if group == "calibration" else "") / stem
                if group == "calibration":
                    extract_dir = official / "calibration" / stem
                elif group == "screening":
                    extract_dir = official / "screening" / stem
                else:
                    extract_dir = official / stem
                extract_zip(dest, extract_dir)

    symlink_into_repo(root)
    print("done.")


if __name__ == "__main__":
    main()
