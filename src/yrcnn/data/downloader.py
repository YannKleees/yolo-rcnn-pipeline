"""
Download & prepare the *full* DENTEX dataset.

The Kaggle mirror now lives at:
    https://www.kaggle.com/datasets/truthisneverlinear/dentex-challenge-2023
If the on‑line download fails (no credentials / no network) we still fall back
to a tiny 5‑image demo‑set so tests keep running green.
"""
from __future__ import annotations
import logging, shutil, zipfile
from pathlib import Path
from typing import Optional
import hashlib, json, os

import yaml

log = logging.getLogger(__name__)

DATA_DIR = Path("data")
RAW_DIR  = DATA_DIR / "dentex"
CFG_DIR  = Path("configs")
CFG_DIR.mkdir(exist_ok=True, parents=True)


# ------------------------------------------------------------------ #
def _sha256(fp: Path) -> str:
    return hashlib.sha256(fp.read_bytes()).hexdigest()


# ------------------------------------------------------------------ #
def _download_kaggle(tmp: Path) -> bool:
    """
    Download **and extract** the whole dataset from Kaggle.

    Returns *True* on success so the caller can skip the HF fallback.
    """
    try:
        import kaggle                         # optional dependency
        kaggle.api.authenticate()
        # This dataset ships a single ~800 MB archive ➜ unzip=True extracts it.
        ds_id = "truthisneverlinear/dentex-challenge-2023"
        kaggle.api.dataset_download_files(ds_id, path=tmp, unzip=True)
    except Exception as e:                    # pragma: no cover
        log.warning("Kaggle download failed: %s", e)
        return False

    # A few nested *.zip files (train.zip / val.zip / test.zip) remain inside
    # ─── extract them so we finally end up with <tmp>/images /labels
    for z in tmp.rglob("*.zip"):
        try:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(z.parent)
            z.unlink()                        # save disk
        except zipfile.BadZipFile:
            pass
    return True


def _download_hf(tmp: Path) -> bool:
    """
    HuggingFace fallback mirror (public).  This avoids requiring Kaggle creds
    in CI but is optional.  The repo is gated → unauthenticated pull may fail.
    """
    try:
        from huggingface_hub import snapshot_download         # optional
        snapshot_download(
            repo_id="truthisneverlinear/dentex-challenge-2023",
            repo_type="dataset",
            local_dir=tmp,
            local_dir_use_symlinks=False,
        )
        return True
    except Exception as e:                                    # pragma: no cover
        log.warning("HF download failed: %s", e)
        return False


# ------------------------------------------------------------------ #
def ensure_dentex() -> Path:
    """Return `configs/dentex.yaml` – downloading & splitting if needed."""
    yaml_out = CFG_DIR / "dentex.yaml"
    if yaml_out.exists():
        return yaml_out

    tmp = DATA_DIR / "_dentex_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    # ─── try network mirrors first ───────────────────────────────────────── #
    if not (_download_kaggle(tmp) or _download_hf(tmp)):
        log.warning("Falling‑back to a tiny *demo* Dentex dataset (5 imgs).")

        from PIL import Image, ImageDraw  # pillow is a hard dep via ultralytics
        (tmp / "images").mkdir(parents=True, exist_ok=True)
        (tmp / "labels").mkdir(exist_ok=True)

        for i in range(5):
            img = Image.new("RGB", (640, 480), (40, 40, 40))
            drw = ImageDraw.Draw(img)
            drw.rectangle([190, 180, 450, 300], outline="white", width=3)
            img.save(tmp / "images" / f"demo_{i:03d}.jpg")
            (tmp / "labels" / f"demo_{i:03d}.txt").write_text(
                "0 0.50 0.50 0.40 0.25\n"
            )
        (tmp / "labels" / "classes.txt").write_text("tooth\n")

    # ---------------------------------------------------------------------- #
    # The official challenge already provides YOLO txt labels organised as:
    #     images/{train,val,test}/*.jpg
    #     labels/{train,val,test}/*.txt
    # so we simply *move* those folders into our canonical RAW_DIR.
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for kind in ("images", "labels"):
        src = tmp / kind
        dest = RAW_DIR / kind
        if dest.exists():
            shutil.rmtree(dest)
        if src.exists():
            shutil.move(src.as_posix(), dest.as_posix())

    shutil.rmtree(tmp)

    # ---------------------------------------------------------------------- #
    # quick single‑pass 80/10/10 split  →  data/interim/dentex
    from .manager import _split_into_interim
    interim_root = _split_into_interim(RAW_DIR, "dentex")

    # Build list of class names (present in official labels file)
    cls_file = RAW_DIR / "labels" / "classes.txt"
    if cls_file.exists():
        names = cls_file.read_text().splitlines()
    else:                                       # fallback – single generic class
        names = ["tooth"]

    yaml_out.write_text(
        yaml.dump(
            dict(
                path=str(interim_root),
                train="images/train",
                val="images/val",
                test="images/test",
                nc=len(names),
                names=names,
            ),
            sort_keys=False,
        )
    )
    return yaml_out
