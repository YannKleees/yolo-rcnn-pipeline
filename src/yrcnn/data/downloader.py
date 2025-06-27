"""
Download *Dentex* automatically from Kaggle or HuggingFace.

Public API
----------
>>> from yrcnn.data.downloader import ensure_dentex
>>> yaml_path = ensure_dentex()
"""
from __future__ import annotations
import logging, zipfile, shutil, hashlib, urllib.request
from pathlib import Path
from typing import Optional

import yaml, tqdm                      # hard deps (tiny)
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
    try:
        import kaggle      # optional
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files("ibrahimhamamci/dentex",
                                          path=tmp, unzip=True)
        return True
    except Exception as e:
        log.warning("Kaggle download failed: %s", e)
        return False


def _download_hf(tmp: Path) -> bool:
    try:
        from huggingface_hub import snapshot_download         # optional

        snapshot_download(
            repo_id="Ibrahim-Hamamci/DENTEX",
            repo_type="dataset",
            local_dir=tmp,
            local_dir_use_symlinks=False,
        )
        return True
    except Exception as e:
        log.warning("HF download failed: %s", e)
        return False


# ------------------------------------------------------------------ #
def ensure_dentex() -> Path:
    """Return `configs/dentex.yaml` (download & split if absent)."""
    yaml_out = CFG_DIR / "dentex.yaml"
    if yaml_out.exists():
        return yaml_out

    tmp = DATA_DIR / "_dentex_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    if not (_download_kaggle(tmp) or _download_hf(tmp)):
        raise RuntimeError("Could not download Dentex automatically.")

    # Move images / labels into canonical RAW_DIR
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for kind in ("images", "labels"):
        dest = RAW_DIR / kind
        if dest.exists():
            shutil.rmtree(dest)
        (tmp / kind).rename(dest)

    shutil.rmtree(tmp)

    # ---------------------------------------------------------------- #
    # quick single‑pass split (80/10/10) →  data/interim/dentex
    from .manager import _split_into_interim
    interim_root = _split_into_interim(RAW_DIR, "dentex")

    # write YAML
    names = (RAW_DIR/"labels/classes.txt").read_text().splitlines()
    yaml_out.write_text(yaml.dump(dict(
        path=str(interim_root),
        train="images/train",
        val="images/val",
        test="images/test",
        nc=len(names),
        names=names,
    ), sort_keys=False))
    return yaml_out
