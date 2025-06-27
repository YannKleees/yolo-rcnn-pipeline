"""
prepare_dataset()  – central dispatcher for 'coco' / 'dentex' / 'local'
"""
from __future__ import annotations
import random, math, shutil, logging, urllib.request, zipfile, hashlib
from pathlib import Path
from functools import lru_cache

import yaml, tqdm

log = logging.getLogger(__name__)
DATA_DIR = Path("data")
CFG_DIR  = Path("configs")
SPLIT_R  = {"train": .8, "val": .1, "test": .1}


def _split(imgs: list[Path]) -> dict[str, list[Path]]:
    random.shuffle(imgs)
    n = len(imgs)
    tr = max(1, math.floor(n * SPLIT_R["train"]))
    va = max(1, math.floor(n * SPLIT_R["val"]))
    te = max(1, n - tr - va)
    return {"train": imgs[:tr], "val": imgs[tr:tr+va], "test": imgs[tr+va:]}


# ------------------------------------------------------------------ #
def _split_into_interim(raw: Path, tag: str) -> Path:
    """Create data/interim/<tag>/{images,labels}/{train|val|test}/*"""
    interim_root = DATA_DIR / "interim" / tag
    for subset, files in _split(sorted((raw / "images").glob("*.[jp][pn]g"))).items():
        for p in files:
            for kind in ("images", "labels"):
                dst = interim_root / kind / subset
                dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(p,  interim_root/"images"/subset/p.name)
            shutil.copy(raw/"labels"/f"{p.stem}.txt",
                        interim_root/"labels"/subset/f"{p.stem}.txt")
    return interim_root


# ------------------------------------------------------------------ #
@lru_cache(maxsize=None)
def prepare_dataset(name: str) -> Path:
    """
    Return a YAML file suitable for Ultralytics.

    • coco  → download COCO‑128, split train=val (for speed)
    • dentex→ ensure_dentex() from downloader
    • local → user‑supplied images/,labels/ beside repo
    """
    name = name.lower()
    if name == "coco":
        from .downloader import DATA_DIR                                    # reuse const
        yaml_out = CFG_DIR / "coco.yaml"
        if yaml_out.exists():
            return yaml_out
        url  = "https://ultralytics.com/assets/coco128.zip"
        zipp = DATA_DIR / "coco128.zip"
        if not zipp.exists():
            log.info("↓ coco128 …")
            urllib.request.urlretrieve(url, zipp)

        root = DATA_DIR / "coco128"
        if not (root/"images").exists():
            zipfile.ZipFile(zipp).extractall(DATA_DIR)

        names = yaml.safe_load((root/"dataset.yaml").read_text())["names"]
        yaml_out.write_text(yaml.dump(dict(
            path=str(root),
            train="images/train2017",
            val="images/train2017",
            test="",
            names={i:n for i, n in enumerate(names)},
        ), sort_keys=False))
        return yaml_out

    if name == "dentex":
        from .downloader import ensure_dentex
        return ensure_dentex()

    if name == "local":
        root = Path("images")  # expects images/, labels/ beside cwd
        assert root.exists(), "Place images/ and labels/ beside repository"
        raw = Path("data/local"); raw.mkdir(parents=True, exist_ok=True)
        if not (raw/"images").exists():
            shutil.copytree(root,         raw/"images")
            shutil.copytree(Path("labels"), raw/"labels")
        interim = _split_into_interim(raw, "local")
        yaml_out = CFG_DIR / "local.yaml"
        cls = (raw/"labels/classes.txt").read_text().splitlines()
        yaml_out.write_text(yaml.dump(dict(
            path=str(interim),
            train="images/train",
            val="images/val",
            test="images/test",
            names=cls,
        ), sort_keys=False))
        return yaml_out

    raise ValueError(name)


# re‑export for downloader.py
__all__ = ["prepare_dataset", "_split_into_interim"]
