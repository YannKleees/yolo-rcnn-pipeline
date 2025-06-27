"""
prepare_dataset()  – central dispatcher for 'coco' / 'dentex' / 'local'
"""

from __future__ import annotations
import logging, math, random, shutil, urllib.request, zipfile
from functools import lru_cache
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

DATA_DIR = Path("data")
CFG_DIR = Path("configs")
SPLIT_R = {"train": 0.8, "val": 0.1, "test": 0.1}


# --------------------------------------------------------------------------- #
def _split(imgs: list[Path]) -> dict[str, list[Path]]:
    """80 / 10 / 10 deterministic split."""
    random.shuffle(imgs)
    n = len(imgs)
    tr = max(1, math.floor(n * SPLIT_R["train"]))
    va = max(1, math.floor(n * SPLIT_R["val"]))
    te = max(1, n - tr - va)
    return {"train": imgs[:tr], "val": imgs[tr : tr + va], "test": imgs[tr + va :]}


def _split_into_interim(raw: Path, tag: str) -> Path:
    """
    Copy raw YOLO folders into  data/interim/<tag>/…

    End layout:
        images/{train,val,test}/*.jpg
        labels/{train,val,test}/*.txt
    """
    interim_root = DATA_DIR / "interim" / tag
    for subset, files in _split(sorted((raw / "images").glob("*.[jp][pn]g"))).items():
        for p in files:
            for kind in ("images", "labels"):
                dst = interim_root / kind / subset
                dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, interim_root / "images" / subset / p.name)
            shutil.copy(
                raw / "labels" / f"{p.stem}.txt",
                interim_root / "labels" / subset / f"{p.stem}.txt",
            )
    return interim_root


# exposed so downloader can reuse it
__all__ = ["prepare_dataset", "_split_into_interim"]


# --------------------------------------------------------------------------- #
def _yaml_points_to_nonempty_train(yaml_path: Path) -> bool:
    """Small helper: return *True* iff the YAML points to at least one train image."""
    try:
        cfg = yaml.safe_load(yaml_path.read_text())
        train_glob = (
            Path(cfg["path"]) / cfg["train"]
        ).glob("*.[jp][pn]g")  # jpg / jpeg / png
        return any(train_glob)
    except Exception:  # anything → treat as broken
        return False


@lru_cache(maxsize=None)
def prepare_dataset(name: str) -> Path:
    """
    Return a **Ultralytics‑compatible YAML** describing the dataset.

    ─────────────────────────────────────────────────────────────────────
    • coco   → download COCO‑128 if absent (≈ 6 MB) and normalise layout  
    • dentex → delegate to  yrcnn.data.downloader.ensure_dentex()  
    • local  → user‑supplied images/, labels/ beside the repo root
    """
    name = name.lower()

    # ----------------------------------------------------------------- #
    if name == "coco":
        yaml_out = CFG_DIR / "coco.yaml"

        # (re‑)build automatically if YAML is missing *or* its train folder
        # is empty – this solves the “split is empty” assertion.
        if yaml_out.exists() and _yaml_points_to_nonempty_train(yaml_out):
            return yaml_out
        if yaml_out.exists():  # stale → rebuild
            yaml_out.unlink(missing_ok=True)

        url = "https://ultralytics.com/assets/coco128.zip"
        zipp = DATA_DIR / "coco128.zip"
        if not zipp.exists():
            log.info("↓ COCO‑128 …")
            urllib.request.urlretrieve(url, zipp)

        root = DATA_DIR / "coco128"
        if not (root / "images").exists():
            zipfile.ZipFile(zipp).extractall(DATA_DIR)

        # --- normalise folder names ---------------------------------- #
        for sub in ("images", "labels"):
            src = root / sub / "train2017"
            dst = root / sub / "train"
            if src.exists() and (
                not dst.exists() or not any(dst.iterdir())
            ):
                shutil.copytree(src, dst, dirs_exist_ok=True)

        for sub in ("images", "labels"):
            va = root / sub / "val"
            tr = root / sub / "train"
            if tr.exists() and (not va.exists() or not any(va.iterdir())):
                shutil.copytree(tr, va, dirs_exist_ok=True)

        # --- dataset.yaml -------------------------------------------- #
        ds_meta = root / "dataset.yaml"
        if not ds_meta.exists():
            COCO_NAMES = [
                "person","bicycle","car","motorcycle","airplane","bus","train",
                "truck","boat","traffic light","fire hydrant","stop sign",
                "parking meter","bench","bird","cat","dog","horse","sheep","cow",
                "elephant","bear","zebra","giraffe","backpack","umbrella",
                "handbag","tie","suitcase","frisbee","skis","snowboard",
                "sports ball","kite","baseball bat","baseball glove","skateboard",
                "surfboard","tennis racket","bottle","wine glass","cup","fork",
                "knife","spoon","bowl","banana","apple","sandwich","orange",
                "broccoli","carrot","hot dog","pizza","donut","cake","chair",
                "couch","potted plant","bed","dining table","toilet","tv",
                "laptop","mouse","remote","keyboard","cell phone","microwave",
                "oven","toaster","sink","refrigerator","book","clock","vase",
                "scissors","teddy bear","hair drier","toothbrush",
            ]
            ds_meta.write_text(yaml.dump({"names": COCO_NAMES}))

        names = yaml.safe_load(ds_meta.read_text())["names"]
        yaml_out.write_text(
            yaml.dump(
                dict(
                    path=str(root),
                    train="images/train",
                    val="images/val",
                    test="",
                    names={i: n for i, n in enumerate(names)},
                ),
                sort_keys=False,
            )
        )
        return yaml_out

    # ----------------------------------------------------------------- #
    if name == "dentex":
        from .downloader import ensure_dentex

        yaml_p = ensure_dentex()
        # if, for some reason, the fallback demo‑set was already present but
        # empty → force re‑creation
        if not _yaml_points_to_nonempty_train(yaml_p):
            yaml_p.unlink(missing_ok=True)
            yaml_p = ensure_dentex()
        return yaml_p

    # ----------------------------------------------------------------- #
    if name == "local":
        root = Path("images")
        assert root.exists(), "Place images/ and labels/ beside repository root"

        raw = DATA_DIR / "local"
        raw.mkdir(parents=True, exist_ok=True)
        if not (raw / "images").exists():
            shutil.copytree(root, raw / "images")
            shutil.copytree(Path("labels"), raw / "labels")

        interim = _split_into_interim(raw, "local")
        yaml_out = CFG_DIR / "local.yaml"
        cls = (raw / "labels/classes.txt").read_text().splitlines()
        yaml_out.write_text(
            yaml.dump(
                dict(
                    path=str(interim),
                    train="images/train",
                    val="images/val",
                    test="images/test",
                    names=cls,
                ),
                sort_keys=False,
            )
        )
        return yaml_out

    raise ValueError(name)
