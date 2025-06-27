"""
Quick visual QA – random images + GT boxes.

Works on Windows & *nx, accepts a benign --device flag (CLI parity).
"""

from __future__ import annotations
import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import yaml

from yrcnn.data.manager import prepare_dataset


# --------------------------------------------------------------------------- #
def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser("Visualise random images with ground‑truth boxes")
    p.add_argument("--data", required=True, help="coco | dentex | local")
    p.add_argument("--n", type=int, default=5, help="#images to display")
    p.add_argument(
        "--device",
        default="cpu",
        help="ignored – kept only so `yrcnn cli` can forward it transparently",
    )
    return p.parse_args()


# --------------------------------------------------------------------------- #
def _label_path_from_image(img_p: Path) -> Path:
    """
    Robust “/images/...jpg → /labels/...txt” conversion that also works
    on Windows (back‑slashes) and with arbitrary sub‑folders.
    """
    parts = list(img_p.parts)
    try:
        i = parts.index("images")
        parts[i] = "labels"
        return Path(*parts).with_suffix(".txt")
    except ValueError:  # unexpected layout → fall back to sibling traversal
        cand = (
            img_p.parent.parent
            / "labels"
            / img_p.parent.name
            / f"{img_p.stem}.txt"
        )
        return cand


def _draw(img_p: Path) -> None:
    img = cv2.imread(str(img_p))
    h, w = img.shape[:2]

    lbl = _label_path_from_image(img_p)
    if not lbl.exists():  # silently skip if label is missing
        return

    for row in lbl.read_text().splitlines():
        cls, xc, yc, bw, bh = map(float, row.split())
        x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
        x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    plt.figure(figsize=(6, 6))
    plt.title(img_p.name)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")


# --------------------------------------------------------------------------- #
def main() -> None:
    a = _cli()
    yaml_p = prepare_dataset(a.data)
    meta = yaml.safe_load(Path(yaml_p).read_text())
    root = Path(meta["path"]) / "images" / "train"

    imgs = list(root.glob("*.[jp][pn]g"))
    if not imgs:
        print("❗ dataset appears to be empty – nothing to visualise")
        return

    for p in random.sample(imgs, min(a.n, len(imgs))):
        _draw(p)
    plt.show()


if __name__ == "__main__":
    main()
