#!/usr/bin/env python3
"""
One public entry‑point for everything
$ yrcnn       train dentex --model yolo  --epochs 10
$ yrcnn       train dentex --model rcnn  --epochs 10
$ yrcnn       viz   coco   --n 8
"""
from __future__ import annotations
import subprocess, sys, argparse, pathlib
from typing import List

PKG_ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable


def _run(cmd: List[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("yrcnn")      # exposed as console‑script
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- train ---------------------------------------------------- #
    sp = sub.add_parser("train")
    sp.add_argument("dataset", choices=["coco", "dentex", "local"])
    sp.add_argument("--model", default="yolo")          # yolo | rcnn | any build() id
    sp.add_argument("--epochs", type=int, default=5)
    sp.add_argument("--batch",  type=int, default=8)
    sp.add_argument("--device", default="cpu")
    sp.add_argument("--dry-run", action="store_true")

    # ---- viz  ----------------------------------------------------- #
    sp = sub.add_parser("viz")
    sp.add_argument("dataset")
    sp.add_argument("--n", type=int, default=5)
    sp.add_argument("--device", default="cpu")

    # ---- infer ---------------------------------------------------- #
    sp = sub.add_parser("infer")
    sp.add_argument("weights")
    sp.add_argument("--source", required=True)
    sp.add_argument("--device", default="cpu")

    return p


def main(argv: List[str] | None = None) -> None:
    a = _parser().parse_args(argv)

    if a.cmd == "train":
        entry = "yrcnn.training.train_yolo" if a.model.lower().startswith("yolo") else \
                "yrcnn.training.train_rcnn"
        _run([PY, "-m", entry,
              "--data",    a.dataset,
              "--model",   a.model,
              "--epochs",  str(a.epochs),
              "--batch",   str(a.batch),
              "--device",  a.device,
              *(["--dry-run"] if a.dry_run else [])])

    elif a.cmd == "viz":
        _run([PY, "-m", "yrcnn.utils.viz",
              "--data",  a.dataset,
              "--n",     str(a.n),
              "--device", a.device])

    elif a.cmd == "infer":
        _run([PY, "-m", "yrcnn.training.inference",
              "--weights", a.weights,
              "--source",  a.source,
              "--device",  a.device])


if __name__ == "__main__":          # pragma: no cover
    main()
