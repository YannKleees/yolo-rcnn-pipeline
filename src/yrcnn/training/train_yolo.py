"""
Thin wrapper around ultralytics.YOLO `.train()`.
Handles wandb & dataset prep.
"""
from __future__ import annotations
import argparse, os, pathlib
from ultralytics import YOLO
from yrcnn.data.manager import prepare_dataset

def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser("train YOLO")
    p.add_argument("--data",   required=True)
    p.add_argument("--model",  default="yolo11n.pt")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch",  type=int, default=16)
    p.add_argument("--imgsz",  type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()

def main() -> None:
    a = _cli()
    yaml_path = prepare_dataset(a.data)

    if a.dry_run:
        print("✓ dry‑run only – dataset prepared:", yaml_path)
        return

    model = YOLO(a.model)
    enable_wandb = bool(os.getenv("WANDB_API_KEY"))
    project = f"{a.data}-yolo11"
    model.train(
        data=str(yaml_path),
        epochs=a.epochs,
        batch=a.batch,
        imgsz=a.imgsz,
        device=a.device,
        project=f"runs/{a.data}",
        name=f"img{a.imgsz}_bs{a.batch}",
        wandb=dict(project=project) if enable_wandb else False,
    )

if __name__ == "__main__":  # pragma: no cover
    main()
