"""
Quick visual QA – random images + GT boxes.
"""
from __future__ import annotations
import argparse, random, cv2, matplotlib.pyplot as plt
from pathlib import Path
from yrcnn.data.manager import prepare_dataset

def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--n",   type=int, default=5)
    return p.parse_args()

def _draw(img_p: Path):
    img = cv2.imread(str(img_p)); h, w = img.shape[:2]
    lbl = Path(str(img_p).replace("/images/","/labels/")).with_suffix(".txt")
    for row in lbl.read_text().splitlines():
        cls, xc, yc, bw, bh = map(float,row.split())
        x1,y1 = int((xc-bw/2)*w), int((yc-bh/2)*h)
        x2,y2 = int((xc+bw/2)*w), int((yc+bh/2)*h)
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
    plt.figure(figsize=(6,6)); plt.title(img_p.name)
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)); plt.axis("off")

def main():
    a = _cli()
    yaml_p = prepare_dataset(a.data)
    root = Path(str(yaml_p).replace(".yaml",""))/"images"/"train"
    imgs = list(root.glob("*.[pj][pn]g"))
    for p in random.sample(imgs, min(a.n,len(imgs))):
        _draw(p)
    plt.show()

if __name__ == "__main__":
    main()
