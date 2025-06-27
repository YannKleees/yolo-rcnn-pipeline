"""
Faster‑RCNN ResNet‑50 – nothing else (no Mask‑RCNN to keep deps light).
"""
from __future__ import annotations
import torchvision


def build(num_classes: int):
    net = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_f = net.roi_heads.box_predictor.cls_score.in_features
    net.roi_heads.box_predictor = \
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_f, num_classes)
    return net
