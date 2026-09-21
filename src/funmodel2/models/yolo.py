"""YOLOv3 实验辅助函数，不会在导入时加载模型或访问本机路径。"""

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from funmodel2.database import set_weight_path


def get_md5(weight: bytes) -> str:
    """计算权重字节内容的 MD5 摘要。"""
    return hashlib.md5(weight).hexdigest()


def get_anchors() -> np.ndarray:
    """返回 YOLOv3 默认锚框数组。"""
    values = "10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326"
    return np.array([float(value) for value in values.split(",")]).reshape(-1, 2)


def load_yolo_models(weights_path: str | Path, classes_path: str | Path = "coco.names") -> tuple[Any, Any]:
    """显式加载两个 YOLOv3 模型用于历史权重实验。"""
    from funkeras.models.yolo3 import YoloBody
    from funkeras.utils import read_lines

    weights_path = Path(weights_path)
    classes = read_lines(str(classes_path))
    set_weight_path(str(weights_path.parent))
    anchors = get_anchors()
    first = YoloBody(anchors=anchors, num_classes=len(classes))
    second = YoloBody(anchors=anchors, num_classes=len(classes))
    first.load_weights(str(weights_path), freeze_body=3)
    second.load_layer_weights()
    return first, second
