"""YOLOv3 实验辅助函数，不会在导入时加载模型或访问本机路径。"""

import hashlib
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np

from funmodel2.database import set_weight_path


def get_md5(weight: bytes) -> str:
    """计算权重字节内容的 MD5 摘要。

    参数:
        weight: 要计算摘要的字节内容。

    返回:
        十六进制 MD5 摘要。
    """
    return hashlib.md5(weight).hexdigest()


def get_anchors() -> np.ndarray:
    """返回 YOLOv3 默认锚框数组。

    返回:
        形状为 ``(9, 2)`` 的浮点数锚框数组。
    """
    values = "10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326"
    return np.array([float(value) for value in values.split(",")]).reshape(-1, 2)


def load_yolo_models(weights_path: str | Path, classes_path: str | Path | None = None) -> tuple[Any, Any]:
    """显式加载两个 YOLOv3 模型用于历史权重实验。

    参数:
        weights_path: 第一个模型要加载的权重文件路径。
        classes_path: 每行一个类别名称的文件路径；不传时使用包内的 COCO 类别文件。

    返回:
        已加载完整权重的模型和按层恢复权重的模型。
    """
    from funkeras.models.yolo3 import YoloBody
    from funkeras.utils import read_lines

    weights_path = Path(weights_path)
    classes_file = (
        Path(classes_path)
        if classes_path is not None
        else Path(str(files("funmodel2.models").joinpath("coco.names")))
    )
    classes = read_lines(str(classes_file))
    set_weight_path(str(weights_path.parent))
    anchors = get_anchors()
    first = YoloBody(anchors=anchors, num_classes=len(classes))
    second = YoloBody(anchors=anchors, num_classes=len(classes))
    first.load_weights(str(weights_path), freeze_body=3)
    second.load_layer_weights()
    return first, second
