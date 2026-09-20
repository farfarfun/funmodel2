import os

import numpy as np

from farlog import getLogger
from funkeras.models.yolo3 import YoloBody
from funkeras.utils import read_lines
from funmodel2.database import set_weight_path

logger = getLogger(__name__)

_COCO_NAMES = os.path.join(os.path.dirname(__file__), "coco.names")


def get_anchors() -> np.ndarray:
    """返回 YOLOv3 默认的 9 组 anchor 尺寸。"""
    anchors = "10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326"
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def compare_yolo_weights(weight_dir: str, model_weights_path: str) -> None:
    """加载一份 YOLOv3 权重文件，分别构建两个模型实例（一个直接加载权重、一个走 `WeightDB` 缓存），
    逐层比较权重是否一致——历史调试脚本，用于验证权重缓存/复用逻辑是否正确。

    `weight_dir` 是 `WeightDB` 缓存权重的存放目录，`model_weights_path` 是待加载的 `.h5` 权重文件路径。
    """
    set_weight_path(weight_dir)
    classes = read_lines(_COCO_NAMES)
    anchors = get_anchors()

    yolo_body1 = YoloBody(anchors=anchors, num_classes=len(classes))
    yolo_body2 = YoloBody(anchors=anchors, num_classes=len(classes))
    yolo_body1.load_weights(model_weights_path, freeze_body=3)

    # save_layers(yolo_body1.yolo_model.layers, model_name='yolov3', filename='yolov3.weight')
    yolo_body2.load_layer_weights()
    # load_layers(yolo_body2.yolo_model.layers, model_name='yolov3')

    for i, layer1 in enumerate(yolo_body1.yolo_model.layers):
        layer2 = yolo_body2.yolo_model.layers[i]

        weight1 = layer1.weights
        weight2 = layer2.weights
        if i in (0, 10, 50, 100, 150, 200, 240):
            logger.info(i)


if __name__ == "__main__":
    compare_yolo_weights(
        weight_dir=os.environ.get("FUNMODEL2_WEIGHT_DIR", "./weights"),
        model_weights_path=os.environ.get("FUNMODEL2_MODEL_WEIGHTS", "./configs/yolov3.h5"),
    )
