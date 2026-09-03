import hashlib

import numpy as np

from funkeras.models.yolo3 import YoloBody
from funkeras.utils import read_lines
from funmodel2.database import set_weight_path

set_weight_path("/Users/liangtaoniu/workspace/MyDiary/src/tianchi/live/data/weights")
# import tensorflow as tf
# tf.config.experimental_run_functions_eagerly(True)

classes = read_lines("coco.names")


def get_md5(weight):
    m = hashlib.md5()
    m.update(weight)
    return m.hexdigest()


def get_anchors():
    anchors = "10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326"
    # anchors = '1.25,1.625, 2.0,3.75, 4.125,2.875, 1.875,3.8125, 3.875,2.8125, 3.6875,7.4375, 3.625,2.8125, 4.875,6.1875, 11.65625,10.1875'
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


anchors = get_anchors()

yolo_body1 = YoloBody(anchors=anchors, num_classes=len(classes))
yolo_body2 = YoloBody(anchors=anchors, num_classes=len(classes))
yolo_body1.load_weights("/Users/liangtaoniu/workspace/MyDiary/tmp/models/yolo/configs/yolov3.h5", freeze_body=3)

# save_layers(yolo_body1.yolo_model.layers, model_name='yolov3', filename='yolov3.weight')
yolo_body2.load_layer_weights()
# load_layers(yolo_body2.yolo_model.layers, model_name='yolov3')


for i, layer1 in enumerate(yolo_body1.yolo_model.layers):
    layer2 = yolo_body2.yolo_model.layers[i]

    weight1 = layer1.weights
    weight2 = layer2.weights
    if i in (0, 10, 50, 100, 150, 200, 240):
        print(i)
