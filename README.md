# funmodel2

这是一个围绕 Keras YOLOv3（依赖另一个包 [notekeras](https://github.com/farfarfun/notekeras) 的 `YoloBody`）做目标检测实验的一次性脚本集合，看起来是当年天池（Tianchi）比赛相关的模型训练/调参代码，**目前未维护、不是通用库**：

- `funmodel2/models/yolo.py`：加载 YOLOv3 权重、逐层比较两个模型权重的一次性脚本。里面硬编码了作者本机路径（如 `/Users/liangtaoniu/workspace/MyDiary/...`），一 import 就会直接执行模型加载逻辑，换台机器基本跑不起来；用到的 `notekeras` 也没有写进 `pyproject.toml` 的依赖里，单独 `pip install` 这个包并不能直接跑通这个模块。
- `funmodel2/database/core.py`：一个基于 SQLite 的模型层权重缓存工具 `WeightDB`，按每层权重的 MD5 对权重去重存储，配合 `save_layers` / `load_layers` 在不同模型之间复用相同的权重块（例如迁移/微调时跳过重复层）。
- `funmodel2/util/core.py`：`get_file_md5`，计算权重字节内容的 MD5，供上面的去重逻辑使用。

## 安装

未发布到 PyPI，没有 `pip install` 方式。如需使用，需要克隆仓库并自行处理 `notekeras` 等依赖。

## 用法示例

```python
from funmodel2.database import WeightDB, save_layers, load_layers, set_weight_path

set_weight_path("/path/to/weights")  # 权重文件的存放目录

db = WeightDB()  # 默认在包目录下建 layer_weight.db
db.insert_if_not_exist(model="yolov3", _class="Conv2D", name="conv1", md5="...", filename="yolov3.weight")
```

`save_layers(layers, model_name, filename)` / `load_layers(layers, model_name, md5_list)` 用于把 Keras 模型各层的权重按 MD5 存进 / 取出 `WeightDB`，从而在多个模型间共享相同的权重块，避免重复保存。

`funmodel2/models/yolo.py` 中的 YOLOv3 加载脚本依赖硬编码的本地路径和未声明的 `notekeras` 依赖，仅作为历史实验记录保留，不建议直接复用。
