# funmodel2

这是一个围绕 Keras YOLOv3（依赖另一个包 [funkeras](https://github.com/farfarfun/funkeras) 的 `YoloBody`）做目标检测实验的一次性脚本集合，看起来是当年天池（Tianchi）比赛相关的模型训练/调参代码，**目前未维护、不是通用库**：

- `funmodel2/models/yolo.py`：加载 YOLOv3 权重、逐层比较两个模型权重的一次性脚本。模型只会在显式调用 `load_yolo_models` 时加载，不会在 import 阶段访问本机路径。
- `funmodel2/database/core.py`：一个基于 SQLite 的模型层权重缓存工具 `WeightDB`，按每层权重的 MD5 对权重去重存储，配合 `save_layers` / `load_layers` 在不同模型之间复用相同的权重块（例如迁移/微调时跳过重复层）。
- `funmodel2/util/core.py`：`get_file_md5`，计算权重字节内容的 MD5，供上面的去重逻辑使用。

## 安装

旧包 `notemodel` 已更名为 `funmodel2`。迁移时卸载旧包并安装新包：

```bash
uv pip uninstall notemodel
uv add funmodel2
```

将代码中的导入统一替换：`import notemodel` 改为 `import funmodel2`，
`notemodel.models` 改为 `funmodel2.models`。旧包转发发布及弃用提示由维护者另行处理。

使用 uv 安装依赖并构建：

```bash
uv sync
uv run python -c "from funmodel2.models.yolo import get_anchors; print(get_anchors().shape)"
```

需要实际加载 YOLO 模型时，再安装可选依赖：`uv sync --extra yolo`。

## 用法示例

```python
from funmodel2.database import WeightDB, save_layers, load_layers, set_weight_path

set_weight_path("/path/to/weights")  # 权重文件的存放目录

db = WeightDB()  # 默认在包目录下建 layer_weight.db
db.insert_if_not_exist(model="yolov3", _class="Conv2D", name="conv1", md5="...", filename="yolov3.weight")
```

`save_layers(layers, model_name, filename)` / `load_layers(layers, model_name, md5_list)` 用于把 Keras 模型各层的权重按 MD5 存进 / 取出 `WeightDB`，从而在多个模型间共享相同的权重块，避免重复保存。

`funmodel2/models/yolo.py` 中的 YOLOv3 加载脚本仍是历史实验记录；调用 `load_yolo_models` 时需要提供实际权重路径和类别文件。

---

## 关于 farfarfun

[farfarfun](https://github.com/farfarfun) 是一个专注于实用工具库的开源组织，
涵盖云存储、数据处理、AI、多媒体与开发工具链等方向。

- 组织主页：<https://github.com/farfarfun>
- PyPI：<https://pypi.org/user/niuliangtao/>
- 联系：farfarfun@qq.com

本项目基于 [MIT](LICENSE) 协议开源。
