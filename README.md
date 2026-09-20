# funmodel2

这是一个围绕 Keras YOLOv3（依赖另一个包 [funkeras](https://github.com/farfarfun/funkeras) 的 `YoloBody`）做目标检测实验的一次性脚本集合，看起来是当年天池（Tianchi）比赛相关的模型训练/调参代码，**目前未维护、不是通用库**：

- `funmodel2/models/yolo.py`：加载 YOLOv3 权重、逐层比较两个模型权重的调试脚本，逻辑收在 `compare_yolo_weights(weight_dir, model_weights_path)` 函数里，只有显式调用（或 `python -m funmodel2.models.yolo`）才会执行，不再在 import 时自动跑；权重目录/权重文件路径通过参数（或 `FUNMODEL2_WEIGHT_DIR` / `FUNMODEL2_MODEL_WEIGHTS` 环境变量）传入，不再硬编码作者本机路径。
- `funmodel2/database/core.py`：一个基于 SQLite 的模型层权重缓存工具 `WeightDB`，按每层权重的 MD5 对权重去重存储，配合 `save_layers` / `load_layers` 在不同模型之间复用相同的权重块（例如迁移/微调时跳过重复层）。
- `funmodel2/util/core.py`：`get_file_md5`，计算权重字节内容的 MD5，供上面的去重逻辑使用。

## 安装

尚未发布到 PyPI，如需使用需克隆仓库自行安装：

```bash
git clone https://github.com/farfarfun/funmodel2.git
cd funmodel2
uv sync
```

`funkeras` 已作为依赖声明在 `pyproject.toml` 中，`uv sync` / `pip install -e .` 会自动装上，不需要单独处理。

## 用法示例

```python
from funmodel2.database import WeightDB, save_layers, load_layers, set_weight_path

set_weight_path("/path/to/weights")  # 权重文件的存放目录

db = WeightDB()  # 默认在包目录下建 layer_weight.db
db.insert_if_not_exist(model="yolov3", _class="Conv2D", name="conv1", md5="...", filename="yolov3.weight")
```

`save_layers(layers, model_name, filename)` / `load_layers(layers, model_name, md5_list)` 用于把 Keras 模型各层的权重按 MD5 存进 / 取出 `WeightDB`，从而在多个模型间共享相同的权重块，避免重复保存。

`funmodel2/models/yolo.py` 中的 `compare_yolo_weights` 是历史天池比赛实验遗留的调试脚本，仅作为历史记录保留，不建议直接复用。

## 关于 farfarfun

[farfarfun](https://github.com/farfarfun) 是一个专注于实用工具库的开源组织，
涵盖云存储、数据处理、AI、多媒体与开发工具链等方向。

- 🏠 组织主页：<https://github.com/farfarfun>
- 📦 PyPI：<https://pypi.org/user/niuliangtao/>
- 📧 联系：farfarfun@qq.com

本项目基于 [MIT](LICENSE) 协议开源。
