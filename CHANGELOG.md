# Changelog

## [Unreleased]

### 变更

- **破坏性变更**：导入名与 PyPI 包名从 `notemodel` 改为 `funmodel2`，与仓库名保持一致（仓库名一直带着结尾的 "2"，旧导入名却没有）。原先 `import notemodel` / `pip install notemodel` 的用法需改为 `import funmodel2` / `pip install funmodel2`。
- 旧的 `notemodel` PyPI 包后续需要发布一个转发到 `funmodel2` 的终版本——**该操作需仓库/PyPI 包拥有者手动完成**，本次未自动发布（同类先例见 farfarfun/todo-list#401）。
- 迁移为 `src/funmodel2/` 布局；`numpy`/`funkeras`/`tqdm` 补上版本下限，新增 `farlog` 依赖并生成 `uv.lock`。

### 修复

- `funmodel2/models/yolo.py` 不再在 import 时硬编码本机路径并自动加载模型权重，逻辑收进 `compare_yolo_weights()`，路径通过参数/环境变量传入。
- `script/build.sh` 改用 `funbuild`，移除遗留的 `setup.py`/`twine` 流程；`script/build.sh`、`script/push.sh` 中自动 `git commit`/`git push -f` 的危险自动化已移除。
- `database/core.py` 中的 `print()` 诊断输出改为 `farlog`，`except Exception` 收窄为具体异常类型。
