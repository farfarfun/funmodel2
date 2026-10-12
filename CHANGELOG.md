# 更新日志

## [Unreleased]

## [0.1.1]

### 新增

- 增加权重数据库和 YOLO 辅助 API 的正常路径测试。

### 修复

- `yolo` extra 的依赖约束保持由 `funkeras` 管理；项目的 Python 最低版本为 `>=3.10`。
- 移除 import 阶段的本机路径和模型加载副作用，改为显式调用。
- 使用参数化 SQL、farlog 和具体异常处理，避免数据损坏与诊断信息丢失。
- `database/core.py` 的 4 处权重加载失败日志误用 stdlib logging 的 `%s` 占位符，
  farlog（loguru）不支持该语法，参数被静默丢弃；统一改为 `{}` 占位符。
- `load_layers` 用 md5 字符串给数组列表 `data` 当下标（`data[md5]`），恒抛
  `TypeError` 并被静默吞掉记录为警告，导致权重恢复从未真正生效；改为直接传入
  `data`（主动排查发现，与 todo-list#864 finding 3 一致）。

### 变更

- **破坏性变更：** 导入包和分发名称从 `notemodel` 改为 `funmodel2`。
  迁移时卸载 `notemodel`、安装 `funmodel2`，并将 `import notemodel` 替换为
  `import funmodel2`，将 `notemodel.models` 替换为 `funmodel2.models`。

### 废弃

- 删除旧的 setup.py/twine 发布脚本及自动提交、强制推送脚本。
