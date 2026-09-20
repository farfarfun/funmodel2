#!/usr/bin/env bash
set -euo pipefail

# 使用组织统一的 funbuild 工具构建/发布，见 SPEC.md §7
uv run funbuild build --multi
