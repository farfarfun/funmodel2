"""SQLite 缓存模型层权重元数据。"""

import hashlib
import os
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from farlog import getLogger

logger = getLogger("funmodel2")
weight_path: str | None = None


def _cache_directory() -> Path:
    """返回运行时缓存目录，避免向安装包目录写入文件。"""
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "funmodel2"


def set_weight_path(path: str | os.PathLike[str]) -> None:
    """设置权重文件目录。

    参数:
        path: 后续读写权重文件时使用的目录。

    返回:
        无。
    """
    global weight_path
    weight_path = os.fspath(path)


def get_file_path(filename: str) -> str:
    """根据文件名返回权重文件路径。

    参数:
        filename: 权重文件名。

    返回:
        配置目录下的权重文件路径。
    """
    file_name = Path(filename)
    if not filename or file_name.is_absolute() or file_name.name != filename or filename in {".", ".."}:
        raise ValueError(f"filename must be a non-empty basename: {filename!r}")

    directory = Path(weight_path) if weight_path is not None else _cache_directory() / "weights"
    directory = directory.resolve()
    path = (directory / file_name).resolve()
    try:
        path.relative_to(directory)
    except ValueError as error:
        raise ValueError(f"filename escapes the configured weight directory: {filename!r}") from error
    return os.fspath(path)


def save_weight(data: Any, file_path: str) -> None:
    """将权重数据序列化到指定文件。

    参数:
        data: 要序列化的权重数据。
        file_path: 目标文件路径。

    返回:
        无。
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as output:
        arrays = {
            f"{md5}:{index}": np.asarray(value)
            for md5, values in data.items()
            for index, value in enumerate(values)
        }
        np.savez_compressed(output, **arrays)


class WeightDB:
    """保存模型层权重元数据的 SQLite 数据库。

    参数:
        db_path: SQLite 数据库路径；不传时使用用户缓存目录。

    返回:
        可用于读写权重元数据的数据库实例。
    """

    def __init__(self, db_path: str | os.PathLike[str] | None = None) -> None:
        """创建数据库连接并初始化元数据表。

        参数:
            db_path: SQLite 数据库路径；不传时使用用户缓存目录。

        返回:
            无。
        """
        self.db_path = os.fspath(db_path or _cache_directory() / "layer_weight.db")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.table_name = "layer_weight"
        self.create()

    def create(self) -> None:
        """创建尚不存在的权重元数据表。

        返回:
            无。
        """
        self.conn.execute(f"""create table if not exists {self.table_name} (
            id integer primary key autoincrement,
            model text not null, class text not null, name text not null default '',
            md5 text not null default '', filename text not null default '')""")
        self.conn.commit()

    def insert(self, model: str = "", _class: str = "", name: str = "", md5: str = "", filename: str = "") -> None:
        """插入一条权重元数据记录。

        参数:
            model: 模型名称。
            _class: 层类型名称。
            name: 层名称。
            md5: 权重内容的 MD5 摘要。
            filename: 权重缓存文件名。

        返回:
            无。
        """
        if filename:
            get_file_path(filename)
        self.conn.execute(f"insert into {self.table_name} (model,class,name,md5,filename) values (?,?,?,?,?)",
                          (model, _class, name, md5, filename))
        self.conn.commit()

    def insert_if_not_exist(self, model: str = "", _class: str = "", name: str = "", md5: str = "", filename: str = "") -> bool:
        """仅在相同记录不存在时插入。

        参数:
            model: 模型名称。
            _class: 层类型名称。
            name: 层名称。
            md5: 权重内容的 MD5 摘要。
            filename: 权重缓存文件名。

        返回:
            插入新记录时为 ``True``，记录已存在时为 ``False``。
        """
        exists = self.conn.execute(
            f"select 1 from {self.table_name} where model=? and class=? and name=? and md5=? and filename=? limit 1",
            (model, _class, name, md5, filename)).fetchone()
        if exists:
            return False
        self.insert(model, _class, name, md5, filename)
        return True

    def select_by_name(self, model: str = "", _class: str = "", name: str = "") -> list[tuple[Any, ...]]:
        """按模型、层类型和层名称查询记录。

        参数:
            model: 模型名称。
            _class: 层类型名称。
            name: 层名称。

        返回:
            至多十条匹配的数据库记录。
        """
        return self.conn.execute(f"select * from {self.table_name} where model=? and class=? and name=? limit 10",
                                 (model, _class, name)).fetchall()

    def select_by_md5(self, model: str = "", md5: str = "") -> list[tuple[Any, ...]]:
        """按模型和权重 MD5 查询记录。

        参数:
            model: 模型名称。
            md5: 权重内容的 MD5 摘要。

        返回:
            至多十条匹配的数据库记录。
        """
        return self.conn.execute(f"select * from {self.table_name} where model=? and md5=? limit 10",
                                 (model, md5)).fetchall()

    def count(self) -> int:
        """返回数据库中的记录数。

        返回:
            当前元数据表的记录总数。
        """
        return self.conn.execute(f"select count(*) from {self.table_name}").fetchone()[0]

    def select(self, size: int = 50) -> list[tuple[str, str]]:
        """返回指定数量的模型和层类型。

        参数:
            size: 最多返回的记录数，必须为非负整数。

        返回:
            模型名称和层类型组成的元组列表。

        异常:
            ValueError: ``size`` 为负数。
        """
        if size < 0:
            raise ValueError("size must be non-negative")
        rows = self.conn.execute(f"select model, class from {self.table_name} limit ?", (size,)).fetchall()
        return [(row[0], row[1]) for row in rows]

    def close(self) -> None:
        """提交未提交的事务并关闭数据库连接。

        返回:
            无。
        """
        self.conn.commit()
        self.conn.close()


def _layer_md5(layer: Any) -> tuple[str, list[Any]]:
    values = [weight.numpy() for weight in layer.weights]
    digest = hashlib.md5()
    for value in values:
        digest.update(value)
    return digest.hexdigest(), values


def save_layers(layers: Iterable[Any], model_name: str, filename: str) -> None:
    """保存模型各层权重并记录去重元数据。

    参数:
        layers: 包含 Keras 风格 ``weights`` 属性的模型层。
        model_name: 用于检索权重的模型名称。
        filename: 保存权重字典的缓存文件名。

    返回:
        无。
    """
    database = WeightDB()
    data: dict[str, list[Any]] = {}
    try:
        for layer in layers:
            if not layer.weights:
                continue
            md5, values = _layer_md5(layer)
            database.insert_if_not_exist(model_name, type(layer).__name__, layer.name, md5, filename)
            data[md5] = values
        save_weight(data, get_file_path(filename))
    finally:
        database.close()


def load_layers(layers: Iterable[Any], model_name: str, md5_list: list[str]) -> None:
    """按 MD5 从缓存恢复模型层权重。

    缺失记录、不可读取的缓存和无法设置的权重会被跳过并记录警告。

    参数:
        layers: 要恢复权重的 Keras 风格模型层。
        model_name: 查询元数据时使用的模型名称。
        md5_list: 按有权重层顺序排列的 MD5 摘要。

    返回:
        无。
    """
    database = WeightDB()
    try:
        index = 0
        for layer in layers:
            if not layer.weights or index >= len(md5_list):
                continue
            records = database.select_by_md5(model_name, md5_list[index])
            index += 1
            if not records:
                continue
            _, _, _, _, md5, filename = records[0]
            try:
                file_path = get_file_path(filename)
            except ValueError as error:
                logger.warning("忽略不安全的权重文件名 {}: {}", filename, error)
                continue
            if not os.path.exists(file_path):
                logger.warning("权重文件不存在: {}", file_path)
                continue
            try:
                with np.load(file_path, allow_pickle=False) as source:
                    prefix = f"{md5}:"
                    keys = sorted(
                        (key for key in source.files if key.startswith(prefix)),
                        key=lambda key: int(key.removeprefix(prefix)),
                    )
                    data = [source[key] for key in keys]
            except (OSError, ValueError, EOFError) as error:
                logger.warning("读取权重缓存失败 {}: {}", file_path, error)
                continue
            if not data:
                logger.warning("权重摘要不存在: {}", md5)
                continue
            try:
                layer.set_weights(data[md5])
            except (TypeError, ValueError) as error:
                logger.warning("设置层权重失败 {}: {}", getattr(layer, "name", "unknown"), error)
    finally:
        database.close()
