"""SQLite 缓存模型层权重元数据。"""

import hashlib
import os
import pickle
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from farlog import getLogger

logger = getLogger("funmodel2")
weight_path: str | None = None


def set_weight_path(path: str | os.PathLike[str]) -> None:
    """设置权重文件目录。"""
    global weight_path
    weight_path = os.fspath(path)


def get_file_path(filename: str) -> str:
    """根据文件名返回权重文件路径。"""
    directory = weight_path or os.path.join(os.path.dirname(__file__), "temp")
    return os.path.join(directory, filename)


def save_weight(data: Any, file_path: str) -> None:
    """将权重数据序列化到指定文件。"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as output:
        pickle.dump(data, output)


class WeightDB:
    """保存模型层权重元数据的 SQLite 数据库。"""

    def __init__(self, db_path: str | os.PathLike[str] | None = None) -> None:
        """创建数据库连接。"""
        self.db_path = os.fspath(db_path or os.path.join(os.path.dirname(__file__), "layer_weight.db"))
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.table_name = "layer_weight"
        self.create()

    def create(self) -> None:
        """创建权重元数据表。"""
        self.conn.execute(f"""create table if not exists {self.table_name} (
            id integer primary key autoincrement,
            model text not null, class text not null, name text not null default '',
            md5 text not null default '', filename text not null default '')""")
        self.conn.commit()

    def insert(self, model: str = "", _class: str = "", name: str = "", md5: str = "", filename: str = "") -> None:
        """插入一条权重元数据记录。"""
        self.conn.execute(f"insert into {self.table_name} (model,class,name,md5,filename) values (?,?,?,?,?)",
                          (model, _class, name, md5, filename))
        self.conn.commit()

    def insert_if_not_exist(self, model: str = "", _class: str = "", name: str = "", md5: str = "", filename: str = "") -> bool:
        """仅在相同记录不存在时插入，并返回是否插入。"""
        exists = self.conn.execute(
            f"select 1 from {self.table_name} where model=? and class=? and name=? and md5=? and filename=? limit 1",
            (model, _class, name, md5, filename)).fetchone()
        if exists:
            return False
        self.insert(model, _class, name, md5, filename)
        return True

    def select_by_name(self, model: str = "", _class: str = "", name: str = "") -> list[tuple[Any, ...]]:
        """按模型、层类型和层名称查询记录。"""
        return self.conn.execute(f"select * from {self.table_name} where model=? and class=? and name=? limit 10",
                                 (model, _class, name)).fetchall()

    def select_by_md5(self, model: str = "", md5: str = "") -> list[tuple[Any, ...]]:
        """按模型和权重 MD5 查询记录。"""
        return self.conn.execute(f"select * from {self.table_name} where model=? and md5=? limit 10",
                                 (model, md5)).fetchall()

    def count(self) -> int:
        """返回数据库中的记录数。"""
        return self.conn.execute(f"select count(*) from {self.table_name}").fetchone()[0]

    def select(self, size: int = 50) -> list[tuple[str, str]]:
        """返回指定数量的模型和层名称。"""
        if size < 0:
            raise ValueError("size must be non-negative")
        rows = self.conn.execute(f"select model, class from {self.table_name} limit ?", (size,)).fetchall()
        return [(row[0], row[1]) for row in rows]

    def close(self) -> None:
        """提交并关闭数据库连接。"""
        self.conn.commit()
        self.conn.close()


def _layer_md5(layer: Any) -> tuple[str, list[Any]]:
    values = [weight.numpy() for weight in layer.weights]
    digest = hashlib.md5()
    for value in values:
        digest.update(value)
    return digest.hexdigest(), values


def save_layers(layers: Iterable[Any], model_name: str, filename: str) -> None:
    """保存模型各层权重并记录去重元数据。"""
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
    """按 MD5 从缓存恢复模型层权重，缺失记录时跳过并记录警告。"""
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
            file_path = get_file_path(filename)
            if not os.path.exists(file_path):
                logger.warning("权重文件不存在: %s", file_path)
                continue
            with open(file_path, "rb") as source:
                data = pickle.load(source)
            if md5 not in data:
                logger.warning("权重摘要不存在: %s", md5)
                continue
            try:
                layer.set_weights(data[md5])
            except (TypeError, ValueError) as error:
                logger.warning("设置层权重失败 %s: %s", getattr(layer, "name", "unknown"), error)
    finally:
        database.close()
