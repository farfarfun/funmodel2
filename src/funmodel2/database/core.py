import hashlib
import os
import pickle
import sqlite3
from typing import Any

from farlog import getLogger
from tqdm import tqdm

logger = getLogger(__name__)

weight_path: str | None = None


def set_weight_path(path: str) -> None:
    """设置权重文件的存放目录。"""
    global weight_path
    weight_path = path


def get_file_path(filename: str) -> str:
    """根据 `set_weight_path` 设置的目录拼出权重文件的完整路径，未设置时回退到包内 `temp/` 目录。"""
    file_dir = weight_path or os.path.abspath(os.path.dirname(__file__)) + '/../../temp/'
    return os.path.join(file_dir, filename)


def save_weight(data: Any) -> None:
    """把权重数据序列化到本地文件。"""
    pickle.dump(data, open('file_path', 'wb'))


class WeightDB:
    """基于 SQLite 的模型层权重去重缓存，按每层权重的 MD5 记录存储位置，避免重复保存相同权重块。"""

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            db_path = os.path.abspath(os.path.dirname(__file__)) + '/layer_weight.db'
        self.db_path = db_path

        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.table_name = "layer_weight"
        self.create()

    def execute(self, sql: str) -> sqlite3.Cursor:
        """执行一条 SQL 并提交事务。"""
        self.conn.commit()
        return self.cursor.execute(sql)

    def create(self) -> None:
        """创建权重记录表（若不存在）。"""
        self.execute("""
        create table if not exists {} (
            id                  integer primary key AUTOINCREMENT
           ,model               varchar(200)
           ,class               varchar(200)
           ,name                varchar(200)  DEFAULT ('')
           ,md5                 varchar(200)  DEFAULT ('')
           ,filename            varchar(200)  DEFAULT ('')
           )
        """.format(self.table_name))

    def insert(self, model: str = '', _class: str = '', name: str = '', md5: str = '', filename: str = '') -> sqlite3.Cursor:
        """插入一条权重记录。"""
        name = name.replace("'", '')
        res = self.execute(
            """insert into {} (model,class,name,md5,filename) values ('{}','{}','{}','{}','{}')
            """.format(self.table_name, model, _class, name, md5, filename)
        )
        return res

    def insert_if_not_exist(self, model: str = '', _class: str = '', name: str = '', md5: str = '', filename: str = '') -> bool:
        """若同名同 MD5 的记录不存在则插入，返回是否发生了插入。"""
        name = name.replace("'", '')
        res = self.execute(
            """select * from {} where model='{}' and class='{}' and name='{}' and md5='{}' and filename='{}' limit 10
            """.format(self.table_name, model, _class, name, md5, filename)
        )
        if len(list(res)) == 0:
            self.insert(model, _class, name, md5, filename)
            return True
        return False

    def select_by_name(self, model: str = '', _class: str = '', name: str = '') -> list[tuple]:
        """按模型名/类名/层名查询权重记录。"""
        name = name.replace("'", '')
        res = self.execute(
            """select * from {} where model='{}' and class='{}' and name='{}' limit 10
            """.format(self.table_name, model, _class, name)
        )

        return list(res)

    def select_by_md5(self, model: str = '', md5: str = '') -> list[tuple]:
        """按模型名/MD5 查询权重记录。"""
        res = self.execute(
            """select * from {} where model='{}' and md5='{}' limit 10
            """.format(self.table_name, model, md5)
        )

        return list(res)

    def count(self) -> list:
        res = self.execute("""select count(1) from  {}""".format(self.table_name))

        urls = []
        return urls

    def select(self, size: int = 50) -> list[tuple]:
        """列出最多 `size` 条权重记录的 (class, name) 组。"""
        res = self.execute("""select * from  {} limit {} """.format(self.table_name, size))

        urls = []
        for line in res:
            urls.append((line[1], line[2]))
        return urls

    def close(self) -> None:
        """关闭数据库连接。"""
        self.cursor.close()
        self.conn.close()


def save_layers(layers: list, model_name: str, filename: str) -> None:
    """把 Keras 模型各层的权重按 MD5 去重后写入 `WeightDB` 并序列化保存到磁盘。"""
    layerWeight = WeightDB()
    layerWeight.create()

    data = {}
    for layer in layers:
        if len(layer.weights) == 0:
            continue
        m = hashlib.md5()
        weight_array = []
        for weight in layer.weights:
            m.update(weight.numpy())
            weight_array.append(weight.numpy())
        md5 = m.hexdigest()
        name = layer.name
        _class = type(layer)._keras_api_names[-1]

        insert = layerWeight.insert_if_not_exist(model=model_name, _class=_class, name=name, md5=md5, filename=filename)
        data[md5] = weight_array
        logger.info("{} {}".format(insert, md5))
    logger.info(layerWeight.count())
    layerWeight.close()
    file_path = get_file_path(filename)
    logger.info("save to {}".format(file_path))
    pickle.dump(data, open(file_path, 'wb'))


def load_layers(layers: list, model_name: str, md5_list: list[str]) -> None:
    """按 MD5 列表从 `WeightDB` 中找回对应权重文件，并加载回模型各层。"""
    layerWeight = WeightDB()
    layerWeight.create()

    md5_i = -1
    for layer in tqdm(layers, desc='load weight'):
        if len(layer.weights) == 0 or md5_i >= len(md5_list) - 1:
            continue
        md5_i += 1

        res = layerWeight.select_by_md5(model_name, md5_list[md5_i])
        # name = layer.name
        # _class = type(layer)._keras_api_names[-1]
        # res = layerWeight.select_by_name(model_name, _class, name)
        if len(res) == 0:
            continue
        elif len(res) > 1:
            logger.warning("error")

        md5, filename = res[0][4], res[0][5]
        file_path = get_file_path(filename)

        if not os.path.exists(file_path):
            logger.warning('file not exist downloading to {}'.format(file_path))

        if os.path.exists(file_path):
            data = pickle.load(open(file_path, 'rb'))

            if md5 in data.keys():
                try:
                    layer.set_weights(data[md5])
                    # [K.set_value(weight, np.array(data[md5][i])) for i, weight in enumerate(layer.weights)]
                except (ValueError, TypeError) as e:
                    logger.error(e)
            else:
                logger.warning('layer weight not find')
        else:
            logger.warning('file not exist')
