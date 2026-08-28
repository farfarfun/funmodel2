import hashlib
import os
import pickle
import sqlite3

from tqdm import tqdm

weight_path = None


def set_weight_path(path):
    global weight_path
    weight_path = path


def get_file_path(filename):
    file_dir = weight_path or os.path.abspath(os.path.dirname(__file__)) + '/../../temp/'
    return os.path.join(file_dir, filename)


def save_weight(data):
    pickle.dump(data, open('file_path', 'wb'))


class WeightDB:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.path.abspath(os.path.dirname(__file__)) + '/layer_weight.db'
        self.db_path = db_path

        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.table_name = "layer_weight"
        self.create()

    def execute(self, sql):
        # print('sql:{}'.format(sql))
        self.conn.commit()
        return self.cursor.execute(sql)

    def create(self):
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

    def insert(self, model='', _class='', name='', md5='', filename=''):
        name = name.replace("'", '')
        res = self.execute(
            """insert into {} (model,class,name,md5,filename) values ('{}','{}','{}','{}','{}')
            """.format(self.table_name, model, _class, name, md5, filename)
        )
        return res

    def insert_if_not_exist(self, model='', _class='', name='', md5='', filename=''):
        name = name.replace("'", '')
        res = self.execute(
            """select * from {} where model='{}' and class='{}' and name='{}' and md5='{}' and filename='{}' limit 10
            """.format(self.table_name, model, _class, name, md5, filename)
        )
        if len(list(res)) == 0:
            self.insert(model, _class, name, md5, filename)
            return True
        return False

    def select_by_name(self, model='', _class='', name=''):
        name = name.replace("'", '')
        res = self.execute(
            """select * from {} where model='{}' and class='{}' and name='{}' limit 10
            """.format(self.table_name, model, _class, name)
        )

        return list(res)

    def select_by_md5(self, model='', md5=''):
        res = self.execute(
            """select * from {} where model='{}' and md5='{}' limit 10
            """.format(self.table_name, model, md5)
        )

        return list(res)

    def count(self):
        res = self.execute("""select count(1) from  {}""".format(self.table_name))

        urls = []
        return urls

    def select(self, size=50):
        res = self.execute("""select * from  {} limit {} """.format(self.table_name, size))

        urls = []
        for line in res:
            urls.append((line[1], line[2]))
        return urls

    def close(self):
        self.cursor.close()
        self.conn.close()


def save_layers(layers, model_name, filename):
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
        print("{} {}".format(insert, md5))
    print(layerWeight.count())
    layerWeight.close()
    file_path = get_file_path(filename)
    print("save to {}".format(file_path))
    pickle.dump(data, open(file_path, 'wb'))


def load_layers(layers, model_name, md5_list):
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
            print("error")

        md5, filename = res[0][4], res[0][5]
        file_path = get_file_path(filename)

        if not os.path.exists(file_path):
            print('file not exist downloading to {}'.format(file_path))

        if os.path.exists(file_path):
            data = pickle.load(open(file_path, 'rb'))

            if md5 in data.keys():
                try:
                    layer.set_weights(data[md5])
                    # [K.set_value(weight, np.array(data[md5][i])) for i, weight in enumerate(layer.weights)]
                except Exception as e:
                    print(e)
            else:
                print('layer weight not find')
        else:
            print('file not exist')
