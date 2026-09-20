import os
import pickle

import numpy as np
import pytest

from funmodel2.database import WeightDB, load_layers, save_layers, set_weight_path


@pytest.fixture()
def db(tmp_path):
    weight_db = WeightDB(db_path=str(tmp_path / "layer_weight.db"))
    yield weight_db
    weight_db.close()


def test_insert_if_not_exist_inserts_once(db):
    assert db.insert_if_not_exist(model="m", _class="Conv2D", name="c1", md5="abc", filename="f.weight") is True
    assert db.insert_if_not_exist(model="m", _class="Conv2D", name="c1", md5="abc", filename="f.weight") is False


def test_select_by_md5_and_by_name(db):
    db.insert_if_not_exist(model="m", _class="Conv2D", name="c1", md5="abc", filename="f.weight")

    assert len(db.select_by_md5(model="m", md5="abc")) == 1
    assert len(db.select_by_name(model="m", _class="Conv2D", name="c1")) == 1
    assert len(db.select_by_md5(model="m", md5="does-not-exist")) == 0


class _FakeWeight:
    def __init__(self, array: np.ndarray):
        self._array = array

    def numpy(self) -> np.ndarray:
        return self._array


class _FakeLayer:
    def __init__(self, name: str, weights: list[np.ndarray]):
        self.name = name
        self.weights = [_FakeWeight(w) for w in weights]
        self._raw_weights = weights

    def set_weights(self, data):
        self._raw_weights = data


class _FakeApiNames(list):
    pass


def _tag_class(layer: _FakeLayer, keras_api_name: str) -> _FakeLayer:
    type(layer)._keras_api_names = _FakeApiNames([keras_api_name])
    return layer


def test_save_layers_then_load_layers_round_trip(tmp_path):
    set_weight_path(str(tmp_path))

    layer = _tag_class(_FakeLayer("conv1", [np.array([1.0, 2.0, 3.0])]), "Conv2D")
    save_layers([layer], model_name="yolov3", filename="yolov3.weight")

    saved_path = tmp_path / "yolov3.weight"
    assert saved_path.exists()
    with open(saved_path, "rb") as f:
        data = pickle.load(f)
    assert len(data) == 1
    (md5,) = data.keys()

    target_layer = _FakeLayer("conv1", [np.zeros(3)])
    load_layers([target_layer], model_name="yolov3", md5_list=[md5])

    np.testing.assert_array_equal(target_layer._raw_weights[0], np.array([1.0, 2.0, 3.0]))
