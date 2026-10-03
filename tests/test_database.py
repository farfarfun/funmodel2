import os
import numpy as np
import pytest

from funmodel2.database import core as database_core
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


def test_select_rejects_negative_size(db):
    with pytest.raises(ValueError, match="non-negative"):
        db.select(size=-1)


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


def _use_temporary_database(monkeypatch, tmp_path):
    db_path = tmp_path / "layer_weight.db"
    monkeypatch.setattr(database_core, "WeightDB", lambda: WeightDB(db_path=db_path))
    return db_path


def test_save_layers_then_load_layers_round_trip(monkeypatch, tmp_path):
    _use_temporary_database(monkeypatch, tmp_path)
    set_weight_path(str(tmp_path))

    layer = _tag_class(_FakeLayer("conv1", [np.array([1.0, 2.0, 3.0])]), "Conv2D")
    save_layers([layer], model_name="yolov3", filename="yolov3.weight")

    saved_path = tmp_path / "yolov3.weight"
    assert saved_path.exists()
    with np.load(saved_path, allow_pickle=False) as data:
        assert len(data.files) == 1
        (key,) = data.files
    md5, _ = key.split(":")

    target_layer = _FakeLayer("conv1", [np.zeros(3)])
    load_layers([target_layer], model_name="yolov3", md5_list=[md5])

    np.testing.assert_array_equal(target_layer._raw_weights[0], np.array([1.0, 2.0, 3.0]))


def test_load_layers_skips_missing_cache_file(monkeypatch, tmp_path):
    set_weight_path(str(tmp_path))
    db_path = _use_temporary_database(monkeypatch, tmp_path)
    db = WeightDB(db_path=db_path)
    db.insert(model="m", _class="Conv2D", name="c1", md5="missing", filename="missing.bin")
    db.close()

    layer = _FakeLayer("conv1", [np.zeros(3)])
    load_layers([layer], model_name="m", md5_list=["missing"])

    np.testing.assert_array_equal(layer._raw_weights[0], np.zeros(3))


@pytest.mark.parametrize("cached_data", [{"other": [np.ones(3)]}, None])
def test_load_layers_skips_missing_or_corrupt_cached_data(monkeypatch, tmp_path, cached_data):
    set_weight_path(tmp_path)
    db_path = _use_temporary_database(monkeypatch, tmp_path)
    db = WeightDB(db_path=db_path)
    db.insert(model="m", _class="Conv2D", name="c1", md5="wanted", filename="cache.bin")
    db.close()

    cache_path = tmp_path / "cache.bin"
    if cached_data is None:
        cache_path.write_bytes(b"not an npz archive")
    else:
        np.savez_compressed(cache_path, **{"other:0": cached_data["other"][0]})

    layer = _FakeLayer("conv1", [np.zeros(3)])
    load_layers([layer], model_name="m", md5_list=["wanted"])

    np.testing.assert_array_equal(layer._raw_weights[0], np.zeros(3))


def test_load_layers_skips_set_weights_failure(monkeypatch, tmp_path):
    class FailingLayer(_FakeLayer):
        def set_weights(self, data):
            raise ValueError("incompatible weights")

    _use_temporary_database(monkeypatch, tmp_path)
    set_weight_path(tmp_path)
    source = _FakeLayer("conv1", [np.ones(3)])
    save_layers([source], model_name="m", filename="cache.bin")
    with np.load(tmp_path / "cache.bin", allow_pickle=False) as data:
        md5, _ = data.files[0].split(":")

    target = FailingLayer("conv1", [np.zeros(3)])
    load_layers([target], model_name="m", md5_list=[md5])

    np.testing.assert_array_equal(target._raw_weights[0], np.zeros(3))


def test_set_weight_path_controls_saved_file_location(monkeypatch, tmp_path):
    _use_temporary_database(monkeypatch, tmp_path)
    configured_path = tmp_path / "weights"
    set_weight_path(configured_path)

    save_layers([_FakeLayer("conv1", [np.ones(1)])], model_name="m", filename="cache.bin")

    assert (configured_path / "cache.bin").is_file()


@pytest.mark.parametrize("filename", ["../outside.bin", "/tmp/outside.bin", "nested/cache.bin", ""])
def test_save_layers_rejects_unsafe_filename(monkeypatch, tmp_path, filename):
    _use_temporary_database(monkeypatch, tmp_path)
    set_weight_path(tmp_path)

    with pytest.raises(ValueError, match="basename"):
        save_layers([_FakeLayer("conv1", [np.ones(1)])], model_name="m", filename=filename)


def test_database_rejects_unsafe_filename(db):
    with pytest.raises(ValueError, match="basename"):
        db.insert(model="m", filename="../outside.bin")
