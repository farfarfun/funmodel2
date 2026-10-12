import sys
from types import ModuleType

import funmodel2
from funmodel2.database import WeightDB
from funmodel2.models import yolo
from funmodel2.models.yolo import get_anchors, get_md5
from funmodel2.util import get_file_md5


def test_import_funmodel2():
    assert funmodel2 is not None


def test_weight_db_round_trip(tmp_path):
    db = WeightDB(tmp_path / "weights.db")
    try:
        assert db.insert_if_not_exist("yolo", "Conv2D", "conv1", "abc", "weights.bin")
        assert not db.insert_if_not_exist("yolo", "Conv2D", "conv1", "abc", "weights.bin")
        assert db.count() == 1
        assert db.select_by_md5("yolo", "abc")[0][4] == "abc"
    finally:
        db.close()


def test_public_hash_and_anchor_helpers():
    assert get_md5(b"weights") == get_file_md5(b"weights")
    assert get_anchors().shape == (9, 2)


def test_load_yolo_models_uses_default_classes_and_weights_directory(monkeypatch, tmp_path):
    calls = []

    class FakeYoloBody:
        def __init__(self, *, anchors, num_classes):
            self.anchors = anchors
            self.num_classes = num_classes
            calls.append(("create", num_classes))

        def load_weights(self, path, *, freeze_body):
            calls.append(("load_weights", path, freeze_body))

        def load_layer_weights(self):
            calls.append(("load_layer_weights",))

    funkeras = ModuleType("funkeras")
    models = ModuleType("funkeras.models")
    yolo3 = ModuleType("funkeras.models.yolo3")
    utils = ModuleType("funkeras.utils")
    yolo3.YoloBody = FakeYoloBody
    utils.read_lines = lambda path: calls.append(("read_lines", path)) or ["person", "car"]
    monkeypatch.setitem(sys.modules, "funkeras", funkeras)
    monkeypatch.setitem(sys.modules, "funkeras.models", models)
    monkeypatch.setitem(sys.modules, "funkeras.models.yolo3", yolo3)
    monkeypatch.setitem(sys.modules, "funkeras.utils", utils)
    monkeypatch.setattr(yolo, "set_weight_path", lambda path: calls.append(("set_weight_path", path)))

    weights_path = tmp_path / "weights.h5"
    first, second = yolo.load_yolo_models(weights_path)

    assert first.num_classes == second.num_classes == 2
    assert ("read_lines", str(yolo.files("funmodel2.models").joinpath("coco.names"))) in calls
    assert ("set_weight_path", str(tmp_path)) in calls
    assert ("load_weights", str(weights_path), 3) in calls
    assert ("load_layer_weights",) in calls


def test_load_yolo_models_uses_explicit_classes_file(monkeypatch, tmp_path):
    classes_path = tmp_path / "classes.txt"
    read_paths = []

    class FakeYoloBody:
        def __init__(self, **_kwargs):
            pass

        def load_weights(self, *_args, **_kwargs):
            pass

        def load_layer_weights(self):
            pass

    funkeras = ModuleType("funkeras")
    models = ModuleType("funkeras.models")
    yolo3 = ModuleType("funkeras.models.yolo3")
    utils = ModuleType("funkeras.utils")
    yolo3.YoloBody = FakeYoloBody
    utils.read_lines = lambda path: read_paths.append(path) or ["custom"]
    monkeypatch.setitem(sys.modules, "funkeras", funkeras)
    monkeypatch.setitem(sys.modules, "funkeras.models", models)
    monkeypatch.setitem(sys.modules, "funkeras.models.yolo3", yolo3)
    monkeypatch.setitem(sys.modules, "funkeras.utils", utils)
    monkeypatch.setattr(yolo, "set_weight_path", lambda _path: None)

    yolo.load_yolo_models(tmp_path / "weights.h5", classes_path)

    assert read_paths == [str(classes_path)]
