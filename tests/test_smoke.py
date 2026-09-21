import funmodel2
from funmodel2.database import WeightDB
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
