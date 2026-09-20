import hashlib

from funmodel2.util import get_file_md5


def test_get_file_md5_matches_hashlib():
    data = b"hello funmodel2"
    assert get_file_md5(data) == hashlib.md5(data).hexdigest()


def test_get_file_md5_differs_for_different_input():
    assert get_file_md5(b"a") != get_file_md5(b"b")
