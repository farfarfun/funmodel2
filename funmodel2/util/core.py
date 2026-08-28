import hashlib


def get_file_md5(weight):
    m = hashlib.md5()
    m.update(weight)
    return m.hexdigest()
