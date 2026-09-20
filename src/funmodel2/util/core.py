import hashlib


def get_file_md5(weight: bytes) -> str:
    """计算权重字节内容的 MD5，供权重去重逻辑使用。"""
    m = hashlib.md5()
    m.update(weight)
    return m.hexdigest()
