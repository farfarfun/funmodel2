import hashlib


def get_file_md5(weight: bytes) -> str:
    """计算字节内容的 MD5 摘要。"""
    m = hashlib.md5()
    m.update(weight)
    return m.hexdigest()
