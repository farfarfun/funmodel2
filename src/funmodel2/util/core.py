import hashlib


def get_file_md5(weight: bytes) -> str:
    """计算字节内容的 MD5 摘要。

    参数:
        weight: 要计算摘要的字节内容。

    返回:
        十六进制 MD5 摘要。
    """
    m = hashlib.md5()
    m.update(weight)
    return m.hexdigest()
