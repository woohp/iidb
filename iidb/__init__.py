import lmdb
import numpy as np
import zstd
from typing import Any, Union, Iterable, Tuple


class IIDB:
    """Images Interchange Database"""
    def __init__(self, path: str, readonly: bool = True) -> None:
        self.env = lmdb.open(path, map_size=1024**4, subdir=False, lock=False, readonly=readonly)
        self.compressor = zstd.ZstdCompressor()
        self.decompressor = zstd.ZstdDecompressor()

    def get(self, key: Union[int, str]):
        with self.env.begin(write=False) as txn:
            value = txn.get(str(key).encode('utf-8'))

        header = np.frombuffer(value[:8], dtype=np.uint16)
        height = header[1]
        width = header[2]
        channels = header[3]
        return np.frombuffer(self.decompressor.decompress(value[8:]), dtype=np.uint8).reshape((height, width, channels))

    def put(self, key: Union[int, str], value):
        height, width = value.shape[:2]
        if len(value.shape) == 2:
            channels = 1
        else:
            channels = value.shape[2]
        header = np.array([0, height, width, channels], dtype=np.uint16)
        compressed_blob = self.compressor.compress(value.tobytes())

        with self.env.begin(write=True) as txn:
            txn.put(str(key).encode('utf-8'), header.tobytes() + compressed_blob, dupdata=False)

    def putmulti(self, items: Iterable[Tuple[Union[str, int], Any]]):
        items = ((str(key).encode('utf-8'), value) for key, value in items)
        with self.env.begin(write=True) as txn:
            return txn.cursor().putmulti(items, dupdata=False)
