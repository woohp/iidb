import lmdb
import numpy as np
import zstd
from typing import Any, Union, Iterable, Tuple

__all__ = ['IIDB']


class IIDB:
    """Images Interchange Database"""
    def __init__(self, path: str, readonly: bool = True) -> None:
        self.env = lmdb.open(path, map_size=1024**4, subdir=False, lock=False, readonly=readonly)
        self.compressor = zstd.ZstdCompressor()
        self.decompressor = zstd.ZstdDecompressor()

    def close(self):
        self.env.close()

    def _compress(self, value):
        height, width = value.shape[:2]
        if len(value.shape) == 2:
            channels = 1
        else:
            channels = value.shape[2]
        header = np.array([0, height, width, channels], dtype=np.uint16)
        compressed_blob = self.compressor.compress(value.tobytes())
        return header.tobytes() + compressed_blob

    def get(self, key: Union[int, str]):
        with self.env.begin(write=False) as txn:
            value = txn.get(str(key).encode('utf-8'))

        header = np.frombuffer(value[:8], dtype=np.uint16)
        height = header[1]
        width = header[2]
        channels = header[3]
        out = np.frombuffer(self.decompressor.decompress(value[8:]), dtype=np.uint8)
        if channels == 1:
            return out.reshape((height, width))
        else:
            return out.reshape((height, width, channels))

    def put(self, key: Union[int, str], value):
        with self.env.begin(write=True) as txn:
            txn.put(str(key).encode('utf-8'), self._compress(value), dupdata=False)

    def putmulti(self, items: Iterable[Tuple[Union[str, int], Any]]):
        items_processed = ((str(key).encode('utf-8'), self._compress(value)) for key, value in items)
        with self.env.begin(write=True) as txn:
            return txn.cursor().putmulti(items_processed, dupdata=False)
