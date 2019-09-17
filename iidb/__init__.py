import lmdb
import numpy as np
import struct
import zstd
import lz4.block
from typing import Iterable, List, Optional, Tuple, Union

__all__ = ['IIDB']


KeyType = Union[int, str]


class IIDB:
    """Images Interchange Database"""

    zstd_compressor = zstd.ZstdCompressor()
    zstd_decompressor = zstd.ZstdDecompressor()
    header_packer = struct.Struct('<HHHH')

    def __init__(self, path: str, readonly: bool = True, mode: int = 0) -> None:
        self.path = path
        self.readonly = readonly
        self.env = lmdb.open(path, map_size=1024**4, subdir=False, lock=False, readonly=readonly)
        self.mode = mode

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def __repr__(self) -> str:
        return f'IIDB(path={self.path!r}, readonly={self.readonly})'

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def __len__(self) -> int:
        with self.env.begin() as txn:
            return txn.stat()['entries']

    @property
    def closed(self):
        return self.env is None

    def _compress(self, value: np.ndarray) -> bytes:
        height, width = value.shape[:2]
        if len(value.shape) == 2:
            channels = 1
        else:
            channels = value.shape[2]

        header_blob = self.header_packer.pack(self.mode, height, width, channels)
        if self.mode == 0:
            compressed_blob = self.zstd_compressor.compress(value.tobytes())
        else:
            compressed_blob = lz4.block.compress(
                value.tobytes(), mode='high_compression', compression=7, store_size=False
            )

        return header_blob + compressed_blob

    def _decompress(self, buf: memoryview) -> np.ndarray:
        header = self.header_packer.unpack(buf[:8])
        mode = header[0]
        height = header[1]
        width = header[2]
        channels = header[3]

        # decompress based on the correct mode
        if mode == 0:
            decompressed_bytes = self.zstd_decompressor.decompress(buf[8:])
        else:
            decompressed_bytes = lz4.block.decompress(buf[8:], height * width * channels)
        out = np.frombuffer(decompressed_bytes, dtype=np.uint8)

        if channels == 1:
            return out.reshape((height, width))
        else:
            return out.reshape((height, width, channels))

    def get(self, key: KeyType) -> np.ndarray:
        with self.env.begin(write=False, buffers=True) as txn:
            buf: Optional[memoryview] = txn.get(str(key).encode('utf-8'))
            if buf is None:
                raise KeyError(key)
            return self._decompress(buf)

    __getitem__ = get

    def getmulti(self, keys: Iterable[KeyType]) -> List[np.ndarray]:
        output = []

        with self.env.begin(write=False, buffers=True) as txn:
            for key in keys:
                buf: Optional[memoryview] = txn.get(str(key).encode('utf-8'))
                if buf is None:
                    raise KeyError(key)
                output.append(self._decompress(buf))

        return output

    def put(self, key: KeyType, value: np.ndarray):
        with self.env.begin(write=True) as txn:
            txn.put(str(key).encode('utf-8'), self._compress(value), dupdata=False)

    __setitem__ = put

    def __contains__(self, key: KeyType) -> bool:
        with self.env.begin(write=False, buffers=True) as txn:
            return txn.get(str(key).encode('utf-8')) is not None

    def putmulti(self, items: Iterable[Tuple[KeyType, np.ndarray]]):
        items_processed = ((str(key).encode('utf-8'), self._compress(value)) for key, value in items)
        with self.env.begin(write=True) as txn:
            return txn.cursor().putmulti(items_processed, dupdata=False)

    def get_image_dimension(self, key: KeyType) -> Tuple[int, int]:
        with self.env.begin(write=False, buffers=True) as txn:
            buf: Optional[memoryview] = txn.get(str(key).encode('utf-8'))
            if buf is None:
                raise KeyError(key)
            header = self.header_packer.unpack(buf[:8])
            height = header[1]
            width = header[2]

        return (height, width)


open = IIDB
