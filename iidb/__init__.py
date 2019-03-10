import lmdb
import numpy as np
import struct
import zstd
from typing import Any, Union, Iterable, Tuple

__all__ = ['IIDB']


class IIDB:
    """Images Interchange Database"""

    compressor = zstd.ZstdCompressor()
    decompressor = zstd.ZstdDecompressor()
    header_packer = struct.Struct('<HHHH')

    def __init__(self, path: str, readonly: bool = True) -> None:
        self.path = path
        self.readonly = readonly
        self.env = lmdb.open(path, map_size=1024**4, subdir=False, lock=False, readonly=readonly)

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

    @property
    def closed(self):
        return self.env is None

    def _compress(self, value) -> bytes:
        height, width = value.shape[:2]
        if len(value.shape) == 2:
            channels = 1
        else:
            channels = value.shape[2]
        header_blob = self.header_packer.pack(0, height, width, channels)
        compressed_blob = self.compressor.compress(value.tobytes())
        return header_blob + compressed_blob

    def get(self, key: Union[int, str]):
        with self.env.begin(write=False, buffers=True) as txn:
            buf = txn.get(str(key).encode('utf-8'))

            header = self.header_packer.unpack(buf[:8])
            height = header[1]
            width = header[2]
            channels = header[3]
            out = np.frombuffer(self.decompressor.decompress(buf[8:]), dtype=np.uint8)
        if channels == 1:
            return out.reshape((height, width))
        else:
            return out.reshape((height, width, channels))

    __getitem__ = get

    def getmulti(self, keys: Iterable[Union[int, str]]):
        output = []

        with self.env.begin(write=False, buffers=True) as txn:
            for key in keys:
                buf = txn.get(str(key).encode('utf-8'))

                header = self.header_packer.unpack(buf[:8])
                height = header[1]
                width = header[2]
                channels = header[3]
                out = np.frombuffer(self.decompressor.decompress(buf[8:]), dtype=np.uint8)
                if channels == 1:
                    output.append(out.reshape((height, width)))
                else:
                    output.append(out.reshape((height, width, channels)))

        return output

    def put(self, key: Union[int, str], value):
        with self.env.begin(write=True) as txn:
            txn.put(str(key).encode('utf-8'), self._compress(value), dupdata=False)

    __setitem__ = put

    def __contains__(self, key: Union[int, str]) -> bool:
        with self.env.begin(write=False, buffers=True) as txn:
            return txn.get(str(key).encode('utf-8')) is not None

    def putmulti(self, items: Iterable[Tuple[Union[str, int], Any]]):
        items_processed = ((str(key).encode('utf-8'), self._compress(value)) for key, value in items)
        with self.env.begin(write=True) as txn:
            return txn.cursor().putmulti(items_processed, dupdata=False)

    def get_image_dimension(self, key: Union[int, str]) -> Tuple[int, int]:
        with self.env.begin(write=False, buffers=True) as txn:
            buf = txn.get(str(key).encode('utf-8'))
            header = self.header_packer.unpack(buf[:8])
            height = header[1]
            width = header[2]

        return (height, width)


open = IIDB
