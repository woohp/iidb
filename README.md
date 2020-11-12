## IIDB

IIDB is a key-value database designed for storing images, with a heavy focus on speed. IIDB is a lightweight wrapper around LMDB and encodes data using zstd or lz4.


## Usage
```python
import iidb
import numpy as np

# put images into db
with iidb.open('images.mdb', readonly=False) as db:
    img = np.zeros((100, 100), dtype=np.uint8)
    db[213] = img

# read images back
with iidb.open('images.mdb') as db:  # readonly by default 
    img2 = db[213]
np.testing.assert_allclose(img2, img)
```
