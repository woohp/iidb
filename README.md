## Usage
```python
import iidb
import numpy as np

# put images into db
db = iidb.open('images.mdb', readonly=False)
img = np.zeros((100, 100), dtype=np.uint8)
db[213] = img
db.close()

# read images back
db = iidb.open('images.mdb')  # readonly by default
img2 = db[213]
np.testing.assert_allclose(img2, img)
```
