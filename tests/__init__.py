import unittest
import random
from iidb import IIDB
import numpy as np
import os


class IIDBTestCase(unittest.TestCase):
    def tearDown(self):
        if os.path.exists('test.mdb'):
            os.remove('test.mdb')

    @staticmethod
    def _make_array(dims=(5, 5)):
        total = np.prod(dims)
        start = random.randint(0, 1000)
        return np.arange(start, start + total, dtype=np.uint8).reshape(dims)

    def test_basic_put_and_get(self):
        db = IIDB('test.mdb', readonly=False)
        data = self._make_array()
        db.put(123, data)
        db.close()

        db2 = IIDB('test.mdb', readonly=True)
        np.testing.assert_array_equal(db2.get('123'), data)

    def test_channels(self):
        db = IIDB('test.mdb', readonly=False)
        data = self._make_array((5, 5, 3))
        db.put(234, data)
        db.close()

        db2 = IIDB('test.mdb', readonly=True)
        np.testing.assert_array_equal(db2.get('234'), data)

    def test_put_multiple(self):
        data = [
            (1, self._make_array()),
            (2, self._make_array()),
        ]
        db = IIDB('test.mdb', readonly=False)
        db.putmulti(data)
        db.close()

        db2 = IIDB('test.mdb', readonly=False)
        np.testing.assert_array_equal(db2.get(1), data[0][1])
        np.testing.assert_array_equal(db2.get(2), data[1][1])

    def test_getitem_and_setitem(self):
        db = IIDB('test.mdb', readonly=False)
        data = self._make_array()
        db[123] = data
        np.testing.assert_array_equal(db.get(123), data)

        data = self._make_array()
        db.put(234, data)
        np.testing.assert_array_equal(db.get(234), data)
