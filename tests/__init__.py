import unittest
from iidb import IIDB
import numpy as np
import os


class IIDBTestCase(unittest.TestCase):
    def testDown(self):
        if os.path.exists('test.mdb'):
            os.remove('test.mdb')

    def test_basic_put_and_get(self):
        db = IIDB('test.mdb', readonly=False)
        data = np.arange(25, dtype=np.uint8).reshape((5, 5))
        db.put(123, data)
        db.close()

        db2 = IIDB('test.mdb', readonly=True)
        data2 = db2.get('123')
        np.testing.assert_array_equal(data2, data)

    def test_channels(self):
        db = IIDB('test.mdb', readonly=False)
        data = np.arange(75, dtype=np.uint8).reshape((5, 5, 3))
        db.put(234, data)
        db.close()

        db2 = IIDB('test.mdb', readonly=True)
        data2 = db2.get('234')
        np.testing.assert_array_equal(data2, data)

    def test_put_multiple(self):
        data = [
            (1, np.arange(0, 25, dtype=np.uint8).reshape((5, 5))),
            (2, np.arange(5, 30, dtype=np.uint8).reshape((5, 5))),
        ]
        db = IIDB('test.mdb', readonly=False)
        db.putmulti(data)
        db.close()

        db2 = IIDB('test.mdb', readonly=False)
        np.testing.assert_array_equal(db2.get(1), data[0][1])
        np.testing.assert_array_equal(db2.get(2), data[1][1])
