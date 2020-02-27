import unittest
import random
import iidb
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
        db = iidb.open('test.mdb', readonly=False)
        data = self._make_array()
        db[123] = data
        self.assertTrue(123 in db)
        self.assertFalse(234 in db)
        self.assertFalse(db.closed)
        db.close()
        self.assertTrue(db.closed)

        db2 = iidb.open('test.mdb', readonly=True)
        np.testing.assert_array_equal(db2[123], data)

    def test_basic_put_and_get_lz4(self):
        db = iidb.open('test.mdb', readonly=False, mode=1)
        data = self._make_array()
        db[123] = data
        self.assertTrue(123 in db)
        self.assertFalse(234 in db)
        self.assertFalse(db.closed)
        db.close()
        self.assertTrue(db.closed)

        db2 = iidb.open('test.mdb', readonly=True)
        np.testing.assert_array_equal(db2[123], data)

    def test_channels(self):
        db = iidb.open('test.mdb', readonly=False)
        data = self._make_array((5, 5, 3))
        db[234] = data
        db.close()

        db2 = iidb.open('test.mdb', readonly=True)
        np.testing.assert_array_equal(db2[234], data)

    def test_put_multiple(self):
        data = [
            (1, self._make_array()),
            (2, self._make_array()),
        ]
        db = iidb.open('test.mdb', readonly=False)
        db.putmulti(data)
        db.close()

        db2 = iidb.open('test.mdb', readonly=False)
        np.testing.assert_array_equal(db2[1], data[0][1])
        np.testing.assert_array_equal(db2[2], data[1][1])

    def test_get_dimension(self):
        db = iidb.open('test.mdb', readonly=False)
        data = self._make_array((4, 5))
        db[123] = data
        self.assertEqual(db.get_image_dimension(123), (4, 5))

    def test_context_manager(self):
        with iidb.open('test.mdb', readonly=False) as db:
            data = self._make_array((4, 5))
            db[123] = data
            self.assertEqual(db.get_image_dimension(123), (4, 5))
            self.assertFalse(db.closed)

        self.assertTrue(db.closed)
