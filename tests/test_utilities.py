import unittest
from source.library.utilities import create_batch_start_stop_indexes


class TestUtilities(unittest.TestCase):

    def test__create_batch_start_stop_indexes(self):
        num_batches = 7
        length = 20
        batch_indexes = create_batch_start_stop_indexes(length=length, num_batches=num_batches)
        self.assertEqual(len(batch_indexes), num_batches)
        self.assertEqual(batch_indexes[num_batches - 1][1], length)
        # make sure each index is accounted for (and only referenced once)
        indexes_found = []
        for batch in batch_indexes:
            for index in range(batch[0], batch[1]):
                indexes_found += [index]
        self.assertEqual(indexes_found, list(range(0, length)))
        del num_batches, length, batch_indexes, indexes_found, batch, index

        num_batches = 1
        length = 20
        batch_indexes = create_batch_start_stop_indexes(length=length, num_batches=num_batches)
        self.assertEqual(len(batch_indexes), num_batches)
        self.assertEqual(batch_indexes[num_batches - 1][1], length)
        # make sure each index is accounted for (and only referenced once)
        indexes_found = []
        for batch in batch_indexes:
            for index in range(batch[0], batch[1]):
                indexes_found += [index]
        self.assertEqual(indexes_found, list(range(0, length)))
        del num_batches, length, batch_indexes, indexes_found, batch, index

        num_batches = 5
        length = 20
        batch_indexes = create_batch_start_stop_indexes(length=length, num_batches=num_batches)
        self.assertEqual(len(batch_indexes), num_batches)
        self.assertEqual(batch_indexes[num_batches - 1][1], length)
        # make sure each index is accounted for (and only referenced once)
        indexes_found = []
        for batch in batch_indexes:
            for index in range(batch[0], batch[1]):
                indexes_found += [index]
        self.assertEqual(indexes_found, list(range(0, length)))
        del num_batches, length, batch_indexes, indexes_found, batch, index

        num_batches = 10
        length = 10
        batch_indexes = create_batch_start_stop_indexes(length=length, num_batches=num_batches)
        self.assertEqual(len(batch_indexes), num_batches)
        self.assertEqual(batch_indexes[num_batches - 1][1], length)
        # make sure each index is accounted for (and only referenced once)
        indexes_found = []
        for batch in batch_indexes:
            for index in range(batch[0], batch[1]):
                indexes_found += [index]
        self.assertEqual(indexes_found, list(range(0, length)))
        del num_batches, length, batch_indexes, indexes_found, batch, index

        # this was causing an issue because the last index had the start > end
        num_batches = 240
        length = 22883
        batch_indexes = create_batch_start_stop_indexes(length=length, num_batches=num_batches)
        self.assertEqual(len(batch_indexes), num_batches)
        self.assertEqual(batch_indexes[num_batches - 1][1], length)
        # make sure each index is accounted for (and only referenced once)
        indexes_found = []
        for batch in batch_indexes:
            for index in range(batch[0], batch[1]):
                indexes_found += [index]
        self.assertEqual(indexes_found, list(range(0, length)))
        del num_batches, length, batch_indexes, indexes_found, batch, index

        self.assertRaises(
            AssertionError,
            lambda: create_batch_start_stop_indexes(length=10, num_batches=11)
        )
