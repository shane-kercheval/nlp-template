import math
from typing import List


def get_config():
    pass


def create_batch_start_stop_indexes(length, num_batches) -> List[tuple]:
    """
    This function creates a list of tuples, each tuple being a start/stop index
    (inclusion/exclusion) of the batch.

    E.g. if there are twenty items (i.e. length=20) and we needed 7 batches, this function would
    return
         [
            (0, 3),
            (3, 6),
            (6, 9),
            (9, 12),
            (12, 15),
            (15, 18),
            (18, 20)
        ]

    Each tuple is meant to work with `range()` so the second index of each tuple is an exclusion
    value i.e. [a, b)

    Args:
        length: the length of the entire process i.e. number of items to batch
        num_batches: the number of batches
    """
    assert num_batches <= length
    batch_size = math.floor(length / num_batches)
    index = 0
    batch_index_ranges = []
    for batch in [batch_size] * num_batches:
        batch_index_ranges += [(index, index + batch)]
        index += batch_size

    # set the last element of the last batch range to `length` (keep first element)
    batch_index_ranges[-1] = (batch_index_ranges[-1][0], length)

    assert batch_index_ranges[-1][0] < length
    assert batch_index_ranges[-1][1] == length
    assert all([x[0] < x[1] for x in batch_index_ranges])

    return batch_index_ranges
