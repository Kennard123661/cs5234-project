import numpy as np
import os
import sys
import math
import binary_search as bs

base_dir = os.path.join(os.path.abspath(__file__), '..', '..')
if __name__ == '__main__':
    sys.path.append(base_dir)
from data_structures.base import WriteOptimizedDS

DEFAULT_MEM_SIZE = 100


class NaiveCOA(WriteOptimizedDS):
    def __init__(self, disk_filepath, block_size, n_blocks, n_input_data):
        super(NaiveCOA, self).__init__(disk_filepath, block_size, n_blocks)
        """" this is a naive implementation of cache-oblivious arrays, without fractional cascading """
        self.n_data = n_input_data
        self.n_arrays = math.ceil(math.log(self.n_data, 2))
        self.array = np.empty(2 ** (self.n_arrays + 1), dtype=int)  # each should contain at least two arrays.
        self.n_array_items = np.zeros(shape=self.n_arrays, dtype=int)
        self.has_item = np.zeros_like(self.array, dtype=bool)
        self.n_items = 0

    def query(self, item):
        n_items = self.n_items
        start_idx = 0
        array_size = 1
        i = 0
        while i < (math.ceil(math.log(n_items, 2)) + 1):
            n_array_item = self.n_array_items[i]
            curr_arr = self.array[start_idx:start_idx+array_size]
            if n_array_item > 0:
                search_array = curr_arr[0:n_array_item]
                idx = bs.search(search_array, item)
                if search_array[idx] == item:
                    return start_idx + idx
            array_size = array_size << 1  # multiply by 2
            start_idx += array_size
            i += 1
        return -1  # no found

    def insert(self, item):
        self.n_items += 1
        if self.n_items >= self.n_data:
            raise BufferError('too much data, are you sure you initialized me correctly?')

        start_idx = 0
        array_size = 2
        n_insertions = 1
        prev_array = np.array([item])
        for i in range(self.n_arrays):
            current_arr = self.array[start_idx:start_idx + array_size]
            n_array_item = self.n_array_items[i]

            if (n_insertions + n_array_item) > (array_size >> 1):  # if it will fill up both arrays
                insert_idx = n_array_item + n_insertions - 1

                # do a merge here, inserting the largest item first
                i1 = n_array_item - 1
                i2 = n_insertions - 1
                while i1 >= 0 and i2 >= 0:
                    if current_arr[i1] > prev_array[i2]:
                        current_arr[insert_idx] = current_arr[i1]
                        i1 -= 1
                    else:
                        current_arr[insert_idx] = prev_array[i2]
                        i2 -= 1
                    insert_idx -= 1

                if i1 >= 0:
                    current_arr[:insert_idx + 1] = current_arr[:i1 + 1]
                else:
                    current_arr[:insert_idx + 1] = prev_array[:i2 + 1]
                prev_array = current_arr
                n_insertions += n_array_item
                self.n_array_items[i] = 0
            else:  # simple
                current_arr[n_array_item:n_array_item+n_insertions] = prev_array[0:n_insertions]
                self.n_array_items[i] += n_insertions
                # print(current_arr)
                break
            start_idx += array_size
            array_size = array_size << 1  # multiply by 2


class Cola(WriteOptimizedDS):
    def __init__(self, block_size, n_blocks, size):
        super(Cola, self).__init__(block_size, n_blocks, size)
        """" this is a naive implementation of cache-oblivious arrays """
        self.size = size
        self.n_levels = math.ceil(math.log(self.size, base=2)) + 1
        self.array = np.empty(2 ** (self.n_levels + 1), dtype=int)
        self.n_array_items = np.zeros(shape=self.n_levels, dtype=int)
        self.has_item = np.zeros_like(self.array, dtype=bool)
        self.n_items = 0

    def search(self, item):
        raise NotImplementedError

    def insert(self, item):
        self.n_items += 1
        if self.n_items >= self.size:
            raise BufferError('too much shit')

        start_idx = 0
        array_size = 2
        n_insertions = 1
        prev_array = np.array([item])
        for i in range(self.n_levels):
            current_arr = self.array[start_idx:start_idx + array_size]
            n_array_item = self.n_array_items[i]

            if [n_insertions + n_array_item] >= (array_size >> 1):  # if it will fill up both arrays
                insert_idx = start_idx + n_array_item - 1

                # do a merge here, inserting the largest item first
                i1 = n_array_item - 1
                i2 = n_insertions - 1
                print(insert_idx)
                while i1 >= 0 and i2 >= 0:
                    if current_arr[i1] > prev_array[i2]:
                        current_arr[insert_idx] = current_arr[i1]
                        i1 -= 1
                    else:
                        current_arr[insert_idx] = prev_array[i2]
                        i2 -= 1
                    insert_idx -= 1

                if i1 >= 0:
                    current_arr[:insert_idx+1] = current_arr[:i1+1]
                else:
                    current_arr[:insert_idx+1] = prev_array[:i1+1]
                n_insertions += n_array_item
                self.n_array_items[i] = 0
            else:  # simple
                self.array[start_idx:start_idx+n_array_item] = prev_array[0:n_insertions]
                self.n_array_items[i] += n_insertions
                break
            start_idx += array_size
            array_size = array_size << 1  # multiply by 2


def test_naive_coa():
    ds = NaiveCOA(n_input_data=50)
    ds.insert(1)
    ds.insert(2)
    ds.insert(3)
    ds.insert(0)
    print(ds.array)
    search_idx = ds.query(0)
    print(search_idx)
    search_idx = ds.query(1)
    print(search_idx)
    search_idx = ds.query(2)
    print(search_idx)
    search_idx = ds.query(3)
    print(search_idx)


def main():
    test_naive_coa()


if __name__ == '__main__':
    main()
