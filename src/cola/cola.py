import numpy as np
import math
import binary_search as bs


class NaiveCoa:
    def __init__(self, size):
        """"
        this is a naive implementation of cache-oblivious arrays
        """
        self.size = size
        self.n_arrays = math.ceil(math.log(self.size, base=2)) + 1
        self.array = np.empty(2 ** (self.n_arrays + 1), dtype=int)
        self.n_array_items = np.zeros(shape=self.n_arrays, dtype=int)
        self.has_item = np.zeros_like(self.array, dtype=bool)
        self.n_items = 0

    def search(self, item):
        n_items = self.n_items
        start_idx = 0
        array_size = 1
        i = 0
        while start_idx < n_items:
            search_array = self.array[start_idx:start_idx+self.n_array_items[i]]
            idx = bs.search(search_array, item)
            if search_array[idx] == item:
                return start_idx + idx
            array_size = array_size << 1  # multiply by 2
            start_idx += array_size
            i += 1
        return -1  # no found

    def insert(self, item):
        self.n_items += 1
        if self.n_items >= self.size:
            raise BufferError('too much shit')

        start_idx = 0
        array_size = 2
        n_insertions = 1
        prev_array = np.array([item])
        for i in range(self.n_arrays):
            current_arr = self.array[start_idx:start_idx + array_size]
            n_array_item = self.n_array_items[i]

            if [n_insertions + n_array_item] >= (array_size >> 1):  # if it will fill up both arrays
                insert_idx = start_idx + n_array_item - 1

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
                    current_arr[:insert_idx + 1] = prev_array[:i1 + 1]
                n_insertions += n_array_item
                self.n_array_items[i] = 0
            else:  # simple
                self.array[start_idx:start_idx + n_array_item] = prev_array[0:n_insertions]
                self.n_array_items[i] += n_insertions
                break
            start_idx += array_size
            array_size = array_size << 1  # multiply by 2


class Cola:
    def __init__(self, size):
        """"
        this is a naive implementation of cache-oblivious arrays
        """
        self.size = size
        self.n_arrays = math.ceil(math.log(self.size, base=2)) + 1
        self.array = np.empty(2 ** (self.n_arrays + 1), dtype=int)
        self.n_array_items = np.zeros(shape=self.n_arrays, dtype=int)
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
        for i in range(self.n_arrays):
            current_arr = self.array[start_idx:start_idx + array_size]
            n_array_item = self.n_array_items[i]

            if [n_insertions + n_array_item] >= (array_size >> 1):  # if it will fill up both arrays
                insert_idx = start_idx + n_array_item - 1

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


def main():
    raise NotImplementedError


if __name__ == '__main__':
    main()
