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
        self.array = np.empty(2**self.n_arrays, dtype=int)
        self.has_item = np.zeros_like(self.array, dtype=bool)
        self.n_items = 0

    def search(self, item):
        last_level = math.ceil(math.log(self.n_items, base=2))

        idx = -1
        start_idx = 0
        for i in range(last_level + 1):
            end_idx = 2**i
            search_array = self.array[start_idx:end_idx]
            idx = bs.search(search_array, item)
            if search_array[idx] == item and self.has_item[idx]:
                idx += start_idx
                break  # idx is found
            else:
                idx = -1  # idx is not yet found
            start_idx = end_idx
        return idx

    def insert(self, item):
        if (self.n_items + 1) >= self.size:
            raise BufferError('size is too small')
        self.n_items += 1  # increment counter

        start_idx = 0
        items_to_insert = [item]
        for i in range(self.n_arrays):
            end_idx = 2**(i+1)
            subarray = self.array[start_idx:end_idx]
            has_items = self.has_item[start_idx:end_idx]

            if np.all(has_items):  # array is full, merge the arrays
                n_insertions = len(items_to_insert)
                n_new_insertions = len(has_items)
                merged_array = list()
                idx, new_idx = 0, 0
                while (idx < n_insertions) and (new_idx < n_new_insertions):
                    item = items_to_insert[idx]
                    new_item = subarray[new_idx]
                    if item <= new_item:
                        merged_array.append(item)
                    else:
                        merged_array.append(new_item)

                if idx < n_insertions:
                    merged_array += items_to_insert[idx:]
                    assert new_idx == n_new_insertions
                elif new_idx < n_new_insertions:
                    merged_array += subarray[new_idx:]
                    assert idx == n_insertions

                items_to_insert = merged_array
                self.has_item &= False  # bitmasking
            else:  # array has space, so we can insert
                insert_idx = 0
                for j, has_item in enumerate(self.has_item):
                    if not has_item:
                        insert_idx = j
                        break
                n_insertions = len(items_to_insert)
                final_idx = insert_idx + n_insertions
                self.array[insert_idx:final_idx] = np.array(items_to_insert)
                self.has_item[insert_idx:final_idx] |= True
                break
            start_idx = end_idx


class Cola:
    def __init__(self, size):
        """"
        this is a naive implementation of cache-oblivious arrays
        """
        self.size = size
        self.n_arrays = math.ceil(math.log(self.size, base=2)) + 1
        self.array = np.empty(2 ** (self.n_arrays + 1), dtype=int)
        self.has_item = np.zeros_like(self.array, dtype=bool)
        self.n_items = 0

    def search(self, item):
        raise NotImplementedError

    @staticmethod
    def get_insert_idx(array, length):
        ones_idx = 0
        zeros_idx = length - 1
        if array[ones_idx] == 0:
            return 0

        while True:
            mid_idx = (ones_idx + zeros_idx) // 2
            if mid_idx:
                ones_idx = mid_idx
            else:
                zeros_idx = mid_idx

            if ones_idx ==


    def insert(self, item):
        self.n_items += 1
        if self.n_items >= self.size:
            raise BufferError('too much shit')

        start_idx = 0
        array_size = 2
        for i in range(self.n_arrays):
            array_size = array_size << 2
            insert_idx = np.argwhere(self.has_item[start_idx:start_idx+(array_size >> 2)])[-1]
            if insert_idx >= :
                co
            start_idx += array_size


def main():
    raise NotImplementedError


if __name__ == '__main__':
    main()
