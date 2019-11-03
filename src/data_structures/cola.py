import numpy as np
import os
import sys
import math
import h5py
import binary_search as bs

base_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..'))
if __name__ == '__main__':
    sys.path.append(base_dir)
storage_dir = os.path.join(base_dir, 'storage')


from data_structures.base import WriteOptimizedDS

DEFAULT_MEM_SIZE = 100


class NaiveCOA(WriteOptimizedDS):
    def __init__(self, disk_filepath, block_size, n_blocks, n_input_data):
        super(NaiveCOA, self).__init__(disk_filepath, block_size, n_blocks, n_input_data)
        print(self.disk_filepath)

        """ this is a naive implementation of cache-oblivious arrays, without fractional cascading """
        self.n_arrays = math.ceil(math.log(self.n_input_data, 2))
        self.n_array_items = np.zeros(shape=self.n_arrays, dtype=int)
        self.total_array_size = 2 ** (self.n_arrays + 1)
        self.arrays = np.empty(self.total_array_size, dtype=int)  # each should contain at least two arrays.
        self.n_items = 0

        disk_dir = os.path.dirname(self.disk_filepath)
        if not os.path.exists(disk_dir):
            os.makedirs(disk_dir)
        elif os.path.exists(self.disk_filepath):
            os.remove(self.disk_filepath)

        disk = h5py.File(self.disk_filepath, 'w')
        disk.create_dataset('dset', shape=(self.total_array_size + self.block_size, ), dtype='i8')
        disk.close()
        self.disk = h5py.File(self.disk_filepath, 'r+')
        self.disk_data = self.disk['dset']

    def __del__(self):
        self.disk.close()
        del self.disk_data

    def load_from_disk(self, start_idx, end_idx):
        """
        :param start_idx: start idx to load form
        :param end_idx: end idx to load from
        :return: no. of chunks to load
        """
        # simulate loading blocks from cache
        n_blocks = int(math.ceil((end_idx - start_idx) / self.block_size) * self.block_size)
        loaded_data = np.zeros(shape=(n_blocks*self.block_size), dtype=int)
        block_start_idx = start_idx
        block_end_idx = start_idx + self.block_size
        for _ in range(n_blocks):
            loaded_data[block_start_idx:block_end_idx] = self.disk_data[block_start_idx:block_end_idx]
            block_start_idx += self.block_size
            block_end_idx += self.block_size
        loaded_data = loaded_data[:end_idx-start_idx]
        return loaded_data

    # def write_to_disk(self, start_idx, end_idx, data_to_write):
    #     n_blocks = int(math.ceil((end_idx - start_idx)))
    #     for _ in range(n_blocks):

    def query(self, item):
        n_items = self.n_items
        start_idx = 0
        array_size = 1
        i = 0
        while i <= (math.ceil(math.log(n_items, 2))):
            n_array_item = self.n_array_items[i]
            end_idx = start_idx + n_array_item
            if n_array_item > 0:
                if start_idx < self.mem_size and end_idx < self.mem_size:
                    search_array = self.arrays[start_idx:start_idx+n_array_item]
                elif start_idx < self.mem_size:  # implies that end_idx > mem_size
                    loaded_data = self.load_from_disk(start_idx=self.mem_size, end_idx=self.mem_size+n_array_item)
                    search_array = self.arrays[start_idx:self.mem_size]
                    search_array = np.array(search_array.tolist() + loaded_data)
                else:  # this means that all was larger than memsize
                    search_array = self.load_from_disk(start_idx=start_idx, end_idx=end_idx)
                idx = bs.search(search_array, item)
                if search_array[idx] == item:
                    return start_idx + idx
            array_size = array_size << 1  # multiply by 2
            start_idx = start_idx + array_size
            i += 1
        return -1  # no found

    def insert(self, item):
        self.n_items += 1
        if self.n_items >= self.n_input_data:
            raise BufferError('too much data, are you sure you initialized me correctly?')

        start_idx = 0
        array_size = 2
        n_insertions = 1
        prev_array = np.array([item])
        for i in range(self.n_arrays):
            current_arr = self.arrays[start_idx:start_idx + array_size]
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
    def __init__(self, disk_filepath, block_size, n_blocks, n_input_data):
        super(Cola, self).__init__(disk_filepath, block_size, n_blocks, n_input_data)
        """" this is a naive implementation of cache-oblivious arrays """
        self.n_levels = math.ceil(math.log(self.size, base=2)) + 1
        self.array = np.empty(2 ** (self.n_levels + 1), dtype=int)
        self.n_array_items = np.zeros(shape=self.n_levels, dtype=int)
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


def dummy_naive_coa():
    ds = NaiveCOA(disk_filepath=os.path.join(storage_dir, 'cola.hdf5'), block_size=50, n_blocks=50, n_input_data=50)
    ds.insert(1)
    ds.insert(2)
    ds.insert(3)
    ds.insert(0)
    print(ds.arrays)
    search_idx = ds.query(0)
    print(search_idx)
    search_idx = ds.query(1)
    print(search_idx)
    search_idx = ds.query(2)
    print(search_idx)
    search_idx = ds.query(3)
    print(search_idx)


def main():
    dummy_naive_coa()


if __name__ == '__main__':
    main()
