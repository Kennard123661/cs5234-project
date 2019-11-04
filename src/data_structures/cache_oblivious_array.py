import math
import numpy as np
import h5py
import binary_search as bs
from data_structures.base import WriteOptimizedDS

N_ARRAYS_PER_LEVEL = 2


class CacheObliviousArray(WriteOptimizedDS):
    def __init__(self, disk_filepath, block_size, n_blocks, n_input_data):
        super(CacheObliviousArray, self).__init__(disk_filepath, block_size, n_blocks, n_input_data)

        self.n_levels = math.ceil(math.log(self.mem_size, base=2)) + 1
        self.total_size = (2 ** self.n_levels * N_ARRAYS_PER_LEVEL) + self.block_size
        self.n_items = 0
        self.n_subarray_items = np.zeros(self.n_levels, dtype=int)
        self.cache_array = np.empty(self.mem_size, dtype=int)

        disk = h5py.File(self.disk_filepath, 'w')
        disk.create_dataset('dset', shape=self.total_size)
        disk.close()
        self.disk = h5py.File(self.disk_filepath, 'r+')
        self.disk_data = self.disk['dset']

    def __del__(self):
        self.disk.close()
        del self.disk_data

    def load_data_from_disk(self, start_idx, end_idx):
        """ loads data from the start_idx to the end_idx """
        assert ((start_idx - end_idx) % self.block_size) == 0
        block_start_idx = start_idx
        loaded_data = list()
        while block_start_idx < end_idx:
            loaded_data += self.load_block_from_disk(block_start_idx).tolist()
            block_start_idx += self.block_size
        n_loaded_data = end_idx - start_idx
        return loaded_data[0:n_loaded_data]

    def load_block_from_disk(self, start_idx):
        """ loads a block of data from the disk """
        end_idx = start_idx + self.block_size
        block_data = self.disk_data[start_idx:end_idx]
        return block_data

    def save_data_to_disk(self, start_idx, end_idx, save_data):
        """ save data to the disk from the start_idx to the end_idx"""
        assert len(save_data) == (end_idx - start_idx)
        insert_idx = 0
        block_start_idx = start_idx
        while start_idx < end_idx:
            block_end_idx = min(start_idx + self.block_size, end_idx)
            n_insertions = block_end_idx - block_start_idx
            self.save_block_to_disk(start_idx, end_idx, save_data[insert_idx:insert_idx+n_insertions])
            insert_idx += n_insertions
            block_start_idx = block_end_idx

    def save_block_to_disk(self, start_idx, end_idx, save_data):
        """ saves the block to the disk """
        self.disk_data[start_idx:end_idx] = save_data

    def query(self, item):
        start_idx = 0
        array_size = 1
        i = 0
        while i <= (math.ceil(math.log(self.n_items, 2))):
            n_subarray_item = self.n_subarray_items[i]
            end_idx = start_idx + n_subarray_item
            if n_subarray_item > 0:
                if start_idx < self.mem_size and end_idx < self.mem_size:
                    search_array = self.cache_array[start_idx:start_idx + n_subarray_item]
                elif start_idx < self.mem_size:  # implies that end_idx > mem_size
                    loaded_data = self.load_data_from_disk(start_idx=self.mem_size,
                                                           end_idx=self.mem_size + n_subarray_item)
                    search_array = self.cache_array[start_idx:self.mem_size]
                    search_array = np.array(search_array.tolist() + loaded_data)
                else:  # this means that all was larger than memsize
                    search_array = self.load_data_from_disk(start_idx=start_idx, end_idx=end_idx)
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
            raise BufferError('too much data, no initialization')

        start_idx = 0
        array_size = 0
        n_insertions = 1
        insert_array = [item]

        for i in range(self.n_levels):
            end_idx = start_idx + array_size
            if start_idx < self.mem_size and end_idx < self.mem_size:
                current_arr = self.cache_array[start_idx:start_idx + array_size]
            elif start_idx < self.mem_size:
                current_arr = self.cache_array[start_idx:self.mem_size]
                current_arr = np.array(current_arr.tolist() + self.load_data_from_disk(self.mem_size, end_idx).tolist())
            else:
                current_arr = self.load_data_from_disk(start_idx, end_idx)
            n_subarray_item = self.n_subarray_items[i]

            if (n_insertions + n_subarray_item) > array_size:  # if it will fill up both arrays
                # do a merge here, inserting the largest item first
                insert_idx = n_subarray_item + n_insertions - 1
                i1 = n_subarray_item - 1
                i2 = n_insertions - 1

                while i1 >= 0 and i2 >= 0:
                    if current_arr[i1] > insert_array[i2]:
                        current_arr[insert_idx] = current_arr[i1]
                        i1 -= 1
                    else:
                        current_arr[insert_idx] = insert_array[i2]
                        i2 -= 1
                    insert_idx -= 1

                if i1 >= 0:
                    current_arr[:insert_idx + 1] = current_arr[:i1 + 1]
                    assert i2 < 0
                else:
                    current_arr[:insert_idx + 1] = insert_array[:i2 + 1]
                    assert i1 < 0
                insert_array = current_arr
                n_insertions += n_subarray_item
                self.n_subarray_items[i] = 0
            else:  # simple
                current_arr[n_subarray_item:n_subarray_item+n_insertions] = insert_array[0:n_insertions]
                self.n_subarray_items[i] += n_insertions
                break
            array_size = array_size << 1  # multiply by 2
            start_idx += array_size
