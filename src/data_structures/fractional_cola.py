import os
import math
import h5py
import numpy as np
import copy
import binary_search as bs
from data_structures.base import WriteOptimizedDS

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
storage_dir = os.path.join(base_dir, 'storage')


class FractionalCola(WriteOptimizedDS):
    """ In this implementation, we assume that there are two arrays per level. we did not implement fractional
    cascading for this one. """
    def __init__(self, disk_filepath, block_size, n_blocks, n_input_data, growth_factor=2):
        super(FractionalCola, self).__init__(disk_filepath, block_size, n_blocks, n_input_data)

        self.growth_factor = int(growth_factor)
        self.duplicate_factor = 4 * self.growth_factor
        self.n_levels = math.ceil(math.log(self.n_input_data, self.growth_factor))

        self.disk_size = self.block_size  # reading and writing will be in blocks.
        for i in range(self.n_levels):
            array_size = self.growth_factor**i
            self.disk_size += (array_size << 1)
        assert self.mem_size < self.disk_size

        # todo: transfer this over to the base.py
        if os.path.exists(disk_filepath):
            os.remove(disk_filepath)
        else:
            dirname = os.path.dirname(disk_filepath)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

        disk = h5py.File(self.disk_filepath, 'w')
        disk.create_dataset('dataset', shape=(self.disk_size, ), dtype='i8')
        disk.close()
        self.disk = h5py.File(self.disk_filepath, 'r+')
        self.data = self.disk['dataset']

        self.n_level_items = np.zeros(shape=self.n_levels, dtype=np.int)
        self.cache_array = np.zeros(shape=self.mem_size, dtype=np.int).tolist()
        self.n_items = 0

    def __del__(self):
        """ save the cache data into the disk and close the disk. """
        self.write_disk(0, self.mem_size, self.cache_array)
        self.disk.close()
        del self.data

    def insert(self, item):
        self.n_items += 1
        array_size = 1
        n_insertions = 1
        insert_arr = [item]

        start_idx = 0
        loaded_arr = copy.deepcopy(self.cache_array)  # ensure no changes to the cache array.
        loaded_arr_start_idx = 0
        loaded_arr_end_idx = self.mem_size
        for i in range(self.n_levels):
            level_size = array_size << 1
            end_idx = start_idx + level_size

            # load as much as we need based on blocks.
            while loaded_arr_end_idx < end_idx:
                loaded_arr += self.read_disk_block(loaded_arr_end_idx)
                loaded_arr_end_idx += self.block_size

            # update the loaded array
            if start_idx > loaded_arr_start_idx:
                loaded_arr = loaded_arr[start_idx-loaded_arr_start_idx:]
                loaded_arr_start_idx = start_idx

            n_level_items = self.n_level_items[i]
            current_arr = loaded_arr[:end_idx-start_idx]

            # perform the merge into the current array, we merge from the back.
            insert_idx = n_level_items + n_insertions - 1
            i1 = n_level_items - 1
            i2 = n_insertions - 1
            while i1 >= 0 and i2 >= 0:
                if current_arr[i1] > insert_arr[i2]:
                    current_arr[insert_idx] = current_arr[i1]
                    i1 -= 1
                else:
                    current_arr[insert_idx] = insert_arr[i2]
                    i2 -= 1
                insert_idx -= 1

            if i1 >= 0:
                current_arr[:insert_idx + 1] = current_arr[:i1 + 1]
                assert i2 < 0
            else:
                current_arr[:insert_idx + 1] = insert_arr[:i2 + 1]
                assert i1 < 0
            # end of merge

            if (n_insertions + n_level_items) > array_size:  # recurse to the next line for insertion
                insert_arr = current_arr
                n_insertions += n_level_items
                self.n_level_items[i] = 0
            else:  # there is enough space to insert all of the data here.
                self.n_level_items[i] += n_insertions
                if start_idx >= self.mem_size and end_idx >= self.mem_size:  # both are larger, copy to disk
                    self.write_disk(start_idx, end_idx, current_arr)
                elif start_idx < self.mem_size and end_idx < self.mem_size:  # both are smaller, copy to cache
                    self.cache_array[start_idx:end_idx] = current_arr
                elif start_idx < self.mem_size:  # copy some to cache and some to the disk.
                    self.cache_array[start_idx:self.mem_size] = current_arr[0:self.mem_size-start_idx]
                    self.write_disk(self.mem_size, end_idx, current_arr[self.mem_size-start_idx:])
                break

            start_idx += level_size
            array_size *= self.growth_factor

    def query(self, item):
        array_size = 1
        loaded_arr = copy.deepcopy(self.cache_array)
        loaded_arr_start_idx = 0
        loaded_arr_end_idx = self.mem_size

        start_idx = 0
        n_search_levels = math.ceil(math.log(self.n_items, self.growth_factor)) + 1
        for i in range(n_search_levels):
            level_size = array_size << 1

            # update the loaded array
            end_idx = start_idx + level_size
            while loaded_arr_end_idx < end_idx:
                loaded_arr += self.read_disk_block(loaded_arr_end_idx)
                loaded_arr_end_idx += self.block_size

            if start_idx > loaded_arr_start_idx:
                loaded_arr = loaded_arr[start_idx - loaded_arr_start_idx:]
                loaded_arr_start_idx = start_idx

            n_arr_items = self.n_level_items[i]
            if n_arr_items > 0:  # begin the search here
                search_arr = loaded_arr[:n_arr_items]
                idx = bs.search(search_arr, item)
                if idx < len(search_arr) and search_arr[idx] == item:
                    return start_idx + idx
            start_idx += level_size
            array_size *= self.growth_factor
        return -1

    def write_disk(self, start_idx, end_idx, data):
        """ writes the data from start_idx to end_idx to the disk """
        block_start_idx = start_idx
        data_start_idx = 0
        while block_start_idx < end_idx:
            block_end_idx = min(block_start_idx + self.block_size, end_idx)
            n_writes = block_end_idx - block_start_idx
            data_end_idx = data_start_idx + n_writes
            self.data[block_start_idx:block_end_idx] = data[data_start_idx:data_end_idx]
            data_start_idx = data_end_idx
            block_start_idx = block_end_idx

    def read_disk_block(self, block_start_idx):
        """ reads block and returns it as a list """
        return list(self.data[block_start_idx:block_start_idx+self.block_size])


def main():
    save_filename = 'cola.hdf5'
    ds = FractionalCola(disk_filepath=os.path.join(storage_dir, save_filename), block_size=2,
                        n_blocks=2, n_input_data=5000)
    ds.insert(1)
    ds.insert(2)
    ds.insert(3)
    ds.insert(0)
    print(len(ds.cache_array))
    print(ds.cache_array)
    print(ds.n_level_items)
    search_idx = ds.query(0)
    print(search_idx)
    search_idx = ds.query(1)
    print(search_idx)
    search_idx = ds.query(2)
    print(search_idx)
    search_idx = ds.query(3)
    print(search_idx)


if __name__ == '__main__':
    main()
