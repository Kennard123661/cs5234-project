import os
import math
import h5py
import numpy as np
from data_structures.base import WriteOptimizedDS


class COLA(WriteOptimizedDS):
    """
    In this implementation, we assume that there are two arrays per level.
    """
    def __init__(self, disk_filepath, block_size, n_blocks, n_input_data, growth_factor=2):
        super(COLA, self).__init__(disk_filepath, block_size, n_blocks, n_input_data)

        self.growth_factor = int(growth_factor)
        self.duplicate_factor = 4 * self.growth_factor
        self.n_levels = math.ceil(math.log(self.n_input_data, self.growth_factor))

        self.disk_size = self.block_size  # reading and writing will be in blocks.
        for i in range(self.n_levels):
            array_size = self.growth_factor**i
            self.disk_size += (array_size << 1)

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
        self.merge1_idxs = np.zeros(shape=self.n_levels, dtype=np.int)  # exclusive end of the first array
        self.merge2_idxs = np.zeros(shape=self.n_levels, dtype=np.int)  # exclusive end of the second array
        self.cache_array = np.zeros(self.mem_size, dtype=np.int)

    def __del__(self):
        raise NotImplementedError  # todo add the logic for copying from cache back into memory i.e shutting down.

    def insert(self, item):
        array_size = 1
        n_insertions = 1
        insert_arr = [item]

        start_idx = 0
        loaded_arr = self.cache_array
        loaded_arr_start_idx = 0
        loaded_arr_end_idx = self.mem_size
        for i in range(self.n_levels):
            level_size = array_size << 1

            # update the loaded array
            if start_idx > loaded_arr_start_idx:
                loaded_arr = loaded_arr[start_idx-loaded_arr_start_idx:]
                loaded_arr_start_idx = start_idx

            end_idx = start_idx + level_size
            while loaded_arr_end_idx < end_idx:
                loaded_arr += self.read_block(loaded_arr_end_idx)
                loaded_arr_end_idx += self.block_size

            n_level_items = self.n_level_items[i]
            current_arr = loaded_arr[:end_idx-start_idx]
            if n_insertions + self.n_level_items > array_size:  # this menas that we need to merge
                insert_arr = current

            start_idx += N_ARRAYS_PER_LEVEL * array_size
            array_size *= self.growth_factor
            raise NotImplementedError



    def query(self, item):
        raise NotImplementedError

    def read_block(self, block_start_idx):
        """ reads block and returns it as a list """
        return list(self.data[block_start_idx:block_start_idx+self.block_size])


