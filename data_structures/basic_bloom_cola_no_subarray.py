from data_structures.base import WriteOptimizedDS
import numpy as np
import math
import os
import binary_search as bs
import h5py
import copy
from pybloom_live import BloomFilter

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
storage_dir = os.path.join(base_dir, 'storage')

INVAlID_SEARCH_IDX = -1

ERROR_RATE = 1e-3


class BasicBloomCola(WriteOptimizedDS):
    """ this augments the basic cola data structure with bloom filters at each subarray level and a larger bloom
    filter that checks for existence across all levels"""
    def __init__(self, disk_filepath, block_size, n_blocks, n_input_data, growth_factor=2, pointer_density=0.1):
        super(BasicBloomCola, self).__init__(disk_filepath, block_size, n_blocks, n_input_data)

        self.g = int(growth_factor)
        self.bloom_filter = BloomFilter(capacity=self.n_input_data, error_rate=ERROR_RATE)

        # compute the number of levels needed to store all input data
        self.n_levels = math.ceil(math.log(self.n_input_data, self.g)) + 1
        self.level_sizes = np.array([self.g**i for i in range(self.n_levels)], dtype=int)
        self.level_n_items = np.zeros(self.n_levels, dtype=int)
        self.disk_size = np.sum(self.level_sizes) + self.block_size

        self.level_start_idxs = np.zeros(self.n_levels, dtype=int)
        for i in range(1, self.n_levels):  # preform prefix sum to get start idxs for the level
            self.level_start_idxs[i] = self.level_start_idxs[i - 1] + self.level_sizes[i - 1]

        # create storage file.
        if os.path.exists(disk_filepath):
            os.remove(disk_filepath)
        else:
            dirname = os.path.dirname(disk_filepath)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
        disk = h5py.File(self.disk_filepath, 'w')
        disk.create_dataset('dataset', shape=(self.disk_size,), dtype=int)
        disk.close()

        self.disk = h5py.File(self.disk_filepath, 'r+')
        self.data = self.disk['dataset']

        self.n_items = 0
        self.final_insert_level = 0

    def insert(self, item):
        insert_data = [item]
        self.n_items += 1
        n_inserts = 1
        self.bloom_filter.add(item)

        # perform the downward merge
        for i in range(self.n_levels):
            level_start_idx = self.level_start_idxs[i]
            level_n_items = self.level_n_items[i]
            level_size = self.level_sizes[i]
            level_end_idx = level_start_idx + level_n_items

            level_data = self.data[level_start_idx:level_end_idx]
            merge_size = n_inserts + level_n_items
            merged_data = np.zeros(shape=merge_size, dtype=int)

            # perform the merge here.
            merged_i, insert_i, level_i = 0, 0, 0
            while level_i < level_n_items and insert_i < n_inserts:
                if level_data[level_i] <= insert_data[insert_i]:  # insert level items
                    merged_data[merged_i] = level_data[level_i]
                    level_i += 1
                else:
                    merged_data[merged_i] = insert_data[insert_i]
                    insert_i += 1
                merged_i += 1

            if insert_i < n_inserts:
                assert level_i == level_n_items
                merged_data[merged_i:] = insert_data[insert_i:]
            elif level_i < level_n_items:
                merged_data[merged_i:] = level_data[level_i:]

            if merge_size > level_size:  # it will be full
                self.level_n_items[i] = 0
                insert_data = copy.deepcopy(merged_data)
                n_inserts = len(insert_data)
            else:
                self.level_n_items[i] = merge_size
                level_end_idx = level_start_idx + merge_size
                self.data[level_start_idx:level_end_idx] = merged_data

                # update for queries
                self.final_insert_level = max(self.final_insert_level, i)
                break

    def query(self, item):
        idx = self._search(item)
        return idx > INVAlID_SEARCH_IDX

    def _search(self, item):
        if item not in self.bloom_filter:  # check bloom filter first.
            return INVAlID_SEARCH_IDX
        n_search_levels = self.final_insert_level + 1

        for i in range(n_search_levels):
            level_n_item = self.level_n_items[i]
            if level_n_item == 0:
                continue  # no items to search

            level_start_idx = self.level_start_idxs[i]
            level_end_idx = level_start_idx + level_n_item
            search_data = self.data[level_start_idx:level_end_idx]
            idx = bs.search(search_data, item)
            if idx < len(search_data) and search_data[idx] == item:
                return level_start_idx + idx
        return INVAlID_SEARCH_IDX


def main():
    save_filename = 'cola.hdf5'
    ds = BasicBloomCola(disk_filepath=os.path.join(storage_dir, save_filename), block_size=2,
                        n_blocks=2, n_input_data=4)
    ds.insert(1)
    ds.insert(2)
    ds.insert(3)
    ds.insert(0)
    print(ds.level_n_items)
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
