from data_structures.base import WriteOptimizedDS
import numpy as np
import math
import os
import h5py


base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
storage_dir = os.path.join(base_dir, 'storage')

INVALID_IDX = -1


class FractCola(WriteOptimizedDS):
    def __init__(self, disk_filepath, block_size, n_blocks, n_input_data, growth_factor=2, pointer_density=0.1):
        super(FractCola, self).__init__(disk_filepath, block_size, n_blocks, n_input_data)

        self.g = int(growth_factor)
        self.p = float(pointer_density)

        # compute the number of levels needed to store all input data
        self.n_levels = 1
        n_elements = 1
        while n_elements < self.n_input_data:
            level_size = 2 * (self.g - 1) * self.g**(self.n_levels - 1)
            level_n_lookahead = int(math.floor(2 * self.p * (self.g - 1) * self.g ** (self.n_levels - 1)))
            n_elements += (level_size - level_n_lookahead)
            self.n_levels += 1

        # compute the number of lookahead pointers
        self.level_sizes = [1] + [(2 * (self.g - 1) * self.g**(i - 1)) for i in range(1, self.n_levels)]
        self.level_n_lookaheads = [0] + [int(math.floor(2 * self.p * (self.g - 1) * self.g ** (i - 1)))
                                         for i in range(1, self.n_levels)]

        self.level_n_items = np.zeros(self.n_levels, dtype=int)
        self.disk_size = np.sum(self.level_sizes)

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
        disk.create_dataset('dataset', shape=(self.disk_size,), dtype=np.int)
        disk.create_dataset('is_lookaheads', shape=(self.disk_size,), dtype=np.bool)
        disk.create_dataset('references', shape=(self.disk_size,), dtype=np.int)
        disk.close()

        self.disk = h5py.File(self.disk_filepath, 'r+')
        self.data = self.disk['dataset']
        self.is_lookaheads = self.disk['is_lookaheads']
        self.references = self.disk['references']
        self.n_items = 0
        self.final_insert_level = 0

    def insert(self, item):
        insert_data = [item]
        self.n_items += 1
        n_inserts = 1
        next_level_data = None

        # perform the downward merge
        last_insert_level = 0
        for i in range(self.n_levels):
            level_start_idx = self.level_start_idxs[i]
            level_n_items = self.level_n_items[i]
            level_size = self.level_sizes[i]
            level_end_idx = level_start_idx + level_n_items

            level_data = self.data[level_start_idx:level_end_idx]
            level_is_lookaheads = self.is_lookaheads[level_start_idx:level_end_idx]
            level_references = self.references[level_start_idx:level_end_idx]

            merge_size = n_inserts + level_n_items
            merged_data = np.zeros(shape=merge_size, dtype=int)
            merged_is_lookaheads = np.zeros(shape=merge_size, dtype=bool)
            merged_references = np.zeros(shape=merge_size, dtype=int)

            # perform the merge here, we merge to the front of the merge array.
            merged_i, insert_i, level_i = 0, 0, 0
            leftmost_lookahead_idx = INVALID_IDX
            while level_i < level_n_items and insert_i < n_inserts:
                if level_data[level_i] <= insert_data[insert_i]:  # insert level items
                    merged_data[merged_i] = level_data[level_i]
                    merged_is_lookaheads[merged_i] = level_is_lookaheads[level_i]
                    if merged_is_lookaheads[merged_i]:  # if is lookahead pointer, then
                        merged_references[merged_i] = level_references[level_i]
                        leftmost_lookahead_idx = merged_i
                    else:  # not lookahead, so point to the nearest lookahead.
                        merged_references[merged_i] = leftmost_lookahead_idx
                    level_i += 1
                else:
                    merged_data[merged_i] = insert_data[level_i]
                    merged_is_lookaheads[merged_i] = False
                    merged_references[merged_i] = leftmost_lookahead_idx
                    insert_i += 1
                merged_i += 1

            if insert_i < n_inserts:
                assert level_i == level_n_items
                merged_data[merged_i:] = insert_data[insert_i:]
                merged_is_lookaheads[merged_i:] = np.zeros_like(insert_data[insert_i:], dtype=bool)
                merged_references[merged_i:] = np.ones_like(insert_data[insert_i:], dtype=int) * leftmost_lookahead_idx
            elif level_i < level_n_items:
                assert insert_i == n_inserts
                merged_data[merged_i:] = level_data[level_i:]
                merged_is_lookaheads[merged_i:] = level_is_lookaheads[level_i:]
                for j, is_lookahead in enumerate(level_is_lookaheads[level_i:]):
                    if is_lookahead:
                        merged_references[merged_i+j] = level_references[level_i+j]
                        leftmost_lookahead_idx = level_i + j
                    else:
                        merged_references[merged_i+j] = leftmost_lookahead_idx

            if level_n_items + n_inserts > level_size:  # it will be full, grab all non-pointers
                self.level_n_items[i] = 0
                data_idxs = np.argwhere(np.bitwise_not(merged_is_lookaheads)).reshape(-1)
                insert_data = merged_data[data_idxs]
                n_inserts = len(insert_data)
            else:
                self.level_n_items[i] = merge_size
                level_end_idx = level_start_idx + merge_size

                # perfrom writes here.
                self.data[level_start_idx:level_end_idx] = merged_data
                self.is_lookaheads[level_start_idx:level_end_idx] = merged_is_lookaheads
                self.references[level_start_idx:level_end_idx] = merged_references

                # update for searches
                self.final_insert_level = max(self.final_insert_level, i)

                # update for the upward insertion of lookahead pointers
                last_insert_level = i
                next_level_data = merged_data
                break

        # perform the upward insertion of lookahead pointers, note that all upper levels were merged
        # and should not have any items, so we can simply override them.
        for i in reversed(range(last_insert_level)):
            level_n_lookahead = self.level_n_lookaheads[i]
            if level_n_lookahead == 0:
                break  # no more lookaheads

            next_level_size = self.level_sizes[i+1]
            next_level_n_items = self.level_n_items[i+1]
            assert len(next_level_data) == next_level_n_items

            lookahead_stride = next_level_size // level_n_lookahead
            lookahead_references = [ref for ref in range(lookahead_stride-1, next_level_n_items, lookahead_stride)]
            n_lookahead = len(lookahead_references)
            if n_lookahead == 0:
                break  # no more lookahead pointers to insert.
            lookahead_data = next_level_data[lookahead_references]

            # update n_items
            self.level_n_items[i] = n_lookahead
            level_start_idx = self.level_start_idxs[i]
            level_end_idx = level_start_idx + n_lookahead

            # write to disk
            self.data[level_start_idx:level_end_idx] = lookahead_data
            self.is_lookaheads[level_start_idx:level_end_idx] = np.ones(shape=n_lookahead, dtype=bool)
            self.references[level_start_idx:level_end_idx] = lookahead_references

            # update for next iteration
            next_level_data = lookahead_data

    def query(self, item):
        idx = self._search(item)
        return idx

    def _search(self, item):
        n_search_levels = self.final_insert_level + 1
        search_start = INVALID_IDX
        search_end = INVALID_IDX

        for i in range(n_search_levels):
            if search_start == INVALID_IDX:
                search_start = 0

            level_n_item = self.level_n_items[i]
            if search_end == INVALID_IDX:
                search_end = level_n_item

            assert search_start <= search_end
            if search_end - search_start == 0:
                search_start = INVALID_IDX
                search_end = INVALID_IDX
                continue

            level_start_idx = self.level_start_idxs[i]
            start_idx = level_start_idx + search_start
            end_idx = level_start_idx + search_end
            search_arr = self.data[start_idx:end_idx]

            l, r = self.binary_search(search_arr, item)
            is_found = (l == r) and (l != INVALID_IDX)
            if is_found:
                loc = start_idx + l
                is_lookahead = self.is_lookaheads[loc]
                if is_lookahead:
                    reference = self.references[loc]
                    search_start = reference
                    search_end = reference+1
                else:
                    return loc
            else:
                if l == INVALID_IDX:
                    search_start = INVALID_IDX
                else:
                    loc = start_idx + l
                    is_lookahead = self.is_lookaheads[loc]
                    reference = self.references[loc]
                    if is_lookahead:
                        search_start = reference
                    else:
                        if reference == INVALID_IDX:
                            search_start = INVALID_IDX
                        else:
                            loc = level_start_idx + reference
                            search_start = self.references[loc]

                if r == INVALID_IDX:
                    search_end = INVALID_IDX
                else:
                    loc = start_idx + r
                    is_lookahead = self.is_lookaheads[loc]
                    reference = self.references[loc]
                    if is_lookahead:
                        search_end = reference
                    else:
                        search_end = INVALID_IDX
                        is_lookaheads = self.is_lookaheads[level_start_idx+r+1:level_start_idx+level_n_item]
                        for j, is_lookahead in enumerate(is_lookaheads):
                            if is_lookahead:
                                reference = self.references[level_start_idx+r+1+j]
                                search_end = reference
        return -1

    @staticmethod
    def binary_search(search_arr, item):
        # boundary conditions
        search_arr = np.array(search_arr, dtype=int)
        last_idx = len(search_arr) - 1
        if item == search_arr[0]:  # if item is found at the startign idx
            return 0, 0

        if item == search_arr[-1]:  # if item is found at the last idx
            return last_idx, last_idx

        if item > search_arr[-1]:  # if item is bigger than all items
            return last_idx, INVALID_IDX

        if item < search_arr[0]:  # if item is smaller than all items
            return INVALID_IDX, 0

        l = 0
        h = last_idx
        while (l + 1) < h:  # terminate when l + 1 = h
            mid = (l + h) // 2
            if item == search_arr[mid]:
                return mid, mid
            elif item < search_arr[mid]:
                h = mid
            else:  # item > search_arr[mid]
                l = search_arr[mid]
        return l, h


def main():
    save_filename = 'cola.hdf5'
    ds = FractCola(disk_filepath=os.path.join(storage_dir, save_filename), block_size=2,
                   n_blocks=2, n_input_data=100)

    for i in range(100):
        ds.insert(i)

    for i in range(100):
        print(ds.query(i) > -1)


if __name__ == '__main__':
    main()
