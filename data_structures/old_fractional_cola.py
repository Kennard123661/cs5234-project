import os
import math
import h5py
import numpy as np
from bisect import bisect_left, bisect_right
import copy
import binary_search as bs
from data_structures.base import WriteOptimizedDS

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
storage_dir = os.path.join(base_dir, 'storage')

N_ARRAY_PER_LEVEL = 2
REAL_POINTER_STRIDE = 8
VIRTUAL_POINTER_STRIDE = 4
INPUT_MUL = 2

NULL_REF = -1


class FractionalCola(WriteOptimizedDS):
    """ In this implementation, we assume that there are two arrays per level. we did not implement fractional
    cascading for this one. """
    def __init__(self, disk_filepath, block_size, n_blocks, n_input_data, growth_factor=2):
        super(FractionalCola, self).__init__(disk_filepath, block_size, n_blocks, n_input_data)

        # we add more levels to contain the real pointers
        self.growth_factor = int(growth_factor)
        self.n_levels = math.ceil(math.log(self.n_input_data * INPUT_MUL, self.growth_factor))

        # compute the size of eah level
        self.level_sizes = np.zeros(shape=self.n_levels, dtype=int)
        for i in range(self.n_levels):
            array_size = self.growth_factor**i
            self.level_sizes[i] = N_ARRAY_PER_LEVEL * array_size
        self.disk_size = np.sum(self.level_sizes) + self.block_size
        assert self.mem_size < self.disk_size

        # compute the start idxs of each level using a prefix sum.
        self.level_start_idxs = np.zeros(shape=self.n_levels, dtype=int)
        for i in range(1, self.n_levels):
            self.level_start_idxs[i] = self.level_start_idxs[i-1] + self.level_sizes[i-1]

        # compute the number of virtual pointers at each level.
        self.level_n_virtual_pointers = np.zeros(shape=self.n_levels, dtype=int)
        for i in range(self.n_levels):
            array_size = 2**i
            self.level_n_virtual_pointers[i] = array_size // (VIRTUAL_POINTER_STRIDE - 1)
        self.n_virtual_pointers = np.sum(self.level_n_virtual_pointers)

        # compute the start idxs of each virtual point in each level
        self.level_virtual_start = np.zeros(shape=self.n_levels, dtype=int)
        for i in range(1, self.n_levels):
            self.level_virtual_start[i] = self.level_n_virtual_pointers[i - 1] + self.level_virtual_start[i - 1]

        # create storage file.
        if os.path.exists(disk_filepath):
            os.remove(disk_filepath)
        else:
            dirname = os.path.dirname(disk_filepath)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
        disk = h5py.File(self.disk_filepath, 'w')
        disk.create_dataset('dataset', shape=(self.disk_size, ), dtype='i8')
        disk.create_dataset('is_pointer', shape=(self.disk_size, ), dtype='i8')
        disk.create_dataset('real_ref', shape=(self.disk_size, ), dtype='i8')
        disk.create_dataset('virtual_left', shape=(self.n_virtual_pointers, ), dtype='i8')
        disk.create_dataset('virtual_right', shape=(self.n_virtual_pointers, ), dtype='i8')
        disk.close()

        # create reference to disk data, we save the virtual pointers onto disk for ease.
        self.disk = h5py.File(self.disk_filepath, 'r+')
        self.disk_data = self.disk['dataset']
        self.disk_v_lefts = self.disk['virtual_left']
        self.disk_v_rights = self.disk['virtual_right']
        self.disk_is_pointers = self.disk['is_pointer']
        self.disk_r_points = self.disk['real_ref']

        # create cache data.
        self.cache_data = np.zeros(shape=self.mem_size, dtype=np.int).tolist()
        self.cache_is_pointers = list(np.zeros(shape=self.mem_size, dtype=bool))
        self.cache_r_points = list(np.zeros(shape=self.mem_size, dtype=np.int))

        self.n_items = 0
        self.level_n_items = np.zeros(shape=self.n_levels, dtype=np.int)

    def __del__(self):
        """ save the cache data into the disk and close the disk. """
        self.write_disk(self.disk_data, 0, self.mem_size, self.cache_data)
        self.write_disk(self.disk_is_pointers, 0, self.mem_size, self.cache_is_pointers)
        self.write_disk(self.disk_r_points, 0, self.mem_size, self.cache_r_points)
        self.disk.close()
        del self.disk_data

    def insert(self, item):
        self.n_items += 1
        array_size = 1
        n_insertions = 1
        insert_arr = [item]

        # initialize the initial data from the cache
        loaded_arr = copy.deepcopy(self.cache_data)  # ensure no changes to the cache array.
        loaded_is_pointers = copy.deepcopy(self.cache_is_pointers)
        loaded_r_points = copy.deepcopy(self.cache_r_points)

        loaded_arr_start_idx = 0
        loaded_arr_end_idx = self.mem_size

        last_level = 0
        last_arr = list()
        for i in range(self.n_levels):
            start_idx = self.level_start_idxs[i]
            level_size = self.level_sizes[i]
            end_idx = start_idx + level_size

            # load as much as we need based on blocks.
            while loaded_arr_end_idx < end_idx:
                loaded_arr += self.read_disk_block(self.disk_data, loaded_arr_end_idx)
                loaded_is_pointers += self.read_disk_block(self.disk_is_pointers, loaded_arr_end_idx)
                loaded_r_points += self.read_disk_block(self.disk_r_points, loaded_arr_end_idx)
                loaded_arr_end_idx += self.block_size

            # update the loaded array by truncating the parts that are from the previous array
            if start_idx > loaded_arr_start_idx:
                loaded_arr = loaded_arr[start_idx-loaded_arr_start_idx:]
                loaded_is_pointers = loaded_is_pointers[start_idx-loaded_arr_start_idx:]
                loaded_r_points = loaded_r_points[start_idx-loaded_arr_start_idx:]
                loaded_arr_start_idx = start_idx

            # only extract relevant portions of the array to work with
            current_arr = loaded_arr[:end_idx-start_idx]
            curr_is_pointers = loaded_is_pointers[:end_idx-start_idx]
            curr_r_points = loaded_r_points[:end_idx-start_idx]

            # perform the merge into the current array, we merge from the back.
            level_n_items = self.level_n_items[i]
            insert_idx = level_n_items + n_insertions - 1
            i1 = level_n_items - 1
            i2 = n_insertions - 1
            while i1 >= 0 and i2 >= 0:
                if current_arr[i1] > insert_arr[i2]:
                    current_arr[insert_idx] = current_arr[i1]
                    curr_is_pointers[insert_idx] = curr_is_pointers[i1]
                    curr_r_points[insert_idx] = curr_r_points[i1]
                    i1 -= 1
                else:
                    # note that items in the inserted array must not be a pointer.
                    current_arr[insert_idx] = insert_arr[i2]
                    curr_is_pointers[insert_idx] = False
                    curr_r_points[insert_idx] = -1
                    i2 -= 1
                insert_idx -= 1

            # some of the arrays might not be fully merged.
            if i1 >= 0:
                current_arr[:insert_idx + 1] = current_arr[:i1+1]
                curr_is_pointers[:insert_idx+1] = curr_is_pointers[:i1+1]
                curr_r_points[:insert_idx+1] = curr_r_points[:i1+1]
                assert i2 < 0
            else:
                current_arr[:insert_idx + 1] = insert_arr[:i2 + 1]
                curr_is_pointers[:insert_idx+1] = np.zeros(shape=i2, dtype=bool)
                curr_r_points[:insert_idx+1] = -np.ones(shape=i2, dtype=int)
                assert i1 < 0
            # end of merge

            if (n_insertions + level_n_items) > array_size:  # recurse to the next line for insertion
                # extract non-pointers and put them into the insert array
                insert_arr = list()
                for j, is_pointer in enumerate(curr_is_pointers):
                    if not is_pointer:
                        insert_arr.append(current_arr[j])
                n_insertions = len(insert_arr)
                self.level_n_items[i] = 0  # reset this to 0.
            else:  # there is enough space to insert all of the data here.
                # perform save only on relevant portions of the merge
                self.level_n_items[i] += n_insertions
                current_arr = current_arr[:self.level_n_items[i]]
                curr_is_pointers = curr_is_pointers[:self.level_n_items[i]]
                curr_r_points = curr_r_points[:self.level_n_items[i]]

                end_idx = start_idx + self.level_n_items[i]
                self.save_data(curr_is_pointers, curr_r_points, current_arr, end_idx, start_idx)
                last_arr = copy.deepcopy(current_arr)

                # perform an update over the virtual pointers.
                if self.level_n_virtual_pointers[i] > 0:
                    lookahead_pointers_idxs = np.argwhere(curr_is_pointers)
                    self.update_virtual_pointers(i, lookahead_pointers_idxs)
                break

            array_size *= self.growth_factor
            last_level += 1

        # insert real lookahead pointers upwards, we do this for levels before the last level, from bottom up.
        for i in reversed(range(last_level)):
            next_level_n_items = self.level_n_items[i + 1]
            if next_level_n_items < REAL_POINTER_STRIDE:
                break  # we skip if there are no real pointers to insert.

            start_idx = self.level_start_idxs[i]
            insert_r_points = range(start=7, stop=next_level_n_items, step=REAL_POINTER_STRIDE)
            insert_is_pointers = np.ones_like(insert_r_points, dtype=bool)
            n_insertions = len(insert_r_points)
            insert_arr = list()
            for j in insert_r_points:
                insert_arr.append(last_arr[j])

            self.level_n_items[i] = n_insertions
            end_idx = start_idx + n_insertions
            self.save_data(insert_is_pointers, insert_r_points, insert_arr, start_idx, end_idx)
            last_arr = copy.deepcopy(insert_arr)

            # perform an update over the virtual pointers.
            if self.level_n_virtual_pointers[i] > 0:
                lookahead_pointers_idxs = range(n_insertions)
                self.update_virtual_pointers(i, lookahead_pointers_idxs)
            break

    def save_data(self, is_pointer_arr, pointer_ref_arr, data_arr, end_idx, start_idx):
        if start_idx >= self.mem_size and end_idx >= self.mem_size:  # both are larger than cache, copy to disk
            self.write_disk(self.disk_data, start_idx, end_idx, data_arr)
            self.write_disk(self.disk_is_pointers, start_idx, end_idx, is_pointer_arr)
            self.write_disk(self.disk_r_points, start_idx, end_idx, pointer_ref_arr)
        elif start_idx < self.mem_size and end_idx < self.mem_size:  # both are within mem_size, copy to cache
            self.cache_data[start_idx:end_idx] = data_arr
            self.cache_is_pointers[start_idx:end_idx] = is_pointer_arr
            self.cache_r_points[start_idx:end_idx] = pointer_ref_arr
        elif start_idx < self.mem_size:  # copy some to cache and some to the disk.
            disk_start_idx = self.mem_size - start_idx
            self.cache_data[start_idx:self.mem_size] = data_arr[0:disk_start_idx]
            self.cache_is_pointers[start_idx:self.mem_size] = is_pointer_arr[0:disk_start_idx]
            self.cache_r_points[start_idx:self.mem_size] = pointer_ref_arr[0:disk_start_idx]

            # copy the remainder to the disk.
            self.write_disk(self.disk_data, self.mem_size, end_idx, data_arr[disk_start_idx:])
            self.write_disk(self.disk_is_pointers, self.mem_size, end_idx, is_pointer_arr[disk_start_idx:])
            self.write_disk(self.disk_r_points, self.mem_size, end_idx, pointer_ref_arr[disk_start_idx:])

    def update_virtual_pointers(self, i, lookahead_pointers_idxs):
        left_idxs, right_idxs = self.compute_left_right_vp(n_virtual_pointers=self.level_n_virtual_pointers[i],
                                                           real_pointers_idxs=lookahead_pointers_idxs)
        # write the virtual pointers back to the disk
        v_start_idx = self.level_virtual_start[i]
        v_end_idx = v_start_idx + self.level_n_virtual_pointers[i]
        self.write_disk(self.disk_v_lefts, v_start_idx, v_end_idx, left_idxs)
        self.write_disk(self.disk_v_rights, v_start_idx, v_end_idx, right_idxs)

    @staticmethod
    def compute_left_right_vp(n_virtual_pointers, real_pointers_idxs):
        """ compute the virtual pointer left and right real pointer locations using a sliding window """
        left_idxs = -np.ones(n_virtual_pointers)
        right_idxs = -np.ones(n_virtual_pointers)

        # we perform a sliding window search from left to right
        left_bound = 0
        right_bound = 0
        n_lookahead_idxs = len(real_pointers_idxs)

        if n_lookahead_idxs > 0:
            v_idxs = np.arange(start=1, stop=n_virtual_pointers + 1) * 3
            for v, v_idx in enumerate(v_idxs):
                if v_idx > real_pointers_idxs[-1]:  # no real pointers to its right
                    right_idxs[v] = -1
                    # finding closest left
                    while (left_bound + 1) < n_lookahead_idxs and \
                            real_pointers_idxs[left_bound + 1] < v_idx:
                        left_bound += 1
                    left_idxs[v] = left_bound
                elif v_idx < real_pointers_idxs[0]:  # no real pointers to its left
                    left_idxs[v] = -1
                    # finding the closest right that is larger
                    while (right_bound + 1) < n_lookahead_idxs and \
                            real_pointers_idxs[right_bound] < v_idx:
                        right_bound += 1
                    right_idxs[v] = right_bound
                else:  # both are within
                    while (right_bound + 1) < n_lookahead_idxs and \
                            real_pointers_idxs[right_bound] < v_idx:
                        right_bound += 1
                    while (left_bound + 1) < n_lookahead_idxs and \
                            real_pointers_idxs[left_bound + 1] < v_idx:
                        left_bound += 1
                    right_idxs[v] = right_bound
                    left_idxs[v] = left_bound
        return left_idxs, right_idxs

    def query(self, item):
        is_blind_search = True
        search_start_idx = 0
        search_end_idx = 0
        for i in range(self.n_levels):
            level_n_item = self.level_n_items[i]

            has_real_pointer = 2**i > REAL_POINTER_STRIDE

            # check termination condition
            if level_n_item == 0:
                if has_real_pointer:
                    break  # there should be a real pointer here, this means that we have not reached this level yet.
                else:
                    continue  #  before any real pointers are available.

            if is_blind_search:
                search_start_idx = self.level_start_idxs[i]
                search_end_idx = search_start_idx + self.level_n_items[i]

            search_arr = list()
            search_is_pointers = list()
            search_r_points = list()
            if search_start_idx < self.mem_size and search_end_idx < self.mem_size:  # read from cache
                search_arr = self.cache_data[search_start_idx:search_end_idx]
                search_is_pointers = self.cache_is_pointers[search_start_idx:search_end_idx]
                search_r_points = self.cache_r_points[search_start_idx:search_end_idx]
            elif search_start_idx >= self.mem_size and search_end_idx >= self.mem_size:  # read from disk
                search_arr = self.read_disk(self.disk_data, search_start_idx, search_end_idx)
                search_is_pointers = self.read_disk(self.disk_is_pointers, search_start_idx, search_end_idx)
                search_r_points = self.read_disk(self.disk_r_points, search_start_idx, search_end_idx)
            elif search_start_idx < self.mem_size:
                # read some from cache
                search_arr = self.cache_data[search_start_idx:self.mem_size]
                search_is_pointers = self.cache_is_pointers[search_start_idx:self.mem_size]
                search_r_points = self.cache_r_points[search_start_idx:self.mem_size]

                # read the remainder from disk
                search_arr += self.read_disk(self.disk_data, self.mem_size, search_end_idx)
                search_is_pointers += self.read_disk(self.disk_is_pointers, self.mem_size, search_end_idx)
                search_r_points += self.read_disk(self.disk_r_points, self.mem_size, search_end_idx)

            # there is at least one item in this list.
            l, h = my_binary_search(search_arr, item)
            is_found = l == h and l != -1
            next_start_idx = self.level_start_idxs[i+1]
            next_n_items = self.level_n_items[i+1]
            v_start = self.level_virtual_start[i]
            v_next_start = self.level_virtual_start[i+1]
            n_vp = self.level_n_virtual_pointers[i]

            if is_found:
                if search_is_pointers[l]:
                    return search_start_idx + l
                else:
                    is_blind_search = False
                    search_start_idx = next_start_idx + search_r_points[l]
                    search_end_idx = next_start_idx + search_r_points[l] + 1
            else:  # not found, search other locations
                if l == -1:
                    search_start_idx = next_start_idx
                else:
                    # find the closest lookahead pointer that is smaller than l
                    v_left, v_right = self.get_closest_virtual(l)
                    if v_left < 0 and (v_right-1) > n_vp:
                        search_start_idx = next_start_idx
                    elif v_left < 0:
                        proposal = None
                        proposed_left = self.disk_v_lefts[v_start + v_right]
                        if proposed_left == -1

                        proposed_right = self.disk_v_rights[v_start + v_right]
                        if proposed_right < l:

                if h == -1:
                    search_end_idx = next_start_idx + next_n_items
                else:
                    # find the closest lookahead pointer that is smaller than l
                    v_left, v_right = self.get_closest_virtual(l)

            #
            # if is_found:
            #     if search_is_pointers[l]:
            #         return search_start_idx + l
            #     else:
            #         is_blind_search = False
            #         search_start_idx = next_start_idx + search_r_points[l]
            #         search_end_idx = next_start_idx + search_r_points[l] + 1
            # else:
            #     if l == -1:
            #         search_start_idx = next_start_idx
            #         # check if h is a pointer
            #         if search_is_pointers[h]:
            #             is_blind_search = False
            #             search_end_idx = next_start_idx + search_r_points[h] + 1
            #         else:  # find the rightmost pointer
            #             level_virtual_pointers = self.
            #             v_left = h // (VIRTUAL_POINTER_STRIDE - 1)
            #             v_right = v_left + 1
            #             if v_left
            #
            #
            #     elif h == -1:
            #         search_end_idx = next_start_idx + level_n_item
            #
            #
            # if idx < len(search_arr):
            #     if search_arr[idx] == item:  # item found but might be a pointer
            #         if not search_is_pointers[idx]:
            #             return search_start_idx + idx  # found it!
            #         else:  # we traverse down the tree to find the item
            #             is_blind_search = False
            #             search_start_idx = next_start_idx + search_r_points[idx]
            #             search_end_idx = next_start_idx + search_r_points[idx] + 1
            #     else:  # this item is larger than the pointer
            #         if search_is_pointers[i]:
            #             is_blind_search = False
            #             search_end_idx = next_start_idx + search_r_points[idx] + 1  # this will be the upper bound
            #
            #             # find the lower bound
            #             v_id = search_end_idx // (VIRTUAL_POINTER_STRIDE - 1)
            #             if v_id == 0:  # there is no such virtual pointer
            #                 search_start_idx = next_start_idx
            #             else:
            #                 v_idx = v_id - 1
            #                 virtual_start = self.level_virtual_start[i]
            #                 vp_left = self.disk_v_lefts[virtual_start + v_idx]
            #                 vp_right = self.disk_v_lefts[virtual_start + v_idx]
            #
            #                 if vp_left
            #
            #
            #
            #
            #
            # n_items = self.level_n_items[i]
            # if n_items > 0:  # begin the search here
            #     search_arr = search_arr[:n_items]
            #     idx = bs.search(search_arr, item)
            #     if idx < len(search_arr) and search_arr[idx] == item:
            #         search_is_ptrs = search_is_pointers[:n_items]
            #         is_ptr = search_is_ptrs[idx]
            #         if is_ptr:
            #         return next_start_idx + idx
            # next_start_idx += level_size
            # array_size *= self.growth_factor
        return -1

    def get_closest_virtual(self, idx):
        closest_left = idx // (VIRTUAL_POINTER_STRIDE - 1)-1
        closest_right = closest_left + 1
        return closest_left, closest_right

    def read_disk(self, disk, start_idx, end_idx):
        temp_idx = start_idx
        contents = list()
        while temp_idx < end_idx:
            contents += self.read_disk_block(disk, temp_idx)
            temp_idx += end_idx
        return contents[:end_idx-start_idx]

    def write_disk(self, disk, start_idx, end_idx, data):
        """ writes the data from start_idx to end_idx to the disk """
        block_start_idx = start_idx
        data_start_idx = 0
        while block_start_idx < end_idx:
            block_end_idx = min(block_start_idx + self.block_size, end_idx)
            n_writes = block_end_idx - block_start_idx
            data_end_idx = data_start_idx + n_writes
            disk[block_start_idx:block_end_idx] = data[data_start_idx:data_end_idx]
            data_start_idx = data_end_idx
            block_start_idx = block_end_idx

    def read_disk_block(self, disk, block_start_idx):
        """ reads block and returns it as a list """
        return list(disk[block_start_idx:block_start_idx + self.block_size])


def my_binary_search(arr, item):
    """
    :param arr: sorted array in ascending order
    :param item: the item to search for
    :return: the lower bound and upper bound. if they are equal, then the item is found.
    """
    # boundary condition, the item is outside of the array
    last_idx = len(arr) - 1
    if item < arr[0]:
        return -1, 0
    elif item > arr[-1]:
        return last_idx, -1
    elif item == arr[0]:
        return 0, 0
    elif item == last_idx:
        return last_idx, last_idx

    l = 0
    h = len(arr) - 1
    while (l+1) < h:
        mid = (l + h) // 2
        if arr[mid] == item:
            return mid, mid
        elif arr[mid] < item:
            l = mid
        elif arr[mid] > item:
            h = mid
    return l, h



def main():
    save_filename = 'cola.hdf5'
    ds = FractionalCola(disk_filepath=os.path.join(storage_dir, save_filename), block_size=2,
                        n_blocks=2, n_input_data=5000)
    ds.insert(1)
    ds.insert(2)
    ds.insert(3)
    ds.insert(0)
    print(len(ds.cache_data))
    print(ds.cache_data)
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
