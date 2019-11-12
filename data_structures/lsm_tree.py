from data_structures.base import WriteOptimizedDS
from sortedcontainers import SortedList
from pybloom_live import ScalableBloomFilter
import pickle
import os
from uuid import uuid4
import copy
import shutil


class LSMTree(WriteOptimizedDS):
    class LevelMetadata():
        def __init__(self):
            self.uuids = list()
            self.first_indices = SortedList()

        def __len__(self):
            return len(self.uuids)

        def insert(self, uuid, first_index):
            idx = self.first_indices.bisect_left(first_index)
            self.first_indices.add(first_index)
            self.uuids.insert(idx, uuid)

        def clear(self, i=None, j=None):
            if i is None and j is None:
                self.uuids.clear()
                self.first_indices.clear()
            else:
                del self.uuids[i:j]
                del self.first_indices[i:j]

        def get_uuid(self, item):
            idx = self.first_indices.bisect_right(item) - 1
            if idx < 0:
                return None
            return self.uuids[idx]

    def __init__(self, disk_filepath, growth_factor=10, enable_bloomfilter=True, bloomfilter_params={'initial_capacity': 3000, 'error_rate': 0.001}, block_size=4096, n_blocks=64, n_input_data=1000):
        super().__init__(disk_filepath, block_size, n_blocks, n_input_data)
        self.growth_factor = growth_factor
        self.memtable = SortedList()
        self.enable_bloomfilter = enable_bloomfilter
        self.bloomfilter_params = bloomfilter_params
        if not os.path.exists(self.disk_filepath):
            os.makedirs(disk_filepath)
        if self.enable_bloomfilter:
            self.bloomfilters = {}
            self.bloomfilters[0] = ScalableBloomFilter(
                **self.bloomfilter_params)

    def insert(self, item):
        if item in self.memtable:
            return
        self.memtable.add(item)
        if self.enable_bloomfilter:
            self.bloomfilters[0].add(item)
        if len(self.memtable) >= self.block_size // 2:
            memtable_copy = list(self.memtable)
            self.dump_to_disk(memtable_copy)
            self.memtable.clear()

    def query(self, item):
        if item not in self.memtable:
            return self.query_from_disk(item)
        else:
            return True

    def dump_to_disk(self, memtable):
        uuid = str(uuid4())
        metadata = LSMTree.LevelMetadata()
        metadata.insert(uuid, memtable[0])
        self.set_level_metadata(0, metadata)
        self.set_level_data(0, uuid, memtable)
        self.compact()

    def get_level_folder(self, level):
        folder = os.path.join(self.disk_filepath, str(level))
        if not os.path.exists(folder):
            os.makedirs(folder)
        return folder

    def is_level_empty(self, level):
        folder = os.path.join(self.disk_filepath, str(level))
        if not os.path.exists(folder):
            os.makedirs(folder)
            return True
        if not os.path.exists(os.path.join(folder, 'metadata')):
            return True
        with open(os.path.join(folder, 'metadata'), 'rb') as f:
            level_meta = pickle.load(f)
        return len(level_meta) == 0

    def get_level_metadata(self, level):
        with open(os.path.join(self.get_level_folder(level), 'metadata'), 'rb') as f:
            meta = pickle.load(f)
        return meta

    def set_level_metadata(self, level, metadata):
        with open(os.path.join(self.get_level_folder(level), 'metadata'), 'wb') as f:
            pickle.dump(metadata, f)

    def get_level_data(self, level, uuid):
        with open(os.path.join(self.get_level_folder(level), uuid), 'rb') as f:
            data = pickle.load(f)
        return data

    def set_level_data(self, level, uuid, data):
        with open(os.path.join(self.get_level_folder(level), uuid), 'wb') as f:
            pickle.dump(data, f)

    def del_level_data(self, level, uuid):
        os.remove(os.path.join(self.get_level_folder(level), uuid))

    def clear_level(self, level):
        folder = self.get_level_folder(level)
        shutil.rmtree(folder)
        self.get_level_folder(level)
        self.set_level_metadata(level, LSMTree.LevelMetadata())

    def compact(self, level=0):
        curr_level_folder = self.get_level_folder(level)
        next_level_folder = os.path.join(self.disk_filepath, str(level + 1))
        curr_level_meta = self.get_level_metadata(level)
        # Current level is not full
        if len(curr_level_meta) < self.growth_factor**level:
            return
        # Current level is full and next level is empty
        if self.is_level_empty(level + 1):
            shutil.rmtree(next_level_folder)
            shutil.copytree(curr_level_folder, next_level_folder)
            shutil.rmtree(curr_level_folder)
            curr_level_folder = self.get_level_folder(level)
            curr_level_meta.clear()
            self.set_level_metadata(level, curr_level_meta)
            if self.enable_bloomfilter:
                self.bloomfilters[level + 1] = self.bloomfilters[level]
                self.bloomfilters[level] = ScalableBloomFilter(
                    **self.bloomfilter_params)
            return
        # Current level is full and next level is not empty
        next_level_meta = self.get_level_metadata(level + 1)
        # Get all data in the current level
        curr_data_list = [self.get_level_data(
            level, uuid) for uuid in curr_level_meta.uuids]
        curr_data = [val for sublist in curr_data_list for val in sublist]
        # Find the indices of the overlapping data in the next level
        next_start_idx = max(next_level_meta.first_indices.bisect_left(curr_data[0]) - 1, 0)
        next_end_idx = next_level_meta.first_indices.bisect_right(curr_data[-1])
        # Get the data in the next level that overlaps this level
        next_data_list = [self.get_level_data(
            level + 1, uuid) for uuid in next_level_meta.uuids[next_start_idx: next_end_idx]]
        next_data = [val for sublist in next_data_list for val in sublist]
        # Delete the data of the next level that was retrieved in the folder and in the metadata
        [self.del_level_data(level + 1, uuid)
            for uuid in next_level_meta.uuids[next_start_idx: next_end_idx]]
        next_level_meta.clear(next_start_idx, next_end_idx)
        # Merge the data in this level and the overlapping data in the next level
        all_sorted_data = []
        i = j = 0
        while i < len(curr_data) and j < len(next_data):
            if curr_data[i] < next_data[j]:
                all_sorted_data.append(curr_data[i])
                i += 1
            else:
                all_sorted_data.append(next_data[j])
                j += 1
        while i < len(curr_data):
            all_sorted_data.append(curr_data[i])
            i += 1
        while j < len(next_data):
            all_sorted_data.append(next_data[j])
            j += 1
        # Break up sorted data into individual files
        for i in range(len(all_sorted_data) // (self.block_size // 2) + 1):
            new_data = all_sorted_data[i*(self.block_size // 2):(i+1)*(self.block_size // 2)]
            if len(new_data) == 0:
                continue
            uuid = str(uuid4())
            first_idx = new_data[0]
            self.set_level_data(level + 1, uuid, new_data)
            next_level_meta.insert(uuid, first_idx)
        # Write next level metadata
        self.set_level_metadata(level + 1, next_level_meta)
        # Clear this level
        self.clear_level(level)
        # Merge bloom filters
        if self.enable_bloomfilter:
            self.bloomfilters[level + 1] = self.bloomfilters[level +
                                                             1].union(self.bloomfilters[level])
            self.bloomfilters[level] = ScalableBloomFilter(
                **self.bloomfilter_params)
        # Compact the next level
        self.compact(level + 1)

    def query_from_disk(self, item):
        total_levels = len(os.listdir(self.disk_filepath))
        for level in range(total_levels):
            if self.enable_bloomfilter and item not in self.bloomfilters[level]:
                continue
            metadata = self.get_level_metadata(level)
            uuid = metadata.get_uuid(item)
            if uuid is None:
                continue
            data = self.get_level_data(level, uuid)
            if item in data:
                return True
        return False
