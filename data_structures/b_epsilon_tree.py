from data_structures.base import WriteOptimizedDS
from cachetools import Cache, LRUCache
from sortedcontainers import SortedSet
import numpy as np
import os
import pickle
import enum


class BEpsilonTree(WriteOptimizedDS):
    class Node():
        def __init__(self):
            pass

    class BranchNode(Node):
        def __init__(self, b):
            self.b = b
            self.buffer = SortedSet()
            self.keys = np.empty(b - 1)
            self.pivots = np.empty(b)

    class RootNode(BranchNode):
        def __init__(self, b):
            pass

    class LeafNode(Node):
        def __init__(self):
            self.keys = SortedSet()

    class Blocks(LRUCache):
        def __init__(self, maxsize, folder):
            LRUCache.__init__(self, maxsize)
            if not os.path.exists(folder):
                os.makedirs(folder)
            self.folder = folder
            self.size = len(os.listdir(folder))

        def __getitem__(self, key):
            assert(isinstance(key, int))
            if not self.__contains__(key):
                with open(os.path.join(self.folder, str(key)), 'rb') as f:
                    value = pickle.load(f)
                self.__setitem__(key, value)
            return LRUCache.__getitem__(self, key)

        def popitem(self):
            key, value = LRUCache.popitem(self)
            with open(os.path.join(self.folder, str(key)), 'wb') as f:
                pickle.dump(value, f)

        def create_block(self):
            key = self.size
            self.size += 1
            self.__setitem__(key, None)
            return key

        def flush(self):
            for key, value in self._Cache__data.items():
                with open(os.path.join(self.folder, str(key)), 'wb') as f:
                    pickle.dump(value, f)

    def __init__(self, disk_filepath, b=4, block_size=4096, n_blocks=64, n_input_data=1024):
        super().__init__(self, disk_filepath, block_size, n_blocks, n_input_data)
        self.blocks = BEpsilonTree.Blocks(n_blocks, disk_filepath)
        self.b = b

    def insert(self, item, node_ptr=0):
        self.blocks[node_ptr].insert(item)
        if len(pickle.dumps(self.cache[node_ptr])) > self.block_size:
            self.flush(node_ptr)

    def flush(self, node_ptr):
        while len(pickle.dumps(self.blocks[node_ptr])) > self.block_size:
            items, child = self.blocks[node_ptr].flush()
            self.insert(item, child)

    def query(self, item, node_ptr=0):
        pass
