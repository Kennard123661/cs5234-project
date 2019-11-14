from abc import ABC, abstractmethod
from data_structures.base import WriteOptimizedDS
from cachetools import Cache, LRUCache
from sortedcontainers import SortedSet, SortedList
import numpy as np
import os
import pickle


class BTree(WriteOptimizedDS):
    class Node(ABC):
        def __init__(self):
            pass

        @abstractmethod
        def add(self, item):
            pass

        @abstractmethod
        def split(self):
            pass

    class BranchNode(Node):
        def __init__(self, tree, keys=[], children=[]):
            self.tree = tree
            self.keys = SortedList(keys)
            self.children = children

        def __len__(self):
            return len(self.children)

        def __contains__(self, item):
            next_child = self.children[self.keys.bisect_right(item)]
            if isinstance(next_child, BTree.BranchNode):
                return item in next_child
            else:
                return item in self.tree.blocks[next_child]

        def add(self, item):
            if item in self.keys:
                 return
            idx = self.keys.bisect_left(item)
            next_child = self.children[idx]
            if isinstance(next_child, BTree.BranchNode):
                if len(next_child) >= self.tree.b * 2:
                    left_node, right_node, median = next_child.split()
                    self.keys.add(median)
                    self.children[idx:idx+1] = left_node, right_node
                    next_child = self.children[idx] if item < median else self.children[idx + 1]
                next_child.add(item)
            else:
                leaf_node = self.tree.blocks[next_child]
                if len(leaf_node) >= (self.tree.block_size // 2): 
                    left_node, right_node, median = leaf_node.split()
                    new_block = self.tree.blocks.create_block()
                    self.tree.blocks[next_child] = left_node
                    self.tree.blocks[new_block] = right_node
                    self.keys.add(median)
                    self.children[idx:idx+1] = next_child, new_block
                    next_child = next_child if item < median else new_block
                self.tree.blocks[next_child].add(item)

        def split(self):
            assert(len(self.keys) > 1)
            median = self.keys[len(self.keys)//2]
            idx = len(self.keys)//2
            left_node = BTree.BranchNode(self.tree, keys=self.keys[:idx], children=self.children[:idx+1])
            right_node = BTree.BranchNode(self.tree, keys=self.keys[idx+1:], children=self.children[idx+1:])
            return left_node, right_node, median

    class LeafNode(Node):
        def __init__(self, records=[]):
            self.records = SortedList(records)

        def __contains__(self, item):
            return item in self.records

        def __len__(self):
            return len(self.records)

        def add(self, items):
            if not isinstance(items, list):
                items = [items]
            self.records.update(items)
            self.records = SortedList(SortedSet(self.records))

        def split(self):
            assert(len(self.records) > 1)
            median = self.records[len(self.records)//2 - 1]
            idx = len(self.records)//2 - 1
            left_node = BTree.LeafNode(self.records[:idx])
            right_node = BTree.LeafNode(self.records[idx:])
            return left_node, right_node, median

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
            for key, value in self._Cache__data.items(): # pylint: disable=maybe-no-member
                with open(os.path.join(self.folder, str(key)), 'wb') as f:
                    pickle.dump(value, f)

    def __init__(self, disk_filepath, b=4, block_size=4096, n_blocks=64, n_input_data=1024):
        super().__init__(disk_filepath, block_size, n_blocks, n_input_data)
        self.blocks = BTree.Blocks(n_blocks, disk_filepath)
        self.b = b
        new_block = self.blocks.create_block()
        self.blocks[new_block] = BTree.LeafNode()
        self.root = BTree.BranchNode(self, keys=[], children=[new_block])

    def insert(self, items):
        if len(self.root) >= self.b * 2:
            left_node, right_node, median = self.root.split()
            self.root = BTree.BranchNode(self, [median], [left_node, right_node])
        self.root.add(items)

    def query(self, item):
        return item in self.root