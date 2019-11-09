from abc import ABC, abstractmethod
from data_structures.base import WriteOptimizedDS
from cachetools import Cache, LRUCache
from sortedcontainers import SortedSet, SortedList
import numpy as np
import os
import pickle


class BEpsilonTree(WriteOptimizedDS):
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
        def __init__(self, buffer=[], keys=[], children=[]):
            self.buffer = SortedList(buffer)
            self.keys = SortedList(keys)
            self.children = children

        def __contains__(self, item):
            return item in self.buffer

        def add(self, items):
            if not isinstance(items, list):
                items = [items]
            self.buffer.update(items)
            self.buffer = SortedList(SortedSet(self.buffer))

        def get_child(self, item):
            return self.children[self.keys.bisect_right(item)]

        def replace_child(self, child_ptr, left_child_ptr, right_child_ptr, median):
            self.keys.add(median)
            idx = self.children.index(child_ptr)
            self.children[idx:idx+1] = left_child_ptr, right_child_ptr

        def pop_most_pending(self):
            pending_sizes = []
            pending_sizes.append(self.buffer.bisect_left(self.keys[0]))
            for i, j in zip(self.keys, self.keys[1:]):
                size = self.buffer.bisect_right(j) - self.buffer.bisect_left(i)
                pending_sizes.append(size)
            pending_sizes.append(len(self.buffer) - self.buffer.bisect_right(self.keys[-1]))
            idx = np.argmax(pending_sizes)
            child_ptr = self.children[idx]
            if child_ptr == self.children[0]:
                right_idx = self.buffer.bisect_right(self.keys[0])
                items = self.buffer[0:right_idx]
                del self.buffer[0:right_idx]
            elif child_ptr == self.children[-1]:
                left_idx = self.buffer.bisect_left(self.keys[-1])
                items = self.buffer[left_idx:-1]
                del self.buffer[left_idx:-1]
            else:
                left_idx = self.buffer.bisect_left(self.keys[idx-1])
                right_idx = self.buffer.bisect_right(self.keys[idx])
                items = self.buffer[left_idx:right_idx]
                del self.buffer[left_idx:right_idx]

            return (child_ptr, items)

        def split(self):
            assert(len(self.keys) > 1)
            median = self.keys[len(self.keys)//2]
            idx = len(self.keys)//2
            left_node = BEpsilonTree.BranchNode(keys=self.keys[:idx], children=self.children[:idx+1], buffer=self.buffer[:self.buffer.bisect_left(median)])
            right_node = BEpsilonTree.BranchNode(keys=self.keys[idx+1:], children=self.children[idx+1:], buffer=self.buffer[self.buffer.bisect_left(median):])
            return left_node, right_node, median

    class RootNode(BranchNode):
        def pop_most_pending(self):
            if len(self.keys) == 0:
                items = list(self.buffer)
                self.buffer.clear()
                child_ptr = self.children[0]
                return (child_ptr, items)
            else:
                return super().pop_most_pending()
        
        def reset(self, left_ptr, right_ptr, median):
            self.buffer.clear()
            self.keys.clear()
            self.children.clear()
            self.keys.add(median)
            self.children = [left_ptr, right_ptr]

    class LeafNode(Node):
        def __init__(self, records=[]):
            self.records = SortedList(records)

        def __contains__(self, item):
            return item in self.records

        def add(self, items):
            if not isinstance(items, list):
                items = [items]
            self.records.update(items)
            self.records = SortedList(SortedSet(self.records))

        def split(self):
            assert(len(self.records) > 1)
            median = self.records[len(self.records)//2 - 1]
            idx = len(self.records)//2 - 1
            left_node = BEpsilonTree.LeafNode(self.records[:idx])
            right_node = BEpsilonTree.LeafNode(self.records[idx:])
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
        self.blocks = BEpsilonTree.Blocks(n_blocks, disk_filepath)
        self.b = b

        if self.blocks.size < 2:
            root_ptr = self.blocks.create_block()
            leaf_ptr = self.blocks.create_block()
            self.blocks[root_ptr] = BEpsilonTree.RootNode(children=[leaf_ptr])
            self.blocks[leaf_ptr] = BEpsilonTree.LeafNode()

    def insert(self, items, node_ptr=0):
        self.blocks[node_ptr].add(items)
        if self.is_items_full(node_ptr):
            if isinstance(self.blocks[node_ptr], BEpsilonTree.BranchNode):
                # Flush node
                child_ptr, items = self.blocks[node_ptr].pop_most_pending()
                res = self.insert(items, child_ptr)
                if res is not None:
                    left_node, right_node, median = res
                    new_node_ptr = self.blocks.create_block()
                    self.blocks[child_ptr] = left_node
                    self.blocks[new_node_ptr] = right_node
                    self.blocks[node_ptr].replace_child(child_ptr, child_ptr, new_node_ptr, median)
                # If node branch is full, split
                if self.is_branches_full(node_ptr):
                    left_node, right_node, median = self.blocks[node_ptr].split()
                    if isinstance(self.blocks[node_ptr], BEpsilonTree.RootNode):
                        left_node_ptr = self.blocks.create_block()
                        right_node_ptr = self.blocks.create_block()
                        self.blocks[left_node_ptr] = left_node
                        self.blocks[right_node_ptr] = right_node
                        self.blocks[node_ptr].reset(left_node_ptr, right_node_ptr, median)
                    else:
                        return (left_node, right_node, median)
            else:
                left_node, right_node, median = self.blocks[node_ptr].split()
                return (left_node, right_node, median)

    def query(self, item, node_ptr=0):
        # Return item existence in leaf node
        if isinstance(self.blocks[node_ptr], BEpsilonTree.LeafNode):
            return item in self.blocks[node_ptr]
        # Return item existence in buffer of root/branch node
        elif item in self.blocks[node_ptr]:
            return True
        # Query child node
        else:
            return self.query(item, self.blocks[node_ptr].get_child(item))

    def is_items_full(self, node_ptr):
        return len(pickle.dumps(self.blocks[node_ptr])) > self.block_size

    def is_branches_full(self, node_ptr):
        return len(self.blocks[node_ptr].children) >= self.b

    def flush(self):
        self.blocks.flush()