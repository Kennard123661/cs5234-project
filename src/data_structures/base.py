from abc import ABC, abstractmethod


class WriteOptimizedDS(ABC):
    def __init__(self, disk_filepath, block_size, n_blocks):
        assert isinstance(block_size, int) and isinstance(n_blocks, int)
        self.disk_filepath = disk_filepath
        self.block_size = block_size
        self.n_blocks = n_blocks
        self.mem_size = self.n_blocks * self.block_size

    @abstractmethod
    def insert(self, item):
        pass

    @abstractmethod
    def query(self, item):
        pass


if __name__ == "__main__":
    pass
