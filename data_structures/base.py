from abc import ABC, abstractmethod


class WriteOptimizedDS(ABC):
    def __init__(self, disk_filepath, block_size, n_blocks, n_input_data):
        """
        :param disk_filepath: the filepath for the disk, it should be a hdf5 file
        :param block_size: the block size of your cache
        :param n_blocks: the number of lines in your cache, the remaining portions must be stored in memory
        :param n_input_data: the number of input data there will be sent in during experimentation.
        """
        assert isinstance(block_size, int) and isinstance(n_blocks, int)
        self.disk_filepath = disk_filepath
        self.block_size = block_size
        self.n_blocks = n_blocks
        self.mem_size = self.n_blocks * self.block_size
        self.n_input_data = n_input_data

    @abstractmethod
    def insert(self, item):
        """ inserts item into the data structure """
        pass

    @abstractmethod
    def query(self, item):
        """ this function should return the pointer/index of the item or -1 if the item does not exist"""
        pass


if __name__ == "__main__":
    pass
