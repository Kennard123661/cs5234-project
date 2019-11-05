import pytest
import shutil
import os
from data_structures import BEpsilonTree


@pytest.fixture(scope='function')
def test_folder():
    folder = 'blocks'
    yield folder
    if os.path.exists(folder):
        shutil.rmtree(folder)


def test_blocks_init(test_folder):
    folder = test_folder
    assert BEpsilonTree.Blocks(10, folder) is not None


def test_blocks_create_block(test_folder):
    blocks = BEpsilonTree.Blocks(10, test_folder)
    key = blocks.create_block()
    assert blocks[key] == None
    assert blocks.currsize == 1


def test_blocks_evict(test_folder):
    blocks = BEpsilonTree.Blocks(1, test_folder)
    key0 = blocks.create_block()
    blocks[key0] = "Test"
    key1 = blocks.create_block()

    assert blocks.currsize == 1
    assert os.path.exists(os.path.join(test_folder, str(key0)))
    val0 = blocks[key0]
    assert val0 == "Test"
    assert os.path.exists(os.path.join(test_folder, str(key1)))


def test_blocks_flush(test_folder):
    blocks = BEpsilonTree.Blocks(10, test_folder)
    for i in range(5):
        blocks[i] = {}
    blocks.flush()
    for i in range(5):
        assert os.path.exists(os.path.join(test_folder, str(i)))
