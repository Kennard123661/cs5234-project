import pytest
import shutil
import os
import numpy as np
from data_structures import LSMTree


@pytest.fixture(scope='function')
def test_folder():
    folder = 'blocks'
    yield folder
    if os.path.exists(folder):
        shutil.rmtree(folder)

def test_level_metadata():
    metadata = LSMTree.LevelMetadata()
    metadata.insert(3, 3)
    metadata.insert(1, 1)
    metadata.insert(2, 2)
    metadata.insert(4, 4)
    metadata.insert(5, 5)
    assert len(metadata) == 5
    assert metadata.uuids == [1, 2, 3, 4, 5]
    assert metadata.first_indices == [1, 2, 3, 4, 5]
    metadata.clear(2, 4)
    assert len(metadata) == 3
    assert metadata.uuids == [1, 2, 5]
    assert metadata.first_indices == [1, 2, 5]
    metadata.insert(3, 3)
    assert len(metadata) == 4
    assert metadata.uuids == [1, 2, 3, 5]
    assert metadata.first_indices == [1, 2, 3, 5]
    metadata.clear()
    assert len(metadata) == 0
    assert metadata.uuids == []
    assert metadata.first_indices == []

def test_lsm_tree_mem_inserts(test_folder):
    n = 10
    tree = LSMTree(test_folder)
    for i in range(n):
        tree.insert(i), i
    for i in range(n):
        assert tree.query(i), i

def test_lsm_tree_disk_inserts(test_folder):
    n = 3000
    tree = LSMTree(test_folder)
    for i in range(n):
        tree.insert(i), i
    for i in range(n):
        assert tree.query(i), i

def test_lsm_tree_large_inserts(test_folder):
    n = 30000
    tree = LSMTree(test_folder)
    for i in range(n):
        tree.insert(i), i
    for i in range(n):
        assert tree.query(i), i

def test_lsm_tree_random_inserts(test_folder):
    n = 3000
    arr = np.arange(n)
    np.random.shuffle(arr)
    tree = LSMTree(test_folder)
    for i in arr:
        tree.insert(i), i
    for i in arr:
        assert tree.query(i), i
