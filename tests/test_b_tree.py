import pytest
import shutil
import os
import numpy as np
from data_structures import BTree


@pytest.fixture(scope='function')
def test_folder():
    folder = 'blocks'
    os.mkdir(folder)
    yield folder
    if os.path.exists(folder):
        shutil.rmtree(folder)

def test_b_tree_small_inserts(test_folder):
    n = 100
    tree = BTree(test_folder)
    for i in range(n):
        tree.insert(i)
    for i in range(n):
        assert tree.query(i)
    assert not tree.query(n)

def test_b_tree_medium_inserts(test_folder):
    n = 10000
    tree = BTree(test_folder)
    for i in range(n):
        tree.insert(i)
    for i in range(n):
        assert tree.query(i)
    assert not tree.query(n)

def test_b_tree_large_inserts(test_folder):
    n = 100000
    tree = BTree(test_folder)
    for i in range(n):
        tree.insert(i)
    for i in range(n):
        assert tree.query(i)
    assert not tree.query(n)

def test_b_tree_large_random_inserts(test_folder):
    n = 100000
    arr = np.arange(n)
    rng = np.random.RandomState(42)
    rng.shuffle(arr)
    tree = BTree(test_folder)
    for i in arr:
        tree.insert(i), i
    for i in np.arange(n):
        assert tree.query(i), i
    assert not tree.query(n)

def test_b_tree_reverse_inserts(test_folder):
    n = 1000000
    tree = BTree(test_folder)
    for i in reversed(range(n)):
        tree.insert(i), i
    assert not tree.query(n)