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


def test_root_node_reset():
    root_node = BEpsilonTree.RootNode(buffer=[1], keys=[1], children=[1, 2])
    root_node.reset(4, 5, 10)
    assert root_node.keys == [10]
    assert root_node.children == [4, 5]


def test_branch_node_add_and_contains():
    branch_node = BEpsilonTree.BranchNode(buffer=[1, 2, 3], keys=[1], children=[1, 2])
    assert 2 in branch_node
    branch_node.add(5)
    assert 5 in branch_node

def test_branch_node_get_child():
    branch_node = BEpsilonTree.BranchNode(buffer=[], keys=[3, 5, 8], children=[1, 2, 3, 4])
    children = []
    for i in range(10):
        children.append(branch_node.get_child(i))
    assert children == [1, 1, 1, 2, 2, 3, 3, 3, 4, 4]

def test_branch_node_replace_child():
    branch_node = BEpsilonTree.BranchNode(buffer=[], keys=[3, 5, 8], children=[1, 2, 3, 4])
    branch_node.replace_child(2, 2, 5, 4)
    assert branch_node.keys == [3, 4, 5, 8]
    assert branch_node.children == [1, 2, 5, 3, 4]

def test_branch_pop_most_pending():
    branch_node = BEpsilonTree.BranchNode(buffer=[2, 3, 4, 5, 6, 7, 8, 9], keys=[3, 6, 8], children=[1, 2, 3, 4])
    child_ptr, items = branch_node.pop_most_pending()
    assert items == [3, 4, 5, 6]
    assert child_ptr == 2
    assert branch_node.buffer == [2, 7, 8, 9]

def test_branch_split():
    branch_node = BEpsilonTree.BranchNode(buffer=[2, 3, 4, 5, 6, 7, 8, 9], keys=[3, 6, 8], children=[1, 2, 3, 4])
    left_node, right_node, median = branch_node.split()
    assert median == 6
    assert left_node.children == [1, 2]
    assert left_node.buffer == [2, 3, 4, 5]
    assert right_node.children == [3, 4]
    assert right_node.buffer == [6, 7, 8, 9]

def test_root_reset():
    root_node = BEpsilonTree.RootNode(buffer=[2, 3, 4, 5, 6, 7, 8, 9], keys=[3, 6, 8], children=[1, 2, 3, 4])
    assert isinstance(root_node, BEpsilonTree.BranchNode)
    assert isinstance(root_node, BEpsilonTree.RootNode)
    root_node.reset(1, 2, 3)
    assert root_node.buffer == []
    assert root_node.keys == [3]
    assert root_node.children == [1, 2]

def test_leaf_add():
    leaf_node = BEpsilonTree.LeafNode(records=[1, 2])
    leaf_node.add(3)
    assert leaf_node.records == [1, 2, 3]

def test_leaf_split():
    leaf_node = BEpsilonTree.LeafNode(records=[1, 2, 3, 4, 5, 6, 7, 8])
    left_node, right_node, median = leaf_node.split()
    assert median == 4
    assert left_node.records == [1, 2, 3]
    assert right_node.records == [4, 5, 6, 7, 8]

def test_b_epsilon_tree_init(test_folder):
    tree = BEpsilonTree(test_folder)
    assert isinstance(tree.blocks[0], BEpsilonTree.RootNode)
    assert isinstance(tree.blocks[1], BEpsilonTree.LeafNode)

def test_b_epsilon_tree_in_buffer_inserts(test_folder):
    n = 100
    tree = BEpsilonTree(test_folder)
    for i in range(n):
        tree.insert(i)
    for i in range(n):
        assert tree.query(i)
    assert not tree.query(n)

def test_b_epsilon_tree_flushed_inserts(test_folder):
    n = 2000
    tree = BEpsilonTree(test_folder)
    for i in range(n):
        tree.insert(i)
    for i in range(n):
        assert tree.query(i)
    assert not tree.query(n)

def test_b_epsilon_tree_leaf_split_inserts(test_folder):
    n = 3000
    tree = BEpsilonTree(test_folder)
    for i in range(n):
        tree.insert(i)
    for i in range(n):
        assert tree.query(i)
    assert not tree.query(n)

def test_b_epsilon_tree_root_split_inserts(test_folder):
    n = 6000
    tree = BEpsilonTree(test_folder)
    for i in range(n):
        tree.insert(i)
    for i in range(n):
        assert tree.query(i)
    assert not tree.query(n)

def test_b_epsilon_tree_branch_split_inserts(test_folder):
    n = 10000
    tree = BEpsilonTree(test_folder)
    for i in range(n):
        tree.insert(i)
    for i in range(n):
        assert tree.query(i)
    assert not tree.query(n)

def test_b_epsilon_tree_large_inserts(test_folder):
    n = 100000
    tree = BEpsilonTree(test_folder)
    for i in range(n):
        tree.insert(i)
    for i in range(n):
        assert tree.query(i)
    assert not tree.query(n)