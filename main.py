from data_structures import BEpsilonTree, LSMTree, BasicCola
from sacred import Experiment
from sacred.observers import MongoObserver
from time import perf_counter
import numpy as np
import shutil
ex = Experiment()
ex.observers.append(MongoObserver())

@ex.named_config
def default_n_input_data():
    n_input_data = int(3e5)

@ex.named_config
def default_wods_params():
    block_size = 4096
    n_blocks = 64


@ex.named_config
def wods_b_epsilon_tree():
    wods_type = 'b_epsilon_tree'


@ex.named_config
def wods_lsm_tree():
    wods_type = 'lsm_tree'


@ex.named_config
def wods_lsm_bf_tree():
    wods_type = 'lsm_bf_tree'


@ex.named_config
def wods_basic_cola():
    wods_type = 'basic_cola'


@ex.named_config
def data_seq():
    inserts_size = 0.5
    random = False


@ex.named_config
def data_random_0_5():
    inserts_size = 0.5
    random = True


@ex.named_config
def data_random_0_75():
    inserts_size = 0.75
    random = True


@ex.named_config
def data_random_0_25():
    inserts_size = 0.25
    random = True


@ex.capture
def get_wods(wods_type, block_size, n_blocks, n_input_data):
    path = f'./experiments/{wods_type}'
    shutil.rmtree(path, ignore_errors=True)
    params = {
        'block_size': block_size,
        'n_blocks': n_blocks,
        'n_input_data': n_input_data,
    }
    if wods_type == 'b_epsilon_tree':
        return BEpsilonTree(disk_filepath=path, **params)
    elif wods_type == 'lsm_tree':
        return LSMTree(disk_filepath=path, enable_bloomfilter=False, **params)
    elif wods_type == 'lsm_bf_tree':
        return LSMTree(disk_filepath=path, enable_bloomfilter=True, **params)
    elif wods_type == 'basic_cola':
        return BasicCola(disk_filepath=path, **params)


@ex.capture
def get_commands(n_input_data, inserts_size, random):
    rng = np.random.RandomState(42)
    inserts = [('insert', i) for i in rng.choice(n_input_data, int(n_input_data*inserts_size))]
    queries = [('query', i) for i in rng.choice(n_input_data, int(n_input_data*(1-inserts_size)))]
    if random:
        inserts = [('insert', i) for i in rng.choice(n_input_data, int(n_input_data*inserts_size))]
        queries = [('query', i) for i in rng.choice(n_input_data, int(n_input_data*(1-inserts_size)))]
        commands = inserts + queries
        rng.shuffle(commands)
    else:
        inserts = [('insert', i) for i in range(int(n_input_data*inserts_size))]
        queries = [('query', i) for i in range(int(n_input_data*(1 - inserts_size)))]
        commands = inserts + queries
    return commands


@ex.main
def run(_run):
    wods = get_wods()
    commands = get_commands()
    for command, data in commands:
        if command == 'insert':
            pre = perf_counter()
            wods.insert(data)
            post = perf_counter()
            _run.log_scalar("time.insert", (post - pre))
        else:
            pre = perf_counter()
            wods.query(data)
            post = perf_counter()
            _run.log_scalar("time.query", (post - pre))


if __name__ == '__main__':
    data_configs = [
        'data_seq',
        'data_random_0_5',
        'data_random_0_75',
        'data_random_0_25',
    ]
    wods_configs = [
        'wods_b_epsilon_tree',
        'wods_lsm_tree',
        'wods_lsm_bf_tree',
        'wods_basic_cola',
    ]
    # ex.run(named_configs=['default_n_input_data', 'default_wods_params', 'data_random_0_5', 'wods_b_epsilon_tree'])
    for data in data_configs:
        for wods in wods_configs:
            ex.run(named_configs=['default_n_input_data', 'default_wods_params', data, wods])
