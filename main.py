from data_structures import BEpsilonTree, LSMTree, BasicCola
from sacred import Experiment
from sacred.observers import MongoObserver
from time import perf_counter
import numpy as np
import pandas as pd
import shutil
import itertools

ex = Experiment()
ex.observers.append(MongoObserver())

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
        return LSMTree(disk_filepath=path, enable_bloomfilter=False, growth_factor=8, **params)
    elif wods_type == 'lsm_bf_tree':
        return LSMTree(disk_filepath=path, enable_bloomfilter=True, growth_factor=8, **params)
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
    results = []
    wods = get_wods()
    commands = get_commands()
    pre_time = perf_counter()
    commands_ran = 0
    for command, data in commands:
        commands_ran += 1
        if command == 'insert':
            pre = perf_counter()
            wods.insert(data)
            post = perf_counter()
            results.append(('insert', post - pre))
        else:
            pre = perf_counter()
            wods.query(data)
            post = perf_counter()
            results.append(('query', post - pre))
        if post - pre_time >= 1:
            _run.log_scalar("commands_per_sec", commands_ran)
            commands_ran = 0
            pre_time = post
    df = pd.DataFrame(data=results, columns=['type', 'time'])
    df.index.name = 'id'
    df.to_csv('Results.csv')
    _run.add_artifact('Results.csv')


if __name__ == '__main__':
    generic_params = [{
        'block_size': 4096,
        'n_blocks': 64,
        'n_input_data': int(1e5),
    }]
    data_params = [
        {
            'inserts_size': 0.5,
            'random': False,
        },
        {
            'inserts_size': 0.25,
            'random': True,
        },
        {
            'inserts_size': 0.5,
            'random': True,
        },
        {
            'inserts_size': 0.75,
            'random': True,
        },
    ]
    wods_params = [
        {
            'wods_type': 'b_epsilon_tree',
        },
        {
            'wods_type': 'lsm_tree',
        },
        {
            'wods_type': 'lsm_bf_tree',
        },
        {
            'wods_type': 'basic_cola',
        }
    ]
    configs = [{**d1, **d2, **d3} for d1, d2, d3 in itertools.product(generic_params, data_params, wods_params)]
    for config in configs:
        ex.run(config_updates=config)
