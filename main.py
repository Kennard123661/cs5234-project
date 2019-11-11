from data_structures import BEpsilonTree, LSMTree, \
    BasicCola, FractionalCola, BasicBloomCola, FractionalBloomCola
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
        return BEpsilonTree(disk_filepath=path, **params, b=8)
    elif wods_type == 'lsm_tree':
        return LSMTree(disk_filepath=path, enable_bloomfilter=False, growth_factor=16, **params)
    elif wods_type == 'lsm_bf_tree':
        return LSMTree(disk_filepath=path, enable_bloomfilter=True, growth_factor=16, **params)
    elif wods_type == 'basic_cola':
        return BasicCola(disk_filepath=path, growth_factor=16, **params)
    elif wods_type == 'fractional_cola':
        return FractionalCola(disk_filepath=path, growth_factor=16, **params)
    elif wods_type == 'basic_bloom_cola':
        return BasicBloomCola(disk_filepath=path, growth_factor=16, **params)


@ex.capture
def get_commands(n_input_data, inserts_size, random, interspersed, reversed):
    rng = np.random.RandomState(42)
    if random:
        inserts = [('insert', i) for i in rng.choice(
            n_input_data, int(n_input_data*inserts_size))]
        queries = [('query', i) for i in rng.choice(
            n_input_data, int(n_input_data*(1-inserts_size)))]
    else:
        inserts = [('insert', i)
                   for i in range(int(n_input_data*inserts_size))]
        queries = [('query', i)
                   for i in range(int(n_input_data*(1 - inserts_size)))]
    if reversed:
        inserts.reverse()
        queries.reverse()
    commands = inserts + queries
    if interspersed:
        rng.shuffle(commands)
    return commands


@ex.main
def run(_run):
    results = []
    wods = get_wods()
    commands = get_commands()
    pre_time = perf_counter()
    inserts_ran = 0
    queries_ran = 0
    for command, data in commands:
        if command == 'insert':
            inserts_ran += 1
            pre = perf_counter()
            wods.insert(data)
            post = perf_counter()
            results.append(('insert', post - pre))
        else:
            queries_ran += 1
            pre = perf_counter()
            wods.query(data)
            post = perf_counter()
            results.append(('query', post - pre))
        if post - pre_time >= 1:
            _run.log_scalar("inserts_per_sec", inserts_ran)
            _run.log_scalar("queries_per_sec", queries_ran)
            inserts_ran = 0
            queries_ran = 0
            pre_time = post
    df = pd.DataFrame(data=results, columns=['type', 'time'])
    df.index.name = 'id'
    df.to_csv('./experiments/Results.csv')
    _run.add_artifact('./experiments/Results.csv')


if __name__ == '__main__':
    generic_params = [{
        'block_size': 4096,
        'n_blocks': 64,
        'n_input_data': int(1e6),
    }]
    data_params = [
        {
            'inserts_size': 0.5,
            'random': False,
            'interspersed': False,
            'reversed': False,
        },
        {
            'inserts_size': 0.5,
            'random': False,
            'interspersed': False,
            'reversed': True,
        },
        {
            'inserts_size': 0.5,
            'random': True,
            'interspersed': False,
            'reversed': False,
        },
        {
            'inserts_size': 0.2,
            'random': True,
            'interspersed': True,
            'reversed': False,
        },
        {
            'inserts_size': 0.4,
            'random': True,
            'interspersed': True,
            'reversed': False,
        },
        {
            'inserts_size': 0.6,
            'random': True,
            'interspersed': True,
            'reversed': False,
        },
        {
            'inserts_size': 0.8,
            'random': True,
            'interspersed': True,
            'reversed': False,
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
        },
        {
            'wods_type': 'fractional_cola',
        },
        {
            'wods_type': 'basic_bloom_cola',
        },
        {
            'wods_type': 'fractional_bloom_cola',
        },
    ]
    # config = {**generic_params[0], **data_params[0], **wods_params[4]}
    # ex.run(config_updates=config)
    configs = [{**d1, **d2, **d3} for d1, d2,
               d3 in itertools.product(generic_params, data_params, wods_params)]
    for config in configs:
        ex.run(config_updates=config)
