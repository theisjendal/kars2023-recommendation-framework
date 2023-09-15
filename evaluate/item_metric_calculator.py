import argparse
import concurrent.futures
import multiprocessing
import pickle
import sys

import os
from collections import Counter, defaultdict
from typing import List

import numpy as np
from loguru import logger
from sklearn.metrics import auc, precision_recall_curve
from tqdm import tqdm

import configuration

from configuration.experiments import experiment_names
from configuration.features import feature_conf_names
from configuration.models import dgl_models
from evaluate.dgl_evaluator import get_experiment
from shared.configuration_classes import FeatureConfiguration
from shared.experiments import Dataset, Fold
from shared.metrics import hr_at_k, ndcg_at_k, recall_at_k, precision_at_k, coverage
from shared.utility import valid_dir, join_paths, get_feature_configuration
from train.dgl_trainer import get_model_name

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=valid_dir, help='path to datasets')
parser.add_argument('--results_path', type=valid_dir, help='path to store results')
parser.add_argument('--experiments', nargs='+', type=str, choices=experiment_names, help='name of experiment')
parser.add_argument('--include', nargs='+', choices=dgl_models.keys(), help='models to include')
parser.add_argument('--seed', default=42, type=int, help='seed for reproducibility')
parser.add_argument('--debug', action='store_true', help='enable debug mode')
parser.add_argument('--folds', nargs='*', default=None, help='folds to run')
parser.add_argument('--study', choices=['standard', 'user_sparsity', 'user_popularity', 'user_test_popularity',
                                        'fixed_neg_sampling', 'idcf_neg_sampling', 'user_group'],
                    default='standard', help='evaluation study to perform')
parser.add_argument('--state', choices=experiment_names, default=None,
                    help='use state from another experiment - useful for coldstart methods')
parser.add_argument('--parameter', choices=experiment_names, default=None,
                    help='use parameters from another experiment - useful for training with pre-determined parameters')
parser.add_argument('--other_model', choices=dgl_models.keys(), default=None,
                    help='use parameters from other method i.e. change one parameter')
parser.add_argument('--feature_configuration', nargs='+', choices=feature_conf_names, default=None,
                    help='feature_configuration to use.')


def _info_fetcher(fp, start_position=0, end_position=-1):
    with open(fp, 'rb') as f:
        cur_pos = 0
        while cur_pos != end_position:
            try:
                s = pickle.load(f)
                if cur_pos >= start_position:
                    yield s.values()
                cur_pos += 1
            except EOFError:
                break


def iter_users(fp, group):
    np.random.seed(42)
    ranks = defaultdict(list)

    for user, predicted_ranking, relevance, utility, _ in _info_fetcher(fp, *group):
        for i, (item, score) in enumerate(zip(predicted_ranking, utility)):
            if score == 1:
                ranks[item].append(i)

    return ranks


def parallel_iter_users(fp, n_users):
    n_cpus = multiprocessing.cpu_count() - 2
    bs = max(min(1024, n_users // n_cpus), 64)  # Have a maximum batch size of 1024 and minimum of 64.
    n_batches = n_users // bs if n_users % bs == 0 else (n_users // bs) + 1
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cpus) as e:
        for i in tqdm(range(n_batches), desc='Submitting process futures to queue'):
            futures.append(e.submit(iter_users, fp, (i*bs, (i+1)*bs)))
            # yield iter_users(fp, (i * bs, (i + 1) * bs))

        # logger.info('Starting processes. Expect first iteration to be slow.')
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass  # yield f.result()  # Yield results as futures are complete

    for f in tqdm(futures, desc='Process data'):
        yield f.result()


def _pickle_load_users(file_path):
    users = []
    with open(file_path, 'rb') as f:
        with tqdm(None, desc='Loading users') as t:
            while True:
                try:
                    info = pickle.load(f)
                    users.append(info['user'])
                    t.update()
                except EOFError:
                    break

    return users


def iter_models(results_path: str, model_names, fold: Fold, seed: int, study, other_fold: Fold = None,
                other_parameter: Fold = None, other_model: str = None, f_conf: List[FeatureConfiguration] = None):
    testing = fold.data_loader.testing()
    meta = fold.data_loader.meta()

    # testing = [[u.index for u in b] for b in testing]  # Reduce memory
    item_popularity = Counter([i for u in testing for i, r in u.ratings])

    for name in model_names:
        results_dir = join_paths(results_path, fold.experiment.name, name)

        if f_conf is not None:
            req = True
        else:
            req = False
            f_conf = []

        model_name = get_model_name(name, f_conf, req, other_state=other_fold, other_parameter=other_parameter,
                                    other_model=other_model)

        fp = os.path.join(results_dir, f'{fold.name}_{model_name}_predictions.pickle')
        users = _pickle_load_users(fp)

        logger.debug(f'Calculating metrics for {name}')
        results = defaultdict(list)
        for ranks in parallel_iter_users(fp, len(users)):
            for item, rs in ranks.items():
                results[item].extend(rs)

        for item, ranks in results.items():
            results[item] = [item_popularity[item], ranks]

        with open(os.path.join(results_dir, f'{fold.name}_{model_name}_item.pickle'), 'wb') as f:
            pickle.dump(results, f)


def run():
    args = parser.parse_args()
    args = vars(args)

    if not args.pop('debug'):
        logger.remove()
        logger.add(sys.stderr, level='INFO')

    data_path, results_path, experiments, models, seed, folds, study, other_state, other_parameters, other_model, \
    feature_names = args.values()

    experiment_configs = [e for e in configuration.experiments.experiments if e.name in experiments]
    datasets = set([e.dataset.name for e in experiment_configs])
    feature_configurations = None
    if feature_names:
        feature_configurations = [get_feature_configuration(feature_name) for feature_name in feature_names]

    other_state = get_experiment(other_state, data_path)
    other_parameters = get_experiment(other_parameters, data_path)

    for dataset in datasets:
        dataset = Dataset(os.path.join(data_path, dataset), experiments)
        for experiment in dataset.experiments():
            logger.info(f'Running experiment: {experiment}')
            iterator = zip(*[e.folds() for e in [experiment, other_state, other_parameters] if e is not None])
            for all_fold in iterator:
                all_fold = list(all_fold)
                fold = all_fold.pop(0)

                other_fold = all_fold.pop(0) if other_state is not None else None
                other_params = all_fold.pop(0) if other_parameters is not None else None

                if folds is None or fold.name in folds:
                    logger.info(f'Running fold: {fold}')
                    iter_models(results_path, models, fold, seed, study, other_fold, other_params,
                                other_model, feature_configurations)
                else:
                    logger.warning(f'Skipping fold: {fold.name}')


if __name__ == '__main__':
    run()
