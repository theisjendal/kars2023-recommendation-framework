import argparse
import json
import os
import sys
from typing import List, Union

import numpy as np
import pickle
from loguru import logger
from tqdm import tqdm

import configuration
from configuration.experiments import experiment_names
from configuration.features import feature_conf_names

from configuration.models import dgl_models
from models.dgl_recommender_base import RecommenderBase
from shared.configuration_classes import FeatureConfiguration
from shared.efficient_validator import Validator
from shared.enums import Sentiment
from shared.experiments import Dataset, Fold
from shared.meta import Meta
from shared.seed_generator import SeedGenerator
from shared.user import LeaveOneOutUser
from shared.utility import valid_dir, join_paths, get_experiment_configuration, get_feature_configuration
from train.dgl_trainer import _instantiate_model, _get_state, _get_model_path, get_model_name

parser = argparse.ArgumentParser(epilog='Arguments --state, --parameter, and --other_model can be used together. '
                                        'Be careful when using them as they affect all model used with the --include '
                                        'argument.')
parser.add_argument('--data', type=valid_dir, help='path to datasets')
parser.add_argument('--results_path', type=valid_dir, help='path to store results')
parser.add_argument('--experiments', nargs='+', type=str, choices=experiment_names, help='name of experiment')
parser.add_argument('--include', nargs='+', choices=dgl_models.keys(), help='models to include')
parser.add_argument('--test_batch', default=None, type=int, help='predict in batches, with default being one user at a '
                                                                 'time')
parser.add_argument('--seed', default=42, type=int, help='seed for reproducibility')
parser.add_argument('--debug', action='store_true', help='enable debug mode')
parser.add_argument('--folds', nargs='*', default=None, help='folds to run')
parser.add_argument('--pre_sample', action='store_true', help='if set to true remove all items ranked lower than 100 - '
                                                              'may affect subsequent compuations depending on the '
                                                              'study')
parser.add_argument('--workers', default=0, type=int, help='number of workers, may give different results when workers '
                                                           'is not 0')
parser.add_argument('--state', choices=experiment_names, default=None,
                    help='use state from another experiment - useful for coldstart methods')
parser.add_argument('--parameter', choices=experiment_names, default=None,
                    help='use parameters from another experiment - useful for training with pre-determined parameters')
parser.add_argument('--parameter_usage', choices=range(2), default=0, type=int,
                    help='when parameter argument was used. 0) it was not passed during training (default) and 1)'
                         'it was passed during training.')
parser.add_argument('--other_model', choices=dgl_models.keys(), default=None,
                    help='use parameters from other method i.e. change one parameter')
parser.add_argument('--other_model_usage', choices=range(3), default=1, type=int,
                    help='where other model was used: 0) location for parameters, 1) location of state, 2) both'
                         'E.g., if you want to load parameters from another model but use state from the model '
                         'itself use 1.')
parser.add_argument('--feature_configuration', nargs='+', choices=feature_conf_names, default=['processed'],
                    help='feature_configuration to use.')


def get_ranks(users, model, meta, sg, item_index, presample):
    predictions = model.predict_all(np.array([user.index for user in users]))

    assert predictions.shape == (len(users), len(meta.items)), f'Expected predictions to be of shape ' \
                                                               f'{(len(users), len(meta.items))} got ' \
                                                               f'{predictions.shape}.'

    topk = 100
    index_item = {idx: i for i, idx in item_index.items()}

    for idx, user in enumerate(users):
        predictions[idx, [item_index[item] for item, rating in user.ratings]] = -np.inf

    if presample:
        indices = np.argpartition(predictions, kth=-topk, axis=-1)[:, -topk:]  # get top 100 indices.
    else:
        indices = np.argsort(predictions, axis=-1)
    item_ranking = np.vectorize(index_item.get)(indices)
    n_liked = np.zeros(len(users))

    rankings = [user.get_ranking(sg.get_seed(), meta) for user in users]

    should_remove = []
    failed = []
    for i, r in enumerate(rankings):
        neg = np.isin(item_ranking[i], r.sentiment_samples[Sentiment.UNSEEN])
        pos = np.isin(item_ranking[i], r.sentiment_samples[Sentiment.POSITIVE])

        # Get train data
        should_remove.append(item_ranking[i, ~(neg | pos)])
        n_liked[i] = len(r.sentiment_samples[Sentiment.POSITIVE])

        # Sum of neg and pos must be larger than 50.
        if np.sum(neg) + np.sum(pos) <= 50:
            failed.append(i)


    ranked_lists = [(r,
        (sorted([item for item in item_ranking[i] if item not in should_remove[i]],
               key=lambda item: (predictions[i][item_index[item]], item), # sort by score, dec.
               reverse=True)))
                    for i, r in enumerate(rankings)]

    assert len(failed) == 0, 'Need to implement recovery code for this case or increase topk variable'

    return ranked_lists, n_liked


def _test(testing: List[LeaveOneOutUser], model: RecommenderBase, meta: Meta, sg: SeedGenerator,
          test_batch: Union[int, None], presample):

    batches = [testing[i:i+test_batch] for i in range(0, len(testing), test_batch)]
    item_index = {item: idx for idx, item in enumerate(meta.items)}

    for users in tqdm(batches, desc='Predicting'):
        ranked_lists, n_liked = get_ranks(users, model, meta, sg, item_index, presample)

        for i, (ranking, ranked_list) in enumerate(ranked_lists):
            # From the ranked list, get ordered binary relevance and utility
            relevance = ranking.get_relevance(ranked_list)
            utility = ranking.get_utility(ranked_list, meta.sentiment_utility)

            yield users[i], ranked_list, relevance, utility, n_liked[i]


def _test_model(model: RecommenderBase, validator, testing, seed, meta, batch_size, presample):
    model.set_seed()

    model.fit(validator)

    yield from _test(testing, model, meta, SeedGenerator(seed), batch_size, presample)


def _write_results(output_path, model_name, metrics, fold: Fold, extension=None):
    results_dir = join_paths(output_path, fold.experiment.name, model_name)
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, f'{fold.name}{"" if extension is None else "_" + extension}.json'), 'w') as fp:
        json.dump(metrics, fp, indent=True)


def get_experiment(e, data_path):
    if e:
        other_experiment = get_experiment_configuration(e)
        other_dataset = Dataset(os.path.join(data_path, other_experiment.dataset.name), [e])
        other_experiment = next(other_dataset.experiments())
    else:
        other_experiment = None

    return other_experiment


def evaluate_fold(out_path: str, model_names, fold: Fold, feature_configurations: List[FeatureConfiguration], seed: int,
                  workers, presample, test_batch=None, other_fold: Fold = None, other_params: Fold = None,
                  parameter_value=None, other_model=None, model_value=None):
    """
    Evaluate models
    :param out_path: Path to datasets
    :param model_names: List of model names
    :param fold: Experiment fold to evaluate
    :param feature_configurations: Features to use for models
    :param seed: seed
    :param workers: number of workers
    :param presample: limits the length of the stored results, but cannot be used with certain studies (e.g. idcf sampling)
    :param test_batch: batch size
    :param other_fold: where to get state from if another experiment is passed as input
    :param other_params: where to get parameters from if another experiment is passed as input
    :param other_model: model state to use if another model is passed as input
    :return: None
    """
    validator = Validator(fold.data_loader.path, test_batch)

    meta = fold.data_loader.meta()
    testing = fold.data_loader.testing()

    for name in model_names:
        logger.info(f'Running model: {name}')

        # Use other fold or other datasets parameter and state information is passed
        model = _instantiate_model(name, meta, SeedGenerator(seed), fold, feature_configurations, workers, out_path,
                                   other_params, other_model if model_value in [0, 2] else None)

        state_name = other_model if other_model is not None and model_value in [1, 2] else name
        model_state_path = _get_model_path(out_path, fold if other_fold is None else other_fold, state_name)

        # IDCF does not use features in the warm start setting but uses a key-user mask in an inductive setting.
        require_features = model.require_features if not (name.startswith('idcf') or name.startswith('inmo')) else False

        state_name = get_model_name(state_name, feature_configurations, require_features,
                                    other_model=other_model if model_value == 0 else None,
                                    other_parameter=other_params if parameter_value == 1 else None)
        state = _get_state(model_state_path, fold, state_name)

        if state is not None:
            logger.debug('Setting state')
            model.set_state(state)

        if model.get_state() is not None:
            logger.debug('Using saved state')

        model_outpath = _get_model_path(out_path, fold, name)
        m_name = get_model_name(name, feature_configurations, model.require_features,
                                        other_parameter=other_params, other_state=other_fold, other_model=other_model)
        fp = os.path.join(model_outpath, f'{fold.name}_{m_name}_predictions.pickle')
        with open(fp, 'wb') as f:
            for user, predicted_ranking, relevance, utility, n_liked in _test_model(model, validator, testing, seed,
                                                                                    meta, test_batch, presample):
                pickle.dump({'user': user.index, 'predicted': predicted_ranking, 'relevance': relevance,
                             'utility': utility, 'n_liked': n_liked}, f)


def run():
    args = parser.parse_args()
    args = vars(args)

    if not args.pop('debug'):
        logger.remove()
        logger.add(sys.stderr, level='INFO')

    data_path, out_path, experiments, models, test_batch, seed, folds, presample, workers, \
        other_state, other_parameters, parameter_value, other_model, model_values, feature_names = args.values()

    experiment_configs = [e for e in configuration.experiments.experiments if e.name in experiments]

    feature_configurations = [get_feature_configuration(feature_name) for feature_name in feature_names]

    datasets = set([e.dataset.name for e in experiment_configs])

    other_state = get_experiment(other_state, data_path)
    other_parameters = get_experiment(other_parameters,data_path)

    for dataset in datasets:
        dataset = Dataset(os.path.join(data_path, dataset), experiments)
        for experiment in dataset.experiments():
            logger.info(f'Running experiment: {experiment}')

            # Get experiments which are not none.
            iterator = zip(*[e.folds() for e in [experiment, other_state, other_parameters] if e is not None])
            for all_fold in iterator:
                all_fold = list(all_fold)
                fold = all_fold.pop(0)

                other_fold = all_fold.pop(0) if other_state is not None else None
                other_params = all_fold.pop(0) if other_parameters is not None else None

                if folds is None or fold.name in folds:
                    logger.info(f'Running fold: {fold}')
                    evaluate_fold(out_path, models, fold, feature_configurations, seed, workers, presample, test_batch,
                                  other_fold, other_params, parameter_value, other_model, model_values)
                else:
                    logger.warning(f'Skipping fold: {fold.name}')


if __name__ == '__main__':
    run()