import argparse

import os
import random
import sys

import pickle
from loguru import logger
from shutil import copyfile

from configuration.experiments import experiment_names
from shared.utility import valid_dir, get_experiment_configuration, load_users, load_pickle, save_pickle

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=valid_dir, help='path to datasets')
parser.add_argument('--experiment_a', choices=experiment_names, help='datasets to use as warm start')
parser.add_argument('--experiment_b', choices=experiment_names, help='datasets to used as cold start')
parser.add_argument('--new_name', help='name of new dataset')
parser.add_argument('--n_cold_users', help='number of cold start users', default=-1, type=int)
parser.add_argument('--test_both', action='store_true', help='warm start test set is deleted by default, use flag to '
                                                             'keep warm start users in test set (i.e. experiment a)')


def run(path, experiment_a, experiment_b, new_name, n_cold_users, test_both):
    ea = get_experiment_configuration(experiment_a)
    eb = get_experiment_configuration(experiment_b)

    # Ensure reproducibility
    random.seed(ea.seed)

    dataset_path_a = os.path.join(path, ea.dataset.name)
    dataset_path_b = os.path.join(path, eb.dataset.name)
    experiment_path_a = os.path.join(dataset_path_a, ea.name)
    experiment_path_b = os.path.join(dataset_path_b, eb.name)
    out = os.path.join(path, eb.dataset.name, new_name + (f'_{n_cold_users}' if n_cold_users != -1 else ''))


    os.makedirs(out, exist_ok=True)
    copyfile(os.path.join(experiment_path_b, 'meta.pickle'), os.path.join(out, 'meta.pickle'))

    meta = pickle.load(open(os.path.join(experiment_path_b, 'meta.pickle'), 'rb'))

    # load users a,b
    users_a = load_users(dataset_path_a, fname_extension='processed')
    users_b = load_users(dataset_path_b, fname_extension='processed')

    orig_index_b = {u.original_id: u.index for u in users_b}

    # Map dataset users from a to b and vice versa
    a_b_map = {u.index: orig_index_b[u.original_id] for u in users_a}
    b_a_map = {orig_index_b[u.original_id]: u.index for u in users_a}

    for fold in sorted(os.listdir(experiment_path_a)):
        if fold.startswith('fold_'):
            logger.info(f'Processing {fold}')
            random.seed(ea.seed)  # Ensure reproducibility

            fold_path_a = os.path.join(experiment_path_a, fold)
            fold_path_b = os.path.join(experiment_path_b, fold)

            # load b train, test, val
            train = load_pickle(os.path.join(fold_path_b, 'train.pickle'))
            test = load_pickle(os.path.join(fold_path_b, 'test.pickle'))
            val = load_pickle(os.path.join(fold_path_b, 'validation.pickle'))

            # remove a's users from b's train test
            train = [u for u in train if u.index not in b_a_map]
            test = [u for u in test if u.index not in b_a_map]
            val = [u for u in val if u.index not in b_a_map]

            # load a train, test, val
            train_a = load_pickle(os.path.join(fold_path_a, 'train.pickle'))
            test_a = load_pickle(os.path.join(fold_path_a, 'test.pickle'))
            val_a = load_pickle(os.path.join(fold_path_a, 'validation.pickle'))

            assert all([item in meta.items for user in train for item, _ in user.ratings])
            assert all([item in meta.items for user in train_a for item, _ in user.ratings])

            # Map index of a to b index
            for d in [train_a, test_a, val_a]:
                for u in d:
                    u.index = a_b_map[u.index]

            # Subsample users from validation set (all user in val is in train and test)
            if n_cold_users > -1:
                random.shuffle(test)
                test = test[:n_cold_users]
            user_set = {u.index for u in test}

            # Prune users from test and train
            train = [u for u in train if u.index in user_set]
            test = [u for u in test if u.index in user_set]

            # insert a's users into b
            train.extend(train_a)
            val.extend(val_a)

            # Default only test on cold-start users. If flag is true test on warm-start as well
            if test_both:
                test.extend(test_a)

            train_items = {i for u in train for i, _ in u.ratings}
            test_items = {i for u in test for i, _ in u.ratings}
            val_items = {i for u in test for i, _ in u.ratings}

            assert test_items.issubset(train_items)
            assert val_items.issubset(train_items)

            for d in [train, test, val]:
                d.sort(key=lambda x: x.index)

            # save
            fold_out = os.path.join(out, fold)
            os.makedirs(fold_out, exist_ok=True)

            save_pickle(os.path.join(fold_out, 'train.pickle'), train)
            save_pickle(os.path.join(fold_out, 'test.pickle'), test)
            save_pickle(os.path.join(fold_out, 'validation.pickle'), val)


if __name__ == '__main__':
    args = parser.parse_args()
    args = vars(args)
    logger.remove()
    logger.add(sys.stderr, level='INFO')
    run(**args)
