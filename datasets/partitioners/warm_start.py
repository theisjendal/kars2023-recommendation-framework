from collections import Counter
from typing import List

import numpy as np
from loguru import logger
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm

from shared.configuration_classes import ExperimentConfiguration
from shared.enums import Sentiment
from shared.seed_generator import SeedGenerator
from shared.user import User, LeaveOneOutUser


def _test(unique, indices, counts, seen_indices, seen_counts, rating_matrix: np.ndarray, seen_ratings: np.ndarray):
    users = np.zeros((len(unique),), dtype=object)
    for i in tqdm(np.arange(len(unique))):
        user, index, count, seen_index, seen_count = unique[i], indices[i], counts[i], seen_indices[i], seen_counts[i]

        users[i] = LeaveOneOutUser(
            user,
            list(map(tuple, seen_ratings[seen_index:seen_index + seen_count].tolist())),
            list(map(tuple, rating_matrix[index:index + count].tolist()))
        )

    return users.tolist()


def _update(users, indices, counts, train):
    for user in tqdm(users):
        index, count = indices[user.index], counts[user.index]
        user.ratings = [tuple(t) for t in train[index:index + count].tolist()]


def create_loo_users(rating_matrix: np.ndarray, seen_ratings: np.ndarray, unique_tuple):
    rating_matrix = rating_matrix[np.argsort(rating_matrix[:, 0])]
    unique, indices, counts = np.unique(rating_matrix[:, 0], return_counts=True, return_index=True)
    rating_matrix = rating_matrix[:, 1:]

    _, seen_indices, seen_counts = [element[unique] for element in unique_tuple]

    users = _test(unique, indices, counts, seen_indices, seen_counts, rating_matrix, seen_ratings)

    return users


def get_ratings_matrix(users: List[User]):
    ratings = [[user.index, item, rating] for user in users for item, rating in user.ratings]

    return np.array(ratings)


def fold_data_iterator_old(path, experiment: ExperimentConfiguration, kfold: KFold, entities, users, relations,
                       sg: SeedGenerator):
    logger.info('Preparing data, may be slow for large datasets')
    np.random.seed(sg.get_seed())
    ratings_matrix = get_ratings_matrix(users)
    positive_ratings_index = ratings_matrix[:, -1] == experiment.dataset.sentiment_utility[Sentiment.POSITIVE]
    positive_ratings = ratings_matrix[positive_ratings_index]
    rest = ratings_matrix[~positive_ratings_index]

    def swap(first, second):
        store = positive_ratings[first]
        positive_ratings[first] = positive_ratings[second]
        positive_ratings[second] = store

    for fold, (train_index, test_index) in tqdm(enumerate(kfold.split(positive_ratings)), total=experiment.folds,
                                                desc='Creating splits', disable=True):
        logger.debug(f'Creating fold: {fold}')
        # Swap train and test such that number of ratings for training is similar to normal datasets but testing is
        # much more accurate as ratings are always true.
        if experiment.dataset.name == 'synthetic':
            tmp = test_index
            test_index = train_index
            train_index = tmp

        # Ensure test users and items occur in train set. If not swap. This means that there might be a small
        # overlap in our partitions, and is therefore not a pure KFOLD.
        undo_list = []
        change_occurred = True
        while change_occurred:
            change_occurred = False
            to_break = False
            train = positive_ratings[train_index]
            test = positive_ratings[test_index]

            # For user and item (i.e., 0,1) find any in test that does not appear in train, and swap one
            # rating triple with random in train.
            for index in range(2):
                difference = set(test[:, index]).difference(train[:, index])

                for i in difference:
                    # Get indices
                    test_sample = np.random.choice(np.where(test[:, index] == i)[0], 1)
                    test_sample = test_index[test_sample]
                    train_sample = np.random.choice(train_index, 1)

                    swap(test_sample, train_sample)
                    undo_list.append((test_sample, train_sample))

                    # update flags
                    change_occurred = True
                    to_break = True

                if to_break:
                    break

        # Remove ratings from users for space efficiency.
        for user in users:
            user.ratings = []

        train = positive_ratings[train_index]
        test = positive_ratings[test_index]

        assert set(test[:, 0]).issubset(train[:, 0]), 'All users in test set are not present in train set'
        assert set(test[:, 1]).issubset(train[:, 1]), 'All items in test set are not present in train set'

        train = np.concatenate((train, rest))
        train = train[np.argsort(train[:, 0])]
        unique, indices, counts = np.unique(train[:, 0], return_counts=True, return_index=True)
        unique_tuple = unique, indices, counts
        train = train[:, 1:]

        test, val = train_test_split(test, test_size=experiment.validation_size / 100.,
                                     random_state=sg.get_seed(), shuffle=True)

        logger.debug('Users')
        # Set ratings for test and validation sets
        for user in tqdm(users):
            index, count = indices[user.index], counts[user.index]
            user.ratings = [tuple(t) for t in train[index:index + count].tolist()]

        yield users
        yield create_loo_users(val, train, unique_tuple)
        yield create_loo_users(test, train, unique_tuple)

        for first, second in undo_list:
            swap(first, second)


def fold_data_iterator(path, experiment: ExperimentConfiguration, kfold: KFold, entities, users, relations,
                       sg: SeedGenerator):
    logger.info('Preparing data, may be slow for large datasets')
    np.random.seed(sg.get_seed())
    ratings_matrix = get_ratings_matrix(users)
    positive_ratings_index = ratings_matrix[:, -1] == experiment.dataset.sentiment_utility[Sentiment.POSITIVE]
    positive_ratings = ratings_matrix[positive_ratings_index]
    rest = ratings_matrix[~positive_ratings_index]
    special_datasets = ['synthetic', 'synthetic-1m']

    # order users and shuffle their ratings within a user slice.
    length = len(positive_ratings)
    positive_ratings = positive_ratings[np.argsort(positive_ratings[:, 0] + np.random.rand(length))]

    # Get num ratings per user and their
    user_counts = Counter(positive_ratings[:, 0])

    # Generate user splits
    user_folds = {}
    cur_index = 0
    while cur_index < length:
        user = positive_ratings[cur_index, 0]
        count = user_counts[user]

        # Only add user if split is possible.
        if count >= experiment.folds:
            ratings = positive_ratings[cur_index:cur_index+count, 1]
            user_folds[user] = kfold.split(ratings)
        else:
            rest = np.concatenate([rest, positive_ratings[cur_index:cur_index+count]], axis=0)

        # Increase index
        cur_index += count

    # Method for swapping ratings in matrix. Used to ensure all items in test and validation occurs in training set.
    def swap(first, second):
        store = positive_ratings[first]
        positive_ratings[first] = positive_ratings[second]
        positive_ratings[second] = store

    # Create folds
    for fold in range(experiment.folds):
        logger.info(f'Running fold: {fold}')
        train_index = []
        test_index = []
        cur_index = 0

        # Get the next fold for each user
        for user, count in user_counts.items():
            if user in user_folds:
                folds = user_folds[user]
                train, test = next(folds)  # Get indices as values from 0-#ratings.
                train_index.append(train + cur_index)  # Ratings are sorted by user so increase indices w/ #ratings.
                test_index.append(test + cur_index)
            cur_index += user_counts[user]

        # Combine all indices to create train/test split.
        train_index = np.concatenate(train_index)
        test_index = np.concatenate(test_index)

        # As the synthetic dataset ratings on everything, which we want to use for better evaluation, we train
        # on a smaller part of the partitioning, i.e, the test part
        if experiment.dataset.name in special_datasets:
            # Swap test and train sets
            tmp = test_index
            test_index = train_index
            train_index = tmp

            # Create validation set
            train_index, val_index = train_test_split(train_index, test_size=experiment.validation_size / 100.,
                                                      random_state=sg.get_seed(), shuffle=True)

            val = positive_ratings[val_index]

        # Ensure test users and items occur in train set. If not swap. This means that there might be a small
        # overlap in our partitions, and is therefore not a pure KFOLD.
        undo_list = []
        change_occurred = True
        while change_occurred:
            change_occurred = False
            to_break = False
            train = positive_ratings[train_index]
            test = positive_ratings[test_index]

            # For user and item (i.e., 0,1) find any in test that does not appear in train, and swap one
            # rating triple with random in train.
            for index in range(2):
                # Check if either all users or all items appear in train set, meaning difference is empty
                difference = set(test[:, index]).difference(train[:, index])

                # If difference, then swap and break
                for i in difference:
                    # Get indices
                    test_sample = np.random.choice(np.where(test[:, index] == i)[0], 1)
                    test_sample = test_index[test_sample]
                    train_sample = np.random.choice(train_index, 1)

                    swap(test_sample, train_sample)
                    undo_list.append((test_sample, train_sample))

                    # update flags
                    change_occurred = True
                    to_break = True

                if to_break:
                    break

        # Remove ratings from users for space efficiency.
        for user in users:
            user.ratings = []

        train = positive_ratings[train_index]
        test = positive_ratings[test_index]

        # For fair validation we also only use the small partition for synthetic. Note: train is the normal
        # test partition in case of synthetic dataset.
        if experiment.dataset.name not in special_datasets:
            test, val = train_test_split(test, test_size=experiment.validation_size / 100.,
                                         random_state=sg.get_seed(), shuffle=True)

        # Ensure test and validation users and items are in train set
        for partition in [test, val]:
            assert set(partition[:, 0]).issubset(train[:, 0]), 'At least one user in the validation or test set is not in the train set'
            assert set(partition[:, 1]).issubset(train[:, 1]), 'At least one item in the validation or test set is not in the train set'

        train = np.concatenate((train, rest))
        train = train[np.argsort(train[:, 0])]
        unique, indices, counts = np.unique(train[:, 0], return_counts=True, return_index=True)
        unique_tuple = unique, indices, counts
        train = train[:, 1:]

        _update(users, indices, counts, train)

        yield users
        yield create_loo_users(val, train, unique_tuple)
        yield create_loo_users(test, train, unique_tuple)

        for first, second in undo_list:
            swap(first, second)