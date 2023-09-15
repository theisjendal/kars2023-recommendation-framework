import os
from typing import List

import pandas as pd

from shared.experiments import Dataset
from shared.meta import Meta
from shared.user import User, LeaveOneOutUser
from shared.utility import get_experiment_configuration


def store_triplets(fold_path, name, data: List[User]):
    triplets = [[user.index, item, rating] for user in data
                for item, rating in (user.ratings if type(user) != LeaveOneOutUser else user.loo)]
    triplets = dict(zip(['user', 'item', 'rating'], zip(*triplets)))

    df = pd.DataFrame.from_dict(triplets)
    df.to_csv(os.path.join(fold_path, name + '.csv'))


def store_simple_info(fold_path, meta: Meta):
    pass


def run(path, experiment):
    experiment = get_experiment_configuration(experiment)

    dataset = Dataset(os.path.join(path, experiment.dataset.name), [experiment.name])

    for exp in dataset.experiments():
        for i, fold in enumerate(exp.folds()):
            data_loader = fold.data_loader
            train = data_loader.training()
            val = data_loader.validation()
            test = data_loader.testing()
            meta = data_loader.meta()

            store_triplets(exp.fold_paths[i], 'train', train)
            store_triplets(exp.fold_paths[i], 'validation', val)
            store_triplets(exp.fold_paths[i], 'test', test)


if __name__ == '__main__':
    run('..', 'ml_mr_1m_warm_start')