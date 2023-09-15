from copy import deepcopy

from configuration.datasets import *
from shared.configuration_classes import ExperimentConfiguration
from shared.enums import ExperimentEnum

# Mindreader
mr_warm_start = ExperimentConfiguration('mr_warm_start', dataset=mindreader, experiment=ExperimentEnum.WARM_START)

# Movielens with Mindreader KG
ml_mr_1m_warm_start = ExperimentConfiguration('ml_mr_1m_warm_start', dataset=ml_mr_1m,
                                              experiment=ExperimentEnum.WARM_START)
ml_user_cold_start = ExperimentConfiguration('ml_user_cold_start', dataset=ml_mr, experiment=ExperimentEnum.WARM_START)

ml_user_cold_start_1250 = ExperimentConfiguration('ml_user_cold_start_1250', dataset=ml_mr,
                                                  experiment=ExperimentEnum.WARM_START)


# Synthetic dataset, validation size is smaller to have same amount as for movielens.
st_1m_warm_start = ExperimentConfiguration('st_1m_warm_start', dataset=synthetic_1m, validation_size=80,
                                           experiment=ExperimentEnum.WARM_START)
st_16_warm_start = ExperimentConfiguration('st_16_warm_start', dataset=synthetic_16, validation_size=50,
                                           experiment=ExperimentEnum.WARM_START)

# DBBook2014
db_warm_start = ExperimentConfiguration('db_warm_start', dataset=dbbook, experiment=ExperimentEnum.WARM_START)

# Last.FM
lfm_warm_start = ExperimentConfiguration('lfm_warm_start', dataset=lastfm, experiment=ExperimentEnum.WARM_START)

# Amazon Book
ab_full_warm_start = ExperimentConfiguration('ab_full_warm_start', dataset=amazon_book, experiment=ExperimentEnum.WARM_START)
ab_user_cold_start = ExperimentConfiguration('ab_user_cold_start', dataset=amazon_book, experiment=ExperimentEnum.WARM_START)
ab_warm_start = ExperimentConfiguration('ab_warm_start', dataset=amazon_book_s, experiment=ExperimentEnum.WARM_START)
test_warm = ExperimentConfiguration('test_warm', dataset=amazon_book_simplex, experiment=ExperimentEnum.WARM_START, folds=1)
val_warm = ExperimentConfiguration('val_warm', dataset=amazon_book_simplex, experiment=ExperimentEnum.WARM_START, folds=1)

beauty_warm = ExperimentConfiguration('beauty_warm', dataset=amazon_beauty, experiment=ExperimentEnum.WARM_START, folds=1)


experiments = [mr_warm_start, ml_mr_1m_warm_start, ml_user_cold_start, ml_user_cold_start_1250, st_1m_warm_start,
               lfm_warm_start, ab_full_warm_start, ab_warm_start, st_16_warm_start, test_warm, val_warm, beauty_warm,
               ab_user_cold_start]
experiment_names = [e.name for e in experiments]

# Create dgl experiments
dgl_experiments = []
for experiment in experiments:
    new_experiment = deepcopy(experiment)
    new_experiment.name = experiment.name + '_dgl'
    dgl_experiments.append(new_experiment)

dgl_experiment_names = [e.name for e in dgl_experiments]

