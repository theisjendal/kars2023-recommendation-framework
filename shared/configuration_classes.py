from typing import Dict, Callable, List

import configuration.datasets
from shared.enums import Sentiment, Evaluation, ExperimentEnum, FeatureEnum

_DEFAULT_SENTIMENT = {
    Sentiment.NEGATIVE: -1,
    Sentiment.POSITIVE: 1,
    Sentiment.UNSEEN: 0,
}


class CountFilter:
    def __init__(self, filter_func: Callable[[int], bool], sentiment: Sentiment):
        self.sentiment = sentiment
        self.filter_func = filter_func


class ConfigurationBase:
    def __init__(self, name: str, seed: int):
        """
        Base class for configurations. Implements a method for getting instance, e.g., getting a dataset instance given a dataset configuration.
        :param name: Should be unique for all configurations
        """
        self.name = name
        self.seed = seed


class DatasetConfiguration(ConfigurationBase):
    def __init__(self, name, ratings_mapping, sentiment_utility: Dict[Sentiment, float] = None,
                 filters: List[CountFilter] = None, max_users=None, seed=42, k_core=10):
        """
        Dataset configurations
        :param name: the dataset name, also used for storing data.
        :param ratings_mapping: Maps ratings to other rating scale.
        :param sentiment_utility: Map sentiment to same rating scale as above, default is seen in enums under shared.
        Preferably ensure reverse mapping is possible, i.e., it is bijective.
        :param max_total_ratings: Max number of positive ratings. Higher leads to slow training for multiple models.
        """
        super().__init__(name, seed)
        self.ratings_mapping = ratings_mapping
        self.sentiment_utility = _DEFAULT_SENTIMENT if sentiment_utility is None else sentiment_utility
        self.filters = [] if filters is None else filters
        self.max_users = max_users
        self.k_core = k_core


class ExperimentConfiguration(ConfigurationBase):
    def __init__(self, name: str, dataset: DatasetConfiguration, experiment: ExperimentEnum, folds: int = 5,
                 validation_size: int = 50, evaluation: Evaluation = Evaluation.LEAVE_ONE_OUT, seed: int = 42):
        """
        Experiment configuration
        :param name: the name of the experiment such as warm-start.
        :param dataset: dataset to work on.
        :param folds: number of folds to make.
        :param validation_size: size of the validation set (taken from the test set).
        :param evaluation: type of evaluation to make, i.e leave one out or other.
        """
        super().__init__(name, seed)
        self.dataset = dataset
        self.experiment = experiment
        self.folds = folds
        self.validation_size = validation_size
        self.evaluation = evaluation


class FeatureConfiguration(ConfigurationBase):
    def __init__(self, name: str, extractor: str, graph_name: str, require_ratings: bool, feature_span: FeatureEnum,
                 scale: bool = False, seed: int = 42, **kwargs):
        """
        Configuration of the feature extractor. Takes a name, extractor and if it uses ratings.
        If using ratings, then the extracted features are split based.
        :param name: the name of the configuration.
        :param extractor: the name of the feature extractor to use.
        :param graph_name: name of the graph to use (see experiment_to_dgl under datasets/converts for all possible graphs).
        :param require_ratings: whether the current feature require training ratings.
        :param feature_span: defines for which nodes features should be extracted to.
        :param scale: whether to scale the feature or not.
        :param **kwargs: parameters to pass to feature extractor.
        """
        super().__init__(name, seed)
        self.extractor = extractor
        self.graph_name = graph_name
        self.require_ratings = require_ratings
        self.feature_span = feature_span  # Maybe not needed
        self.scale = scale
        self.kwargs = kwargs
