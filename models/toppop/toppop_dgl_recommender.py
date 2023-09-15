from collections import defaultdict
import random

import dgl
import numpy as np
import torch
from loguru import logger

from models.dgl_recommender_base import RecommenderBase
from shared.efficient_validator import Validator


class TopPopDGLRecommender(RecommenderBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, validator: Validator):
        g, = self.graphs
        self.set_seed()

        u, v = g.in_edges(self.meta.items)
        unique, counts = torch.unique(v, return_counts=True)
        popularity = defaultdict(int)
        for item, count in zip(unique.tolist(), counts.tolist()):
            popularity[item] = count

        self._model = popularity
        # score = validator.validate(self)
        # logger.debug(f'Val: {score}')

    def predict_all(self, users) -> np.array:
        predictions = np.zeros((len(users), len(self.meta.items)))
        preds = [self._model[item] for item in self.meta.items]

        predictions[:] = preds

        return predictions

    def set_seed(self):
        random.seed(self._seed)
        np.random.seed(self._seed)
        dgl.seed(self._seed)
        dgl.random.seed(self._seed)

    def set_parameters(self, parameters):
        pass