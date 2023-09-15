import time

import numpy as np
import torch
from torch import nn

from models.dgl_recommender_base import RecommenderBase
from shared.efficient_validator import Validator
from shared.meta import Meta


class MockRecommender(RecommenderBase):
    def __init__(self, meta: Meta, attention=None, trainable_users=False, relations=True, activation=False,
                 batch_size=1024, **kwargs):
        features = kwargs.pop('features', True)  # Default is true
        kwargs['use_cuda'] = True
        super().__init__(meta, **kwargs, features=features)
        self.attention = attention
        self.trainable_user = trainable_users
        self.relations = relations
        self.n_entities = len(meta.entities)
        self.user_fn = lambda u: u + self.n_entities
        self.n_users = len(meta.users)
        self.activation = activation
        self.batch_size = batch_size
        self.lr = 0.001
        self.weight_decay = 1e-5
        self.n_layers = 3
        self.autoencoder_layer_dims = [128, 32]
        self.layer_dims = [32, 16, 8, 8, 8]
        self.autoencoder_weight = 0.001
        self.dropouts = [0.1, 0.1, 0.1]
        self.gate_type = 'concat'
        self.aggregator = 'gcn'
        self.n_layer_samples = [10, 10, 10]
        self.feature_dim = None
        self.attention_dim = None if self.attention is None else 32

        self.optimizer = None
        self.entity_embeddings = None
        self.user_embeddings = None

    def _create_model(self, trial):
        self._model = nn.Linear(1,1)
        self._sherpa_load(trial)

    def fit(self, validator: Validator):
        super(MockRecommender, self).fit(validator)

    def _fit(self, validator: Validator, first_epoch, final_epoch=1000, trial=None):

        for e in range(first_epoch, final_epoch):
            score = self._best_score
            if self._no_improvements < self._early_stopping:
                s = np.random.randint(0, 10, (1,))[0]
                time.sleep(s)
                with torch.no_grad():
                    score = validator.validate(self, 4)
            elif trial is None:  # Skip last iterations as irrelevant
                break

            self._on_epoch_end(trial, score, e)

        if trial is not None:
            self._on_trial_end(trial)

    def predict_all(self, users) -> np.array:
        preds = np.random.random((len(users), len(self.meta.items)))
        return preds

    def set_parameters(self, parameters):
        self.n_layers = parameters.get('layers', 3)
        self.lr = parameters['learning_rates']
        self.dropouts = [parameters['dropouts']] * self.n_layers
        self.gate_type = parameters['gate_types']
        self.aggregator = parameters['aggregators']
        self.autoencoder_weight = parameters['autoencoder_weights']
        self.weight_decay = parameters.get('weight_decays', 1e-5)
        self.n_layer_samples = [10] * self.n_layers
