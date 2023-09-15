from os.path import join
from typing import List, Tuple

import dgl
import numpy as np
import os

from shared.configuration_classes import FeatureConfiguration, ExperimentConfiguration
from shared.entity import Entity
from shared.experiments import Dataset
from shared.meta import Meta
from shared.relation import Relation
from shared.user import User
from shared.utility import load_entities, load_users, load_relations, load_pickle


class FeatureExtractionBase:
    def __init__(self, use_cuda, name):
        self._use_cuda = use_cuda
        self.name = name

    # def extract_features(self, graph: dgl.DGLGraph, feature_configuration: FeatureConfiguration, path: str,
    #                      ws_g: dgl.DGLGraph = None, mappings = None) -> np.ndarray:
    #     raise NotImplementedError()

    def extract_features(self, data_path: str, feature_configuration: FeatureConfiguration,
                         experiment: ExperimentConfiguration, cold_experiment: ExperimentConfiguration=None,
                         fold: str = 0) -> np.ndarray:
        raise NotImplementedError()

    def save(self, path, features: np.ndarray):
        np.save(join(path, f'{self.name}_feats.npy'), features)

    @staticmethod
    def _load_graph(dataset_path, fold, feature_configuration, experiment, graph_name=None):
        graph_path = os.path.join(os.path.join(dataset_path, experiment.dataset.name, experiment.name), f'fold_{fold}')
        graph_name = feature_configuration.graph_name if graph_name is None else graph_name
        graph = dgl.load_graphs(os.path.join(graph_path, graph_name + '.dgl'))[0][0]
        return graph

    @staticmethod
    def load_experiment(data_path, experiment, extension='processed', limit: List[str] = None):
        dataset_path = os.path.join(data_path, experiment.dataset.name)
        res = []
        if limit is None or 'entities' in limit:
            res.append(load_entities(dataset_path, extension))
        if limit is None or 'users' in limit:
            res.append(load_users(dataset_path, extension))
        if limit is None or 'relations' in limit:
            res.append(load_relations(dataset_path, extension))
        if limit is None or 'meta' in limit:
            res.append(load_pickle(os.path.join(dataset_path, experiment.name, 'meta.pickle')))
        return res

    @staticmethod
    def _get_mappings(path, experiment, other_experiment):
        # get paths without fold variable.
        e_path, ws_e_path = (os.path.join(path, e.dataset.name) for e in (experiment, other_experiment))
        mappings = {}
        for t in ['user', 'entity']:
            if t == 'user':
                org = load_users(e_path, 'processed')
                ws = load_users(ws_e_path, 'processed')
            else:
                org = load_entities(e_path, 'processed')
                ws = load_entities(ws_e_path, 'processed')

            org_idx = {e.original_id: e.index for e in org}
            mappings[t] = {e.index: org_idx[e.original_id] for e in ws}

        return mappings

    def __str__(self):
        return self.name
