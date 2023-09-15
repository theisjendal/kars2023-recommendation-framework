import os.path

import dgl
import numpy as np
import torch
try:
    from pykeen.pipeline import pipeline
    from pykeen.triples import CoreTriplesFactory
except ModuleNotFoundError:
    print('WARNING: Switch python venv if using complex extractor or install pykeen for current. '
          'Does not guarantee reproducibility.')

from datasets.feature_extractors.feature_extraction_base import FeatureExtractionBase
from shared.configuration_classes import FeatureConfiguration, ExperimentConfiguration
from shared.enums import FeatureEnum
from shared.utility import load_entities


class ComplEXFeatureExtractor(FeatureExtractionBase):
    """
    Traines ComplEX and saves features for entities.
    """
    def __init__(self, model_name='TransE', use_cuda=False):
        super().__init__(use_cuda, 'text')
        self.model_name = model_name

    def extract_features(self, data_path: str, feature_configuration: FeatureConfiguration,
                         experiment: ExperimentConfiguration, cold_experiment: ExperimentConfiguration=None,
                         fold: str = 0) -> np.ndarray:
        entities, = self.load_experiment(data_path, experiment, limit=['entities'])
        graph = self._load_graph(data_path, fold, feature_configuration, experiment)
        u, v, eid = graph.edges('all')
        types = graph.edata['type'][eid]
        triples = torch.stack([u, types, v]).T
        n_entities = torch.max(triples) + 1
        n_relations = torch.max(triples[:, 1]) + 1
        factory = CoreTriplesFactory(triples, n_entities, n_relations, torch.arange(n_entities),
                                     torch.arange(n_relations))

        ratios = [0.8, 0.1, 0.1]
        training_factory, testing_factory, validation_factory = factory.split(ratios, random_state=experiment.seed)
        result = pipeline(training=training_factory, validation=validation_factory, testing=testing_factory,
                          model=self.model_name, epochs=5, random_seed=experiment.seed)

        if self.model_name == 'TransE':
            embeddings = result.model.state_dict()['entity_embeddings._embeddings.weight']
        elif self.model_name == 'complex':
            embeddings = result.model.state_dict()['entity_representations.0._embeddings.weight']
        else:
            raise NotImplementedError()

        if feature_configuration.feature_span == FeatureEnum.ENTITIES:
            features = embeddings
        elif feature_configuration.feature_span == FeatureEnum.DESC_ENTITIES:
            features = embeddings[torch.tensor([e.index for e in entities if not e.recommendable])]
        else:
            raise NotImplementedError()

        return features.cpu().numpy()
