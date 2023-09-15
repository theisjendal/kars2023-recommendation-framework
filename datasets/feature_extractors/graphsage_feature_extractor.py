from typing import List

import dgl
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from nltk import tokenize


from datasets.feature_extractors.feature_extraction_base import FeatureExtractionBase
from datasets.feature_extractors.utility import extract_degree
from shared.configuration_classes import FeatureConfiguration, ExperimentConfiguration
from shared.entity import Entity
from shared.enums import FeatureEnum
from shared.utility import load_entities


class GraphSAGEFeatureExtractor(FeatureExtractionBase):
    """
    Extracts features similar to the method described in GraphSAGE
    """
    def __init__(self, attributes, use_cuda=False):
        super().__init__(use_cuda, 'text')
        self._model = SentenceTransformer('stsb-roberta-base')
        self.attributes = attributes
        self.text_dim = 768
        self._model.eval()

        if self._use_cuda:
            self._model.to('cuda')

    def _text_to_vec(self, text: List[str]):
        with torch.no_grad():
            # Get embeddings
            sentence_embedding = self._model.encode(text)

        return sentence_embedding  # Ensure numpy for later processing

    def flatten(self, s):
        if isinstance(s, str):
            return [s]
        elif len(s) == 0:
            return []
        elif isinstance(s[0], list):
            return self.flatten(s[0]) + self.flatten(s[1:])
        else:
            return s[:1] + self.flatten(s[1:])

    def _get_text(self, entity: Entity):
        text = [getattr(entity, a) for a in self.attributes]
        text = self.flatten(text)
        text = list(filter(lambda x: bool(x), text))

        grouped_text = []
        for t in text:
            grouped_text.append(tokenize.sent_tokenize(t))

        return grouped_text

    # def extract_features(self, graph: dgl.DGLGraph, feature_configuration: FeatureConfiguration, path: str,
    #                      ws_g: dgl.DGLGraph = None, mappings=None) -> np.ndarray:
    def extract_features(self, data_path: str, feature_configuration: FeatureConfiguration,
                         experiment: ExperimentConfiguration, cold_experiment: ExperimentConfiguration = None,
                         fold: str = 0) -> np.ndarray:
        entities, = self.load_experiment(data_path, experiment, limit=['entities'])
        graph = self._load_graph(data_path, fold, feature_configuration, experiment)
        n_entities = len(entities)

        # Length is the text dim, number of entity types as one hot, and node degree (undirected)
        feature_length = self.text_dim + 1

        features = np.zeros((n_entities, feature_length), dtype=np.float)
        batch_size = 128
        batches = [entities[b*batch_size:(b+1)*batch_size] for b in range(len(entities)//batch_size+1)]

        for i, batch in tqdm(enumerate(batches), total=len(batches), desc='Extracting features from entities'):
            texts = []
            sum_counts = []
            inner_features = np.zeros((len(batch), feature_length))
            for j, entity in enumerate(batch):
                # Get text
                text_list = self._get_text(entity)

                # flatten
                sum_counts.append([len(tx) for tx in text_list])
                texts.extend([t for ts in text_list for t in ts])
                inner_features[j, -1] = graph.out_degrees(entity.index)

            text_features = self._text_to_vec(texts)
            index = 0
            for i, counts in enumerate(sum_counts):
                f = np.zeros((len(counts), self.text_dim))
                for j, count in enumerate(counts):
                   f[j] = np.sum(text_features[index:index+count], axis=0)
                   index += count

                inner_features[i, :self.text_dim] = np.sum(f, axis=0)

            features[[entity.index for entity in batch]] = inner_features

        if feature_configuration.feature_span == FeatureEnum.ITEMS:
            features = features[[entity.index for entity in entities if entity.recommendable]]
        elif feature_configuration.feature_span == FeatureEnum.DESC_ENTITIES:
            features = features[[entity.index for entity in entities if entity.recommendable]]
        elif feature_configuration.feature_span == FeatureEnum.ENTITIES:
            pass  # Already limited to entities
        else:
            raise NotImplementedError()

        return features
