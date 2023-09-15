import argparse
from os.path import join
from typing import List, Dict

import numpy as np
from sklearn.preprocessing import StandardScaler

from configuration.experiments import experiment_names
from datasets.feature_extractors import feature_extractor
from datasets.feature_extractors.feature_extraction_base import FeatureExtractionBase
from shared.configuration_classes import FeatureConfiguration, ExperimentConfiguration
from shared.enums import FeatureEnum
from shared.utility import valid_dir, get_feature_configuration, get_experiment_configuration

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=valid_dir, help='path to datasets')
parser.add_argument('--experiment', choices=experiment_names, help='Experiment to generate features from')
parser.add_argument('--feature_configurations', nargs=3, help='feature configuration names. Combines the first and '
                                                                'second feature configuration and saves using the '
                                                                'thirds name. Note features must be extracted for the'
                                                                'two first prior to running this program.')


def load(path, experiment: ExperimentConfiguration, confs: List[FeatureConfiguration]) \
        -> Dict[FeatureConfiguration, np.ndarray] :
    features = {}
    for conf in confs:
        in_path = join(path, experiment.dataset.name, experiment.name, f'features_{conf.name}.npy')
        features[conf] = np.load(in_path)
    return features


def run(path, experiment: ExperimentConfiguration, conf_a: FeatureConfiguration, conf_b: FeatureConfiguration,
        conf_out: FeatureConfiguration):
    features = load(path, experiment, [conf_a, conf_b])
    meta, = FeatureExtractionBase.load_experiment(path, experiment, limit=['meta'])

    if conf_out.feature_span == FeatureEnum.ENTITIES:
        dim = sum([feat.shape[-1] for feat in features.values()])
        new_features = np.zeros((len(meta.entities), dim))
        start = 0
        for conf, feature in sorted(features.items(), key=lambda x: (x[0].feature_span, x[0].name)):
            end = start + feature.shape[-1]
            if conf.feature_span == FeatureEnum.ITEMS:
                new_features[:len(meta.items), start:end] = feature
            elif conf.feature_span == FeatureEnum.DESC_ENTITIES:
                new_features[len(meta.items):, start:end] = feature
            else:
                raise NotImplementedError()

            start = end
    else:
        raise NotImplementedError()

    scaler = None
    if conf_out.scale:
        scaler = StandardScaler().fit(new_features)
        new_features = scaler.transform(new_features)

    feature_extractor.save(join(path, experiment.dataset.name, experiment.name), new_features, scaler, conf_out.name)


if __name__ == '__main__':
    args = parser.parse_args()
    experiment = get_experiment_configuration(args.experiment)
    conf_a, conf_b, conf_out = [get_feature_configuration(c) for c in args.feature_configurations]
    run(args.path, experiment, conf_a, conf_b, conf_out)