import argparse
import os.path
from os.path import join

from sklearn.preprocessing import StandardScaler

from configuration.experiments import experiment_names
from datasets.feature_extractors.complex_extractor import ComplEXFeatureExtractor
from datasets.feature_extractors.feature_extraction_base import FeatureExtractionBase
from datasets.feature_extractors.graphsage_feature_extractor import GraphSAGEFeatureExtractor
from datasets.feature_extractors.idcf_feature_extractor import IDCFFeatureExtractor
from datasets.feature_extractors.simple_feature_extractor import SimpleFeatureExtractor
from shared.configuration_classes import FeatureConfiguration
from shared.utility import valid_dir, save_numpy, save_pickle, get_experiment_configuration, get_feature_configuration

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=valid_dir, help='path to datasets')
parser.add_argument('--experiment', choices=experiment_names, help='Experiment to generate features from')
parser.add_argument('--other_experiment', default=None, choices=experiment_names,
                    help='For a cold start setting features might need to be based on the warm-start equivalent.'
                         'When scaling or similar will use this instead of experiment. Assumes item and user indices to'
                         'be equivalent. For example for key and query users, where we assume warm-start users to'
                         'be key and cold-start users to be query.')
parser.add_argument('--feature_configuration', default='graphsage', help='feature configuration name')
parser.add_argument('--cuda', action='store_true', help='use cuda, default false')


def run_feature_extraction(data_path, feature_extractor: FeatureExtractionBase,
                           feature_configuration: FeatureConfiguration, fold, experiment, other_experiment=None):

    features = feature_extractor.extract_features(data_path, feature_configuration, experiment, other_experiment, fold)

    scaler = None
    if feature_configuration.scale:
        scaler = StandardScaler().fit(features)
        features = scaler.transform(features)

    return features, scaler


def get_feature_extractor(feature_configuration: FeatureConfiguration, use_cuda):
    if feature_configuration.extractor == 'simple':
        extractor = SimpleFeatureExtractor
    elif feature_configuration.extractor == 'graphsage':
        extractor = GraphSAGEFeatureExtractor
    elif feature_configuration.extractor == 'idcf':
        extractor = IDCFFeatureExtractor
    elif feature_configuration.extractor == 'complex':
        extractor = ComplEXFeatureExtractor
    else:
        raise NotImplementedError()

    return extractor(use_cuda=use_cuda, **feature_configuration.kwargs)


def save(path, features, scaler, feature_name):
    # If no cold-start items, features can be scaled using this method.
    if scaler is not None:
        save_pickle(join(path, f'feature_{feature_name}_meta.pickle'), {'scaler': scaler})

    save_numpy(join(path, f'features_{feature_name}.npy'), features)


def run(path, experiment_name, f_conf_name, cuda, other_experiment=None):
    experiment = get_experiment_configuration(experiment_name)
    if other_experiment is not None:
        other_experiment = get_experiment_configuration(other_experiment)
    else:
        other_experiment = None

    f_conf = get_feature_configuration(f_conf_name)
    extractor = get_feature_extractor(f_conf, cuda)

    # If extractor requires ratings, it needs to iterate all folds and the features are store under each fold.
    # If not, we just use the first fold (range(1) ~= [0]).
    out_path = os.path.join(path, experiment.dataset.name, experiment.name)
    for i in range(experiment.folds if f_conf.require_ratings else 1):
        features, scaler = run_feature_extraction(path, extractor, f_conf, i, experiment, other_experiment)

        # If we do not require ratings save to experiment folder instead of fold folder (in experiment).
        out_path_i = out_path if not f_conf.require_ratings else os.path.join(out_path, f'fold_{i}')
        save(out_path_i, features, scaler, f_conf.name)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args.path, args.experiment, args.feature_configuration, args.cuda, args.other_experiment)
