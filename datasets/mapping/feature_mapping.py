import argparse

import os

import numpy as np

from configuration.datasets import dataset_names
from shared.utility import valid_dir, get_dataset_configuration, load_entities, save_entities

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=valid_dir, help='path to datasets')
parser.add_argument('--dataset_a', choices=dataset_names, help='datasets to map to dataset b')
parser.add_argument('--dataset_b', choices=dataset_names, help='datasets to used as reference for dataset a')
parser.add_argument('--fname_extension_a', help='which entity version to use for dataset a')
parser.add_argument('--fname_extension_b', help='which entity version to use for dataset b')
parser.add_argument('--fname_extension_a_out', help='output name')


def run(path, dataset_a, dataset_b, fname_extension_a, fname_extension_b, fname_extension_a_out):
    # Get datasets
    # dataset_a = get_dataset_configuration(dataset_a)
    # dataset_b = get_dataset_configuration(dataset_b)

    # dataset one 2 dataset 2 mapping

    # Feature type of one and 2
    # a -> b, aka we store a's features as matching b's indices.

    da_path = os.path.join(path, dataset_a)
    db_path = os.path.join(path, dataset_b)
    # Load features of both datasets
    entities_a = load_entities(da_path, fname_extension=fname_extension_a)
    entities_b = load_entities(db_path, fname_extension=fname_extension_b)

    orig_e_a = {e.original_id: e for e in entities_a}

    features = np.load(os.path.join(da_path, f'features_{fname_extension_a}.npy'))

    indices = [orig_e_a[e.original_id].index for e in entities_b]

    if indices == list(range(len(entities_b))):
        print('Entity order is identical, no change will occur.')
        if [e.recommendable for e in entities_a] == [e.recommendable for e in entities_b]:
            save_entities(db_path, {e.index: e for e in entities_a}, fname_extension=fname_extension_a)

        name = f'features_{fname_extension_a}.npy'
    else:
        features = features[indices]
        name = f'features_{dataset_a}_{fname_extension_a}.npy'

    np.save(os.path.join(db_path, name), features)

if __name__ == '__main__':
    args = parser.parse_args()
    args = vars(args)
    run(**args)