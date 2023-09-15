import argparse
import gzip
import json
import os.path
import shutil
from collections import defaultdict
from enum import IntEnum
from os.path import join, isfile, basename
from typing import Tuple, Dict
from zipfile import ZipFile

from loguru import logger
from tqdm import tqdm
import pandas as pd

from shared.entity import Entity
from shared.relation import Relation
from shared.user import User
from shared.utility import valid_dir, save_entities, save_relations, save_users, beautify

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', nargs=1, type=valid_dir, default='../dbbook',
                    help='path to store downloaded data')
parser.add_argument('--out_path', nargs=1, type=valid_dir, default='../dbbook',
                    help='path to store downloaded data')


def create_entities(path):
    df = pd.read_csv(os.path.join(path, 'entities.csv'))
    df_uri_idx = {uri: idx for idx, uri in df.uri.iteritems()}

    # Load entity mapping and convert to dictionary
    idx_uri = pd.read_csv(os.path.join(path, 'dbbook2014/kg/e_map.dat'), header=None, names=['index', 'uri'],
                        sep='\t', index_col='index').uri.to_dict()
    recommendable = pd.read_csv(os.path.join(path, 'dbbook2014/i2kg_map.tsv'), header=None, sep='\t')[0].to_list()

    entities = {}
    for idx, uri in idx_uri.items():
        df_index = df_uri_idx.get(uri)
        if df_index is not None:
            df_uri, wikidata, wikilink, name, description, comment = df.iloc[df_index]

            # Set description as comment if none
            if type(description) != str:
                description = comment

            # If still None continue
            if type(description) != str:
                description = None

            i = len(entities)
            entities[i] = Entity(i, name, idx in recommendable, original_id=idx, description=description)

    e_beautify = [e for e in entities.values() if e.description is not None]
    beautify(e_beautify)  # Updates entities of in entities dictionary in place.

    return entities


def create_relations(path, entities):
    reverse_idx = {e.original_id: i for i, e in entities.items()}

    base = os.path.join(path, 'dbbook2014/kg')
    df_test = pd.read_csv(os.path.join(base, 'test.dat'), header=None, names=['head', 'tail', 'relation'], sep='\t')
    df_train = pd.read_csv(os.path.join(base, 'train.dat'), header=None, names=['head', 'tail', 'relation'], sep='\t')
    df_valid = pd.read_csv(os.path.join(base, 'valid.dat'), header=None, names=['head', 'tail', 'relation'], sep='\t')

    idx_name = pd.read_csv(os.path.join(base, 'r_map.dat'), header=None, index_col=0, sep='\t')[1].to_dict()
    df = pd.concat([df_test, df_train, df_valid])
    relations = {}
    n_removed = 0
    for index, group in df.groupby('relation'):
        edges = group[['head', 'tail']].applymap(reverse_idx.get)
        n_removed += edges.isnull().values.sum()
        edges = edges.dropna().to_records(index=False).tolist()
        relations[index] = Relation(index, idx_name[index], edges, index)

    logger.warning(f'Removed {n_removed} (~{n_removed/len(df):.2f}%) of {len(df)}  edges as entities not in dataset')

    return relations


def create_users(path, entities):
    base = os.path.join(path, 'dbbook2014')
    df_test = pd.read_csv(os.path.join(base, 'test.dat'), header=None, names=['user', 'item', 'rating'], sep='\t')
    df_train = pd.read_csv(os.path.join(base, 'train.dat'), header=None, names=['user', 'item', 'rating'], sep='\t')
    df_valid = pd.read_csv(os.path.join(base, 'valid.dat'), header=None, names=['user', 'item', 'rating'], sep='\t')
    df = pd.concat([df_test, df_train, df_valid])

    # Map item id to original item id
    item_orig = pd.read_csv(os.path.join(base, 'i_map.dat'), header=None, index_col=0, sep='\t')[1].to_dict()

    # Map from original item id to entity uri.
    orig_uri = pd.read_csv(os.path.join(base, 'i2kg_map.tsv'), header=None, index_col=0, sep='\t')[2].to_dict()

    # Map uri to entity id
    uri_eid = pd.read_csv(os.path.join(base, 'kg/e_map.dat'), header=None, index_col=1, sep='\t')[0].to_dict()

    # Reverse entity mapping
    reverse = {e.original_id: index for index, e in entities.items()}

    # Get mapping from item to new index.
    item_index = {item: index for item, orig in item_orig.items()
                if (uri := orig_uri.get(orig)) and (eid := uri_eid.get(uri)) and
                  (index := reverse.get(eid))}

    n_removed = 0
    users = {}
    for user, ratings in df.groupby('user'):
        ratings['item'] = ratings['item'].apply(item_index.get)
        ratings = ratings[['item', 'rating']]
        n_removed += ratings.isnull().values.sum()
        ratings = ratings.dropna().to_records(index=False).tolist()
        users[user] = User(str(user), user, ratings)

    logger.warning(f'Removed {n_removed} (~{n_removed/len(df):.2f}%) of {len(df)}  interactions as no matching entity')
    return users


def run(path, out):
    logger.info('Creating entities')
    entities = create_entities(path)

    logger.info('Creating relations')
    relations = create_relations(path, entities)

    logger.info('Creating users')
    users = create_users(path, entities)

    logger.info('Saving all')
    save_entities(out, entities)
    save_relations(out, relations)
    save_users(out, users)


if __name__ == '__main__':
    args = parser.parse_args()
    path = args.in_path

    run(args.in_path, args.out_path)
