import argparse

import os
from typing import Dict

import pandas as pd
from loguru import logger

from shared.entity import Entity
from shared.relation import Relation
from shared.user import User
from shared.utility import valid_dir, beautify, save_entities, save_relations, save_users

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=valid_dir, default='../lastfm',
                    help='path to downloaded data')


def create_entities(path):
    df = pd.read_csv(os.path.join(path, 'entities.csv'))
    dfm = pd.read_csv(os.path.join(path, 'mappings.tsv'), sep='\t', header=None, names=['itemId', 'name', 'uri'])
    items = set(dfm.uri.tolist())

    entities = {}
    for i, (uri, _, _, name, description, comment) in df.iterrows():
        # Set description as comment if none
        if type(description) != str:
            description = comment

        # If still None continue
        if type(description) != str:
            description = None

        entities[i] = Entity(i, name, uri in items, original_id=uri, description=description)

    e_beautify = [e for e in entities.values() if e.description is not None]
    beautify(e_beautify)

    return entities


def create_relations(path, entities: Dict[int, Entity]):
    relations = {}
    uri_id = {e.original_id: eid for eid, e in entities.items()}

    g_path = os.path.join(path, 'graphs')
    for i, file in enumerate(sorted(os.listdir(g_path))):
        df = pd.read_csv(os.path.join(g_path, file), sep=' ', header=None, names=['head', 'tail'])
        edges = df.applymap(uri_id.get)
        edges = edges.dropna().to_records(index=False).tolist()
        relations[i] = Relation(i, file, edges, file)

    return relations


def create_users(path, entities):
    # Get old id to new entity id
    dfm = pd.read_csv(os.path.join(path, 'mappings.tsv'), sep='\t', header=None, names=['itemId', 'name', 'uri'])
    uri_eid = {e.original_id: eid for eid, e in entities.items()}

    # Get feedback
    df = pd.read_csv(os.path.join(path, 'feedback.csv'), sep='\t', header=None, names=['userId', 'itemId', '?'],
                     encoding='latin-1')

    # Get uris pruning all without matches
    df = df.merge(dfm)

    # Remove duplicates and ensure it has entity id.
    df = df[['userId', 'uri']].drop_duplicates()[df.uri.isin(uri_eid)]

    # Get entity Id
    df['eid'] = df.uri.apply(uri_eid.get)

    users = {}
    for name, group in df.groupby('userId'):
        ratings = [(item, 1) for item in group.eid.tolist()]
        uid = len(users)
        users[uid] = User(name, uid, ratings)

    return users


def run(path):
    logger.info('Creating entities')
    entities = create_entities(path)

    logger.info('Creating relations')
    relations = create_relations(path, entities)

    logger.info('Creating users')
    users = create_users(path, entities)

    logger.info('Saving all')
    save_entities(path, entities)
    save_relations(path, relations)
    save_users(path, users)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args.path)