import argparse
import ast
import os

from collections import defaultdict
import pandas as pd
from loguru import logger
from tqdm import tqdm

from shared.entity import Entity
from shared.relation import Relation
from shared.user import User
from shared.utility import valid_dir, save_entities, save_relations, save_users
from datasets.converters.ab_converter_og import prune

parser = argparse.ArgumentParser()
parser.add_argument('--path', nargs=1, type=valid_dir, default='../amazon-book',
                    help='amazon dataset path')


def parse(path):
    try:
        g = open(path, 'r')
        n_lines = int(os.popen(f'wc -l {path}').read().split(' ')[0])
    except OSError:
        raise OSError('Must run ab-converter-og.py before this file.')
    for i, l in tqdm(enumerate(g), desc='Loading reviews', total=n_lines):
        yield ast.literal_eval(l)  # json.loads(l)


def getDF(path, items):
    descriptions = {}
    item_reviews = defaultdict(list)
    for d in parse(path):
        if item := d.get('asin'):
            descriptions[item] = d

    for item in tqdm(items):
        if desc := descriptions.get(item):
            if text := desc.get('title'):
                item_reviews[item].append(text)
            if text := desc.get('description'):
                item_reviews[item].append(text)
    return item_reviews


def create_entities(path):
    logger.info('Creating entities')
    d = {'freebaseId': [], 'remap_id': []}
    with open(os.path.join(path, 'entity_list.txt')) as f:
        f.readline() # Skip headers
        for line in f:
            freebase_id, index = line.rsplit(' ', maxsplit=1)
            d['freebaseId'].append(freebase_id)
            d['remap_id'].append(index)

    items = []
    with open(os.path.join(path, 'item_list.txt')) as f:
        f.readline() # Skip headers
        for line in f.read().splitlines():
            items.append(line.split(' ', maxsplit=2))

    df = pd.DataFrame.from_dict(d).astype({'freebaseId': str, 'remap_id': int})
    df_item = pd.DataFrame.from_records(items, columns=['orgId', 'remap_id', 'freebaseId'])\
        .astype({'freebaseId': str, 'remap_id': int, 'orgId': str})

    df = pd.merge(df, df_item, on='freebaseId', how='left')
    df = df.sort_values(by=['orgId', 'freebaseId'])
    descriptions = getDF(os.path.join(path, 'meta_Books.json'), df[['orgId']].dropna().drop_duplicates()['orgId'].tolist())

    entities = {}

    no_desc = 0
    for index, (_, (freebaseId, remap_id, org_id, _)) in tqdm(enumerate(df.iterrows()), total=len(df)):
        recommendable = not pd.isna(org_id)
        if not recommendable or (description := descriptions.get(org_id)) is not None:
            entities[index] = Entity(index, name=freebaseId, original_id=remap_id,
                                 recommendable=recommendable, description=description)
        else:
            no_desc += 1

    if no_desc:
        logger.warning(f'There are {no_desc} items without any descriptions')

    entities =  {i: e for i, e in enumerate(entities.values())}
    for i, e in entities.items():
        e.index = i

    return entities


def create_relations(path, remap_index):
    logger.info('Creating relations')
    df_kg = pd.read_csv(os.path.join(path, 'kg_final.txt'), header=None, names=['head', 'relation', 'tail'],
                            sep=' ')
    df_kg['head'] = df_kg['head'].apply(remap_index.get)
    df_kg['tail'] = df_kg['tail'].apply(remap_index.get)
    before = len(df_kg)
    df_kg = df_kg.dropna()
    diff = before - len(df_kg)

    if diff:
        logger.warning(f'Removed {diff} edges as entities have been removed.')

    df_relation = pd.read_csv(os.path.join(path, 'relation_list.txt'), sep=' ')
    remap_org = df_relation.set_index('remap_id')['org_id'].to_dict()

    relations = {}
    for group, df in df_kg.groupby('relation'):
        uri = remap_org[group]
        edges = df[['head', 'tail']].to_records(index=False).tolist()
        relations[group] = Relation(group, uri, edges, uri)

    return relations


def create_users(path, remap_id):
    logger.info('Creating users')
    df = pd.read_csv(os.path.join(path, 'user_list.txt'), sep=' ')
    users = {}
    for _, (org_id, index) in tqdm(df.iterrows(), total=len(df), desc='Loading users'):
        users[index] = User(org_id, index)

    with open(os.path.join(path, 'train.txt')) as f_train, open(os.path.join(path, 'test.txt')) as f_test:
        for line in tqdm(f_train.read().splitlines() + f_test.read().splitlines(), desc='Inserting ratings'):
            user, ratings = line.split(' ', maxsplit=1)
            if not ratings:
                continue
            tuples = [(idx, 1) for r in ratings.split(' ') if (idx := remap_id.get(int(r))) is not None]
            users[int(user)].ratings.extend(tuples)

    return users


def run(path):
    data_path = os.path.join(path, 'data')
    entities = create_entities(data_path)
    remap_index = {e.original_id: e.index for e in entities.values()}
    relations = create_relations(data_path, remap_index)
    users = create_users(data_path, remap_index)

    users, entities, relations = prune(users, entities, relations)

    save_entities(path, entities)
    save_relations(path, relations)
    save_users(path, users)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args.path)
