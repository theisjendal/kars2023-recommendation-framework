import argparse
import random
from collections import defaultdict
from os.path import join
from typing import List, Dict, Tuple, Set

import networkx as nx
from loguru import logger
from tqdm import tqdm

from configuration.datasets import dataset_names, datasets
from shared.configuration_classes import DatasetConfiguration
from shared.entity import Entity
from shared.enums import Sentiment
from shared.relation import Relation
from shared.user import User
from shared.utility import valid_dir, save_entities, load_entities, save_relations, load_relations, save_users, \
    load_users

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=valid_dir, help='Location of datasets', default='..')
parser.add_argument('--dataset', choices=dataset_names, help='Datasets to process')


def prune_users(users: List[User], dataset: DatasetConfiguration, k=10) -> Tuple[List[User], bool]:
    pre_c = len(users)
    users = filter(lambda u: len(u.ratings) >= k, tqdm(users, desc='Pruning users with too few ratings.'))

    # Apply all filters
    for d_filter in dataset.filters:
        rating_type = dataset.sentiment_utility[d_filter.sentiment]
        users = filter(lambda u: d_filter.filter_func(len([i for i, r in u.ratings if r == rating_type])), users)

    users = sorted(users, key=lambda u: u.index)  # Sort by index
    post_c = len(users)

    # Create new indices if users have been removed
    change = bool(pre_c - post_c)
    if change:
        # Reassign index
        for i, u in tqdm(enumerate(users), total=len(users), desc='Reassigning user indices'):
            u.index = i

    return users, change


def prune_items(users: List[User], dataset: DatasetConfiguration, k=10) -> Tuple[List[User], bool]:
    item_count = defaultdict(list)

    for user in tqdm(users, desc='Finding rating count of items'):
        for item, r in user.ratings:
            item_count[item].append(r)

    items = filter(lambda i: len(i[1]) >= k, item_count.items())

    # {i for i, c in item_count.items() if c < k}
    # Apply filters
    for d_filter in dataset.filters:
        rating_type = dataset.sentiment_utility[d_filter.sentiment]
        items = filter(lambda i:  d_filter.filter_func(len([r for r in i[1] if r == rating_type])), items)

    prune_set = set(item_count.keys()).difference(dict(items).keys())
    to_prune = bool(prune_set)
    if to_prune:
        for user in tqdm(users, desc='Removing items'):
            user.ratings = [(i, r) for i, r in user.ratings if i not in prune_set]

    return users, to_prune


def subsample_users(users: List[User], dataset: DatasetConfiguration):
    logger.info(f'Sampling {dataset.max_users} users.')
    if len(users) <= dataset.max_users:
        change_occured = False
    else:
        permutation = list(range(len(users)))
        random.shuffle(permutation)
        permutation = sorted(permutation[:dataset.max_users])
        users = [users[p] for p in permutation]
        change_occured = True

    return users, change_occured


def create_k_core(users: List[User], dataset: DatasetConfiguration, k=10) -> Dict[int, User]:
    change_occured = True
    iter_count = 0
    logger.info('Going to iteratively prune users or items with too few ratings until convergence (or max 50 iter.)')
    while change_occured:
        logger.info(f'Pruning iter: {iter_count}')
        users, u_change = prune_users(users, dataset, k)
        users, i_change = prune_items(users, dataset, k)

        change_occured = u_change or i_change

        # If no items nor users were removed stop loop
        if iter_count >= 50:
            change_occured = False
        else:
            iter_count += 1

        if not change_occured and dataset.max_users is not None:
            users, change_occured = subsample_users(users, dataset)

    users = {user.index: user for user in users}

    for i, user in users.items():
        user.index = i

    return users


def map_ratings(users: List[User], dataset: DatasetConfiguration) -> List[User]:
    unseen = dataset.sentiment_utility[Sentiment.UNSEEN]
    for user in tqdm(users, desc='Mapping user ratings'):
        # Skip non polarized ratings and add rest to ratings
        mapped = [(i, dataset.ratings_mapping(r)) for i, r in user.ratings]
        user.ratings = [(i, r) for i, r in mapped if r != unseen]

    return users


def get_rated_items(users: List[User]) -> Set[int]:
    items = set()

    for user in users:
        items.update([r[0] for r in user.ratings])

    return items


def remove_duplicate_edges(relations: List[Relation]) -> List[Relation]:
    for relation in relations:
        relation.edges = list(set(relation.edges))

    return relations


def num_components(relations: List[Relation]) -> int:
    logger.info('Getting number of connected components')
    edges = []

    for relation in relations:
        edges.extend(relation.edges)

    g = nx.Graph()

    g.add_edges_from(edges)

    return nx.number_connected_components(g)


def prune_relations(entities: Dict[int, Entity], relations: List[Relation]) -> List[Relation]:
    # remove invalid edges
    for relation in relations:
        relation.edges = [(src, dst) for src, dst in relation.edges if src in entities and dst in entities]

    return [r for r in relations if r.edges]  # Remove relations with no edges


def get_n_hop_connections(entities: Dict[int, Entity], relations: List[Relation], n_hops: int)\
        -> Set[int]:
    # The start set is the recommendable entities.
    if n_hops <= 0:
        return {e.index for e in entities.values() if e.recommendable}

    connections = get_n_hop_connections(entities, relations, n_hops-1)

    for relation in relations:
        for src, dst in relation.edges:
            if src in connections:
                connections.add(dst)
            elif dst in connections:
                connections.add(src)

    return connections


def remove_unrated(entities: List[Entity], relations: List[Relation], rated_entities: Set[int]) \
        -> Tuple[List[Entity], List[Relation]]:
    start_len = len(entities)
    # remove item entities without ratings
    entities = filter(lambda e: not e.recommendable or e.index in rated_entities, entities)
    entities = {e.index: e for e in entities}

    relations = prune_relations(entities, relations)

    # Find number of times pointing to or from a recommendable entity.
    connections = get_n_hop_connections(entities, relations, n_hops=2)

    # Ensure entity is connected to at least one rated entity or is recommendable.
    entities = {idx: e for idx, e in entities.items() if e.recommendable or idx in connections}

    relations = prune_relations(entities, relations)

    logger.info(f'Removed {start_len - len(entities)} items from graph due to lack of ratings')

    return list(entities.values()), relations


def prune_entities(entities: List[Entity], relations: List[Relation], rated_entities: Set[int], min_degree: int = 2) \
        -> Tuple[Dict[int, Entity], Dict[int, Relation]]:

    # entities, relations = remove_unrated(entities, relations, rated_entities)
    entities = {e.index: e for e in entities}

    changed = True
    iter_count = 0
    while changed:
        degree = defaultdict(int)
        logger.info(f'Iteratively pruning entities, iter: {iter_count}, n_entities: {len(entities)}')
        # Get degree
        for relation in relations:
            for src, dst in relation.edges:
                degree[src] += 1
                degree[dst] += 1

        l = len(entities)
        entities = {i: e for i, e in entities.items() if e.index in rated_entities or
                    (i in degree and degree[i] >= min_degree)}
        to_prune = bool(l-len(entities))

        if to_prune:
            relations = prune_relations(entities, relations)

            iter_count += 1
        else:
            changed = False

    assert num_components(relations) == 1, 'Some models require single connected component. Implement code adding' \
                                           'entities back into the graph.'

    return entities, {r.index: r for r in relations}


def reindex(entities: Dict[int, Entity], relations: Dict[int, Relation], users: Dict[int, User]) \
        -> Tuple[Dict[int, Entity], Dict[int, Relation], Dict[int, User]]:
    logger.info('Reindexing entities, relations, and users')

    mapping = {e: i for i, e in enumerate(sorted(entities.keys()))}

    # Reindex entities
    for i, e in entities.items():
        e.index = mapping[i]

    entities = {e.index: e for e in entities.values()}

    # Reindex relations
    for relation in relations.values():
        relation.edges = [(mapping[src], mapping[dst]) for src, dst in relation.edges]

    # Reindex users
    for i, user in enumerate(users.values()):
        user.index = i
        user.ratings = [(mapping[item], rating) for item, rating in user.ratings]

    return entities, relations, users


def run(path, dataset):
    dataset = next(filter(lambda d: d.name == dataset, datasets))
    random.seed(dataset.seed)
    in_path = out_path = join(path, dataset.name)
    users = load_users(in_path)
    users = map_ratings(users, dataset)
    users = create_k_core(users, dataset, k=dataset.k_core)

    items = get_rated_items(list(users.values()))

    relations = load_relations(in_path)
    relations = remove_duplicate_edges(relations)

    entities = load_entities(in_path)
    entities, relations = prune_entities(entities, relations, items)
    entities, relations, users = reindex(entities, relations, users)

    logger.info('Saving the entities, relations and users')
    save_users(out_path, users, fname_extension='processed')
    save_entities(out_path, entities, fname_extension='processed')
    save_relations(out_path, relations, fname_extension='processed')


if __name__ == '__main__':
    args = parser.parse_args()
    run(args.path, args.dataset)
