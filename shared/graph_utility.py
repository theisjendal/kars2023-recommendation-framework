from collections import defaultdict
from typing import List, Tuple

import dgl
import torch
import numpy as np
import dgl.backend as F
from dgl.dataloading.negative_sampler import _BaseNegativeSampler
from loguru import logger

from shared.enums import Sentiment
from shared.meta import Meta
from shared.relation import Relation
from shared.user import User


def dgl_homogeneous(meta: Meta, relations: List[Relation] = None, ratings: List[Tuple[int, int, int]] = None,
                    store_edge_types=False, rating_times=None, **kwargs) -> dgl.DGLHeteroGraph:
    """
    Build a dgl homogeneous graph using relations and ratings. Assumes entity and user ids are mapped to non intersecting values.
    :param meta: the meta data.
    :param relations: if parsed to method, builds a graph with relations.
    :param ratings: if parsed to method, builds a graph with ratings.
    :param kwargs: arguments parsed to dgl.
    :return: a dgl graph based on tensors.
    """
    edges = []
    edge_types = []

    if relations is not None:
        for i, relation in enumerate(sorted(relations, key=lambda x: x.index)):
            edges.extend([(head, tail) for head, tail in relation.edges])

            if store_edge_types:
                edge_types.extend([i] * len(relation.edges))

    if ratings is not None:
        positive_rating = meta.sentiment_utility[Sentiment.POSITIVE]
        edges.extend([(user, item) for user, item, rating in ratings if rating is positive_rating])

    if not edges:
        logger.warning('Creating empty graph, No edges as neither relations nor ratings have been passed.')
        return dgl.graph([])

    edges = torch.LongTensor(edges)

    # Remove duplicate edges if we do not use edge types.
    if not store_edge_types:
        edges = torch.unique(edges, dim=0)

    edges = edges.T

    g = dgl.graph((edges[0], edges[1]), **kwargs)

    recommendable = torch.full_like(g.nodes(), False, dtype=torch.bool)
    recommendable[meta.items] = True
    g.ndata['recommendable'] = recommendable

    if rating_times is not None:
        edata = []
        for s, d in zip(*[t.tolist() for t in g.edges()]):
            if (s, d) in rating_times:
                edata.append(rating_times[(s, d)])
            elif (d, s) in rating_times:
                edata.append(rating_times[(d, s)])
            else:
                edata.append(0)

        edata = torch.tensor(edata)
        g.edata['rating_time'] = edata

    if store_edge_types:
        g.edata['type'] = torch.LongTensor(edge_types).T

    return g


def create_reverse_relations(relations: List[Relation]):
    n_relations = len(relations)
    out_relations = []

    for relation in relations:
        out_relations.append(Relation(relation.index + n_relations, relation.name + '_R',
                                  edges=[(e2, e1) for e1, e2 in relation.edges]))

    return out_relations


def create_rating_relations(meta: Meta, train: List[User], user_fn, sentiments: List[Sentiment] = None):
    n_relations = len(meta.relations)
    relations = []

    train = np.array([[user_fn(user.index), item, rating] for user in train for item, rating in user.ratings]).T

    # Remove unseen ratings
    unseen = meta.sentiment_utility[Sentiment.UNSEEN]
    train = train[:, train[-1] != unseen]

    # For all relevant sentiments add relation.
    for sentiment in sentiments:
        rating_type = meta.sentiment_utility[sentiment]

        mask = train[-1] == rating_type
        data = train[:, mask]

        edges = [(user, item) for user, item, _ in data.T]
        relations.append(Relation(len(relations) + n_relations, sentiment.name, edges))

    return relations


class UniformRecommendableItemSampler(_BaseNegativeSampler):
    def __init__(self, k, edges=None, use_pr=False, methodology='uniform'):
        self.k = k
        self.use_pr = use_pr
        self.edges = edges
        self.dst_nodes = None
        assert methodology in ['uniform', 'popularity', 'non-rated']
        self.methodology = methodology
        if edges is not None:
            dictionary = {}
            src, dst = (e.numpy() for e in edges)
            unique, count = np.unique(src, return_counts=True)
            id_ifc = {nid: idx for idx, nid in enumerate(unique)}
            i = 0
            while i < len(src):
                nid = src[i]
                idx = id_ifc[nid]
                c = count[idx]
                dictionary[nid] = torch.tensor(dst[i:i+c], dtype=edges[0].dtype)
                i += c

            self.edges = dictionary
            self.dst_nodes = torch.unique(edges[1])

    def _generate(self, g, eids, canonical_etype):
        _, _, vtype = canonical_etype
        shape = F.shape(eids)
        dtype = F.dtype(eids)
        ctx = F.context(eids)
        shape = (shape[0] * self.k,)
        src_org, _ = g.find_edges(eids, etype=canonical_etype)
        src = F.repeat(src_org, self.k, 0)

        if len(g.ntypes) > 1:
            selection = g.ndata['recommendable'][vtype]
        else:
            selection = g.ndata['recommendable']

        if self.edges is None or self.methodology == 'uniform':
            recommendable = g.nodes(ntype=vtype)[selection]
            indices = F.randint(shape, dtype, ctx, 0, len(recommendable))
            dst = recommendable[indices]
        elif self.methodology == 'popularity':
            probability = torch.ones((len(src_org), len(selection)))
            probability[:] = selection.float()
            for i, node in enumerate(src_org):
                probability[i, self.edges[node.item()]] = 0

            indices = probability.multinomial(num_samples=self.k, replacement=True)
            dst = g.nodes(ntype=vtype)[indices].flatten()
        elif self.methodology == 'non-rated':
            recommendable = g.nodes(ntype=vtype)[selection]
            indices = F.randint(shape, dtype, ctx, 0, len(recommendable))
            dst = recommendable[indices]
            for i, node in enumerate(src_org):
                while dst[i] in self.edges[node.item()]:
                    dst[i] = recommendable[F.randint((1,), dtype, ctx, 0, len(recommendable))]

        else:
            raise NotImplementedError

        return src, dst.to(src.device)
