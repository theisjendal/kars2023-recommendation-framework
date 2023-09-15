from typing import List

import dgl
import torch

from shared.enums import FeatureEnum
from shared.user import User


def get_user_item_map(train: List[User]):
    user_map = {u.index: i + 1 for i, u in enumerate(train)}
    items = {i for user in train for i, _ in user.ratings}
    item_map = {item: idx + 1 for idx, item in enumerate(sorted(items))}

    return user_map, item_map


def construct_user_item_heterogeneous_graph(meta, g, self_loop=False, norm='both', global_nodes=False):
        users = torch.LongTensor(meta.users)
        items = torch.LongTensor(meta.items)
        n_entities = len(meta.entities)

        user_ntype = 'user'
        item_ntype = 'item'

        # Get user-item edges while changing user indices.
        u, i = g.edges()
        u = u - n_entities
        ui = u, i
        iu = i, u

        if not self_loop:
            graph_data = {
                (user_ntype, 'ui', item_ntype): ui,
                (item_ntype, 'iu', user_ntype): iu
            }
        else:
            graph_data = {
                (user_ntype, 'ui', item_ntype): ui,
                (item_ntype, 'iu', user_ntype): iu,
                (user_ntype, 'self_user', user_ntype): (users, users),
                (item_ntype, 'self_item', item_ntype): (items, items)
            }

        n_users, n_items = len(meta.users), len(meta.entities)
        if global_nodes:
            graph_data[(user_ntype, 'global_user', user_ntype)] = (torch.full_like(users, len(users)), users)
            graph_data[(item_ntype, 'global_item', item_ntype)] = (torch.full_like(items, n_entities), items)
            n_users += 1
            n_items += 1

        new_g = dgl.heterograph(graph_data, num_nodes_dict={'user': n_users, 'item': n_items})
        new_g.nodes['user'].data['recommendable'] = torch.zeros(new_g.num_nodes('user'), dtype=torch.bool)
        new_g.nodes['item'].data['recommendable'] = torch.ones(new_g.num_nodes('item'), dtype=torch.bool)

        if norm == 'both':
            for etype in new_g.etypes:
                # get degrees
                src, dst = new_g.edges(etype=etype)
                dst_degree = new_g.in_degrees(dst, etype=etype).float()  # obtain degrees
                src_degree = new_g.out_degrees(src, etype=etype).float()

                # calculate norm in eq. 3 of both ngcf and lgcn papers.
                norm = torch.pow(src_degree * dst_degree, -0.5).unsqueeze(1)  # compute norm
                new_g.edges[etype].data['norm'] = norm

        return new_g

def deterministic_feature_getter(features, nodes, meta):
    input_features = {}
    mask = torch.zeros_like(nodes)
    for enum in FeatureEnum:
        if enum in features:
            if enum == FeatureEnum.USERS:
                selection = nodes >= len(meta.entities)
            elif enum == FeatureEnum.ITEMS:
                selection = nodes < len(meta.items)
            elif enum == FeatureEnum.DESC_ENTITIES:
                selection = torch.logical_and(nodes < len(meta.entities), nodes >= len(meta.items))
            elif enum == FeatureEnum.ENTITIES:
                selection = nodes < len(meta.entities)
            else:
                raise NotImplementedError()

            # Ensure features have not been used previously. I.e. items are a subgroup of entities.
            selection = torch.logical_and(torch.logical_not(mask), selection)
            mask = torch.logical_or(mask, selection)
            input_features[enum] = features[enum][selection]

    return input_features