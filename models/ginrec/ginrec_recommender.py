import dgl
import numpy as np
import torch
from dgl.dataloading import EdgeDataLoader
from loguru import logger
from torch import optim, nn
from tqdm import tqdm

from models.dgl_recommender_base import RecommenderBase
from models.ginrec.ginrec import GInRec
from shared.efficient_validator import Validator
from shared.enums import Sentiment, FeatureEnum
from shared.graph_utility import UniformRecommendableItemSampler
from shared.meta import Meta
from shared.utility import is_debug_mode


class GInRecRecommender(RecommenderBase):
    def __init__(self, meta: Meta, attention=None, trainable_users=False, relations=True, activation=True, vae=False,
                 bipartite=False, batch_size=128, **kwargs):
        features = kwargs.pop('features', True)  # Default is true
        kwargs['use_cuda'] = True
        super().__init__(meta, **kwargs, features=features)
        self.attention = attention
        self.trainable_user = trainable_users
        self.relations = relations
        self.n_entities = len(meta.entities)
        self.user_fn = lambda u: u + self.n_entities
        self.n_users = len(meta.users)
        self.activation = activation
        self.vae = vae
        self.bipartite = bipartite
        self.batch_size = batch_size
        self.lr = 0.001
        self.weight_decay = 1e-5
        self.n_layers = 3
        self.kl_weight = 0.1
        self.autoencoder_layer_dims = [128, 32]
        self.layer_dims = [32, 16, 8, 8, 8]
        self.autoencoder_weight = 0.001
        self.dropouts = [0.1, 0.1, 0.1]
        self.gate_type = 'concat'
        self.aggregator = 'gcn'
        self.n_layer_samples = [10, 10, 10]
        self.feature_dim = None
        self.attention_dim = None if self.attention is None else 32

        self.optimizer = None
        self.entity_embeddings = None
        self.user_embeddings = None
        self.multi_features = False

    def _get_features(self):
        if self.require_features:
            e = self.entity_embeddings
        else:
            logger.warning('Using learned entity embeddings')
            e = self._model.entity_embeddings

        if self.trainable_user:
            logger.warning('Using learned user embeddings')
            u = self._model.user_embeddings
        else:
            u = self.user_embeddings

        if self.multi_features:
            f = self.entity_embeddings
            if (e := f.get(FeatureEnum.DESC_ENTITIES)) is not None:
                f[FeatureEnum.DESC_ENTITIES] = torch.cat([e, u])
            else:
                raise NotImplementedError()
        else:
            f = torch.cat([e, u])

        return f

    def _create_model(self, trial):
        self.set_seed()
        n_entities = len(self.meta.entities)
        n_users = len(self.meta.users)
        n_relations = torch.max(self.graphs[-1].edata['type']) + 1 if self.relations else 1

        tanh_range = True
        if self.require_features:
            # Load features and add users.
            if not isinstance(self._features, dict):
                self.feature_dim = self._features.shape[-1]
                self.entity_embeddings = torch.FloatTensor(np.array(self._features))
                self.multi_features = False

                # if features in range -1 to 1
                tanh_range = tanh_range and self.entity_embeddings.min() >= -1 and self.entity_embeddings.max() <= 1
            else:
                self.multi_features = True
                features = {}
                self.feature_dim = {}
                for fnum, feature in self._features.items():
                    f = torch.FloatTensor(np.array(feature))
                    features[fnum] = f
                    self.feature_dim[fnum.name] = f.size(-1)

                    # if all features are in range -1 to 1
                    tanh_range = tanh_range and f.min() >= -1 and f.max() <= 1

                self.entity_embeddings = features
        else:
            self.feature_dim = 256
            entity_features = torch.empty(n_entities, self.feature_dim)
            nn.init.xavier_uniform_(entity_features)
            self.entity_embeddings = nn.Parameter(entity_features, requires_grad=True)

        user_dim = 0
        if isinstance(self.feature_dim, dict):
            if dim := self.feature_dim.get(FeatureEnum.DESC_ENTITIES.name):
                user_dim = dim
            else:
                raise NotImplementedError()
        else:
            user_dim = self.feature_dim
        users_features = torch.zeros(n_users, user_dim,requires_grad=False)
        if self.trainable_user:
            nn.init.xavier_uniform_(users_features)
            users_features = nn.Parameter(users_features, requires_grad=True)

        self.user_embeddings = users_features

        self._model = GInRec(n_relations, self.feature_dim, self.user_fn, device=self.device,
                             autoencoder_layer_dims=self.autoencoder_layer_dims, dropouts=self.dropouts,
                             gate_type=self.gate_type, aggregator=self.aggregator, attention=self.attention,
                             attention_dim=self.attention_dim, relations=self.relations,
                             dimensions=self.layer_dims[:self.n_layers], activation=self.activation, vae=self.vae,
                             kl_weight=self.kl_weight, tanh_range=tanh_range)

        if not self.require_features:
            self._model.register_parameter('entity_embeddings', self.entity_embeddings)

        if self.trainable_user:
            self._model.register_parameter('user_embeddings', self.user_embeddings)

        self._optimizer = optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.use_cuda:
            self._model = self._model.cuda()
            # self.user_embeddings = self.user_embeddings.cuda()
            # self.entity_embeddings = self.entity_embeddings.cuda()

        self._sherpa_load(trial)

    def _get_graph(self):
        pos_relation_id = self.infos[0][Sentiment.POSITIVE.name]
        g, = self.graphs

        if self.bipartite:
            pos_rev = pos_relation_id + len(self.infos[0]) // 2
            mask = torch.logical_or(g.edata['type'] == pos_relation_id, g.edata['type'] == pos_rev)
            eids = g.edges('eid')[mask]
            g = g.edge_subgraph(eids, preserve_nodes=True)

        return g

    def fit(self, validator: Validator):
        super(GInRecRecommender, self).fit(validator)

        with torch.no_grad():
            self._model.eval()
            features = self._get_features()
            if self.multi_features:
                features = {k.name: v.to(self.device) for k, v in features.items()}
            else:
                features = features.to(self.device)

            g = self._get_graph()
            self._model.inference(g, features, batch_size=self.batch_size)
            logger.info(validator.validate(self))

    def _fit(self, validator: Validator, first_epoch, final_epoch=1000, trial=None):
        # # Get relevant information
        pos_relation_id = self.infos[0][Sentiment.POSITIVE.name]
        g = self._get_graph()
        features = self._get_features()

        if self.multi_features:
            features = {k.name: v.to(self.device) for k, v in features.items()}
        else:
            features = features.to(self.device)

        # Get positive relations.
        mask = g.edata['type'] == pos_relation_id
        eids = g.edges('eid')[mask]

        # In graph creation reverse relation are added in same order after all original relations. Reverse eids can
        # therefore be defined as in DGL documentation.
        n_edges = g.number_of_edges()
        reverse_eids = torch.cat([torch.arange(n_edges // 2, n_edges), torch.arange(0, n_edges // 2)])

        # Create cf dataloader
        cf_sampler = dgl.dataloading.MultiLayerNeighborSampler(self.n_layer_samples, prefetch_edge_feats=['type'])

        dataloader = EdgeDataLoader(
            g, eids.to(self.device), cf_sampler, device=self.device, exclude='reverse_id', reverse_eids=reverse_eids,
            negative_sampler=UniformRecommendableItemSampler(1), batch_size=self.batch_size,
            shuffle=True, drop_last=False, use_uva=self.use_cuda)
        n_entities = len(self.meta.entities)
        n_iter = len(dataloader)
        for e in range(first_epoch, final_epoch):
            self._model.train()
            score = self._best_score
            if self._no_improvements < self._early_stopping:
                tot_losses = 0
                tot_ae_loss = 0
                tot_correct = 0
                with tqdm(dataloader, disable=not is_debug_mode()) as progress:
                    for i, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(progress, 1):
                        input_features, ae_loss = self._model.loss_ae(features, input_nodes)

                        cf_loss, correct = self._model.loss(positive_graph, negative_graph,
                                                                     blocks, input_features)

                        ae_loss = self.autoencoder_weight * ae_loss
                        loss = cf_loss + ae_loss

                        loss.backward()

                        self._optimizer.step()
                        self._optimizer.zero_grad()

                        cf_loss = cf_loss.detach()
                        ae_loss = ae_loss.detach()
                        acc = (torch.sum(correct) / correct.size(-1)).detach()

                        tot_losses += cf_loss
                        tot_ae_loss += ae_loss
                        tot_correct += acc
                        progress.set_description(f'Epoch {e}, CFL: {tot_losses / i:.5f}, '
                                                 f'AEL: {tot_ae_loss / i:.5f}, '
                                                 f'ACC: {tot_correct / i:.5f}')
                        step = e * n_iter + i
                        self.summary_writer.add_scalar('cf_loss', cf_loss, step)
                        self.summary_writer.add_scalar('ae_loss', ae_loss, step)
                        self.summary_writer.add_scalar('acc', acc, step)

                self._model.eval()
                with torch.no_grad():
                    self._model.inference(g, features, batch_size=self.batch_size)
                    score = validator.validate(self)
                    self.summary_writer.add_scalar('ndcg', score, e)
            elif trial is None:  # Skip last iterations as irrelevant
                break

            self._on_epoch_end(trial, score, e)

        if trial is not None:
            self._study.finalize(trial=trial)

    def predict_all(self, users) -> np.array:
        with torch.no_grad():
            user_t = torch.LongTensor(users) + len(self.meta.entities)
            items = torch.LongTensor(self.meta.items)
            preds = self._model(user_t, items, rank_all=True, apply_user_fn=False)

        return preds.cpu().numpy()

    def set_parameters(self, parameters):
        self.n_layers = parameters.get('layers', 3)
        self.lr = parameters['learning_rates']
        self.dropouts = [parameters['dropouts']] * self.n_layers
        self.gate_type = parameters['gate_types']
        self.aggregator = parameters['aggregators']
        self.autoencoder_weight = parameters['autoencoder_weights']
        self.weight_decay = parameters.get('weight_decays', 1e-5)
        self.n_layer_samples = [10] * self.n_layers
        self.kl_weight = parameters.get('kl_weights', 0.1)
