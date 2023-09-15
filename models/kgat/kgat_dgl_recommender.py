import dgl
import numpy as np
import torch
from dgl.dataloading import EdgeDataLoader
from dgl.dataloading.negative_sampler import Uniform
from loguru import logger
from torch import optim, nn
from tqdm import tqdm

from models.dgl_recommender_base import RecommenderBase
from models.kgat.kgat import KGAT
from models.shared.dgl_dataloader import BPRSampler
from shared.efficient_validator import Validator
from shared.enums import Sentiment
from shared.graph_utility import UniformRecommendableItemSampler
from shared.utility import is_debug_mode


class KGATDGLRecommender(RecommenderBase):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.user_fn = lambda x: x + len(self.meta.entities)
        self.batch_size = 1024
        self.relation_dim = 64
        self.entity_dim = 64
        self.layers = 3
        self.layer_dims = [64, 32, 16]
        self.dropout = 0.1
        self.reg_weight = 1e-5
        self.lr = 0.05

        self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')

    def _create_model(self, trial):
        self.set_seed()
        g, = self.graphs

        n_relations = torch.max(g.edata['type']) + 1
        self._model = KGAT(g, g.number_of_nodes(), n_relations, self.entity_dim, self.relation_dim,
                     self.layers, self.layer_dims, dropout=self.dropout, user_fn=self.user_fn,
                     use_cuda=self.use_cuda)

        # Define optimizer
        self._optimizer = optim.Adam(self._model.parameters(), lr=self.lr)

        if self.use_cuda:
            self._model = self._model.cuda()

        self._sherpa_load(trial)

    def _set_edge_weight(self, g):
        with torch.no_grad():
            if self._model.mode == 'attention':
                g.edata['a'] = self._model.compute_attention(g).pin_memory()
            elif self._model.mode == 'gates':
                g.edata['g'] = self._model.compute_gates(g).pin_memory()

        return g

    def fit(self, validator: Validator):
        super(KGATDGLRecommender, self).fit(validator)

        # Generate embeddings
        self._model.eval()
        with torch.no_grad():
            g, = self.graphs
            g = self._set_edge_weight(g)
            self._model.store_embeddings(g)

    def _fit(self,  validator: Validator, first_epoch, final_epoch=1000, trial=None):
        # Get relevant information
        pos_relation_id = self.infos[0][Sentiment.POSITIVE.name]
        g, = self.graphs

        # Get positive relations.
        mask = g.edata['type'] == pos_relation_id
        eids = g.edges('eid')[mask]

        trans_sampler = BPRSampler()
        trans_sampler = dgl.dataloading.as_edge_prediction_sampler(
            trans_sampler, prefetch_labels=['type'],
            negative_sampler=dgl.dataloading.negative_sampler.Uniform(1)
        )
        trans_dataloader = EdgeDataLoader(
            g, g.all_edges('eid').to(self.device), trans_sampler, batch_size=self.batch_size, shuffle=True,
            drop_last=False, device=self.device, use_uva=self.use_cuda
        )

        # Same as in dgl documentation
        n_edges = g.number_of_edges()
        reverse_eids = torch.cat([torch.arange(n_edges // 2, n_edges), torch.arange(0, n_edges // 2)])

        # Create cf dataloader
        cf_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3, prefetch_edge_feats=['type', 'a'])
        # cf_sampler = dgl.dataloading.as_edge_prediction_sampler(
        #     cf_sampler, exclude='reverse_id', reverse_eids=reverse_eids,
        #     negative_sampler=UniformRecommendableItemSampler(1)
        # )
        cf_dataloader = EdgeDataLoader(
            g, eids.to(self.device), cf_sampler, device=self.device, batch_size=self.batch_size,
            shuffle=True, drop_last=False, use_uva=self.use_cuda, exclude='reverse_id', reverse_eids=reverse_eids,
            negative_sampler=UniformRecommendableItemSampler(1))

        # Define loss
        ls = nn.LogSigmoid()
        trans_r_loss = lambda pos, neg: torch.sum(-ls(neg - pos))
        cf_loss = lambda pos, neg: torch.sum(-ls(pos - neg))

        def reg_loss():
            reg = []
            for param in list(self._model.parameters())[:3]:
                reg.append(torch.norm(param, 2))

            return torch.sum(torch.stack(reg))

        for e in range(first_epoch, final_epoch):
            self._model.train()
            score = self._best_score
            if self._no_improvements < self._early_stopping:
                tot_loss = 0
                progress = tqdm(trans_dataloader, disable=not is_debug_mode())
                for i, (input_nodes, positive_graph, negative_graph, _) in enumerate(progress):
                    negative_graph.edata['type'] = positive_graph.edata['type']  # Neg triples have same relation type
                    pos = self._model.trans_r(positive_graph)
                    neg = self._model.trans_r(negative_graph)

                    reg = self.reg_weight * self._model.l2_loss(positive_graph, negative_graph,
                                                                self._model.entity_embed(input_nodes),
                                                                trans_r=True)
                    loss = trans_r_loss(pos, neg) + reg

                    loss.backward()
                    self._optimizer.step()
                    self._optimizer.zero_grad()

                    tot_loss += loss.detach()
                    progress.set_description(f'Epoch {e}, TLoss: {tot_loss / i:.5f}, RegLoss: {reg:.5f}')

                g = self._set_edge_weight(g)

                tot_loss = 0
                progress = tqdm(cf_dataloader, disable=not is_debug_mode())
                for i, (_, positive_graph, negative_graph, blocks) in enumerate(progress):
                    embeddings = self._model.embedder(blocks)
                    pos_pred = self._model.predict(positive_graph, embeddings)
                    neg_pred = self._model.predict(negative_graph, embeddings)

                    reg = self.reg_weight * self._model.l2_loss(positive_graph, negative_graph, embeddings)
                    loss = cf_loss(pos_pred, neg_pred) + reg

                    loss.backward()
                    self._optimizer.step()
                    self._optimizer.zero_grad()

                    tot_loss += loss.detach()
                    progress.set_description(f'Epoch {e}, CFLoss: {tot_loss / i:.5f}, RegLoss: {reg:.5f}')

                self._model.eval()
                with torch.no_grad():
                    self._model.store_embeddings(g)
                    score = validator.validate(self)
            elif trial is None:
                break

            self._on_epoch_end(trial, score, e)

        if trial is not None:
            self._study.finalize(trial=trial)

    def predict_all(self, users) -> np.array:
        with torch.no_grad():
            user_t = torch.LongTensor(users)
            items = torch.LongTensor(self.meta.items)
            preds = self._model(user_t, items, rank_all=True)

        return preds.cpu().numpy()

    def set_parameters(self, parameters):
        self.lr = parameters['learning_rates']
        self.reg_weight = parameters['weight_decays']
        self.dropout = parameters['dropouts']