from typing import List

import dgl
import torch
import torch.nn as nn
import dgl.function as fn
from dgl import ops
from dgl.heterograph import DGLBlock
from dgl.utils import expand_as_pair

from models.shared.modules import Parallel
from shared.enums import FeatureEnum


class Autoencoder(nn.Module):
    def __init__(self, entity_dim, layer_dims, use_activation=True, use_vae=False, kl_weight=0.1, tanh_range=False):
        super().__init__()
        n_layers = len(layer_dims)
        self.use_activation = use_activation
        self.use_vae = use_vae
        self.kl_weight = kl_weight
        self.tanh_range = tanh_range

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        out_dim = entity_dim

        for layer in range(n_layers):
            in_dim = out_dim
            out_dim = layer_dims[layer]

            self.encoders.append(nn.Linear(in_dim, out_dim))

            if layer + 1 == n_layers and use_vae:
                self.decoders.append(nn.Linear(out_dim // 2, in_dim))
            else:
                self.decoders.append(nn.Linear(out_dim, in_dim))

        if self.use_vae:
            fc_mu = nn.Linear(out_dim, out_dim // 2)
            fc_logvar = nn.Linear(out_dim, out_dim // 2)
            self.encoders.append(Parallel(torch.cat, fc_mu, fc_logvar, dim=1))

        self.decoders = nn.ModuleList(reversed(self.decoders))

        self.activation = nn.ReLU()
        self.out_activation = nn.Tanh()

        self.loss_fn = nn.MSELoss(reduction='none')

    def reparameterize(self, mulogvar):
        half = mulogvar.shape[1] // 2
        mu, logvar = mulogvar[:, :half], mulogvar[:, half:]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def propagate(self, embeddings, mode):
        if mode == 'encode':
            iterator = self.encoders
        elif mode == 'decode':
            iterator = self.decoders
        else:
            raise ValueError('Invalid mode')

        length = len(iterator)
        for i, layer in enumerate(iterator, 1):
            embeddings = layer(embeddings)

            if self.use_activation:
                if mode == 'decode' and i == length:
                    if self.tanh_range:
                        embeddings = self.out_activation(embeddings)
                    else:
                        embeddings = embeddings
                else:
                    embeddings = self.activation(embeddings)

        return embeddings

    def forward(self, embeddings):
        encoded = self.propagate(embeddings, 'encode')
        if self.use_vae:
            decoded = self.propagate(self.reparameterize(encoded), 'decode')
        else:
            decoded = self.propagate(encoded, 'decode')
        return encoded, decoded

    def loss(self, target, decoded, encoded):
        loss = self.loss_fn(decoded, target)

        if self.use_vae:
            half = encoded.shape[1] // 2
            mu, logvar = encoded[:, :half], encoded[:, half:]
            loss += self.kl_weight * torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

        return loss


class SimpleRecConv(nn.Module):
    def __init__(self, input_dim, output_dim, activation=True, normalize=False, softmax=False, gate_fn=nn.Sigmoid(),
                 update='', gate_type='concat', aggregator='graphsage', attention=None, attention_dim=None,
                 relations=True):
        super().__init__()
        self.softmax = softmax
        self.activation = activation
        self.normalize = normalize
        self.activation_fn = nn.LeakyReLU()
        self.relations = relations
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.gate_fn = gate_fn
        if gate_type in ['concat', 'inner_product', None]:
            self.gate_type = gate_type
        else:
            raise ValueError('Unsupported gate type.')

        self.attention_type = attention
        if attention_dim is None != attention is None:
            raise ValueError('Either pass no attention or attention dim or pass both')
        elif attention == 'gatv2':
            self.W_a = nn.Linear(input_dim*2, attention_dim)
            self.attention_fn = nn.LeakyReLU()

        if aggregator in ['gcn', 'graphsage', 'bi-interaction', 'lightgcn', 'lstm']:
            self.aggregator = aggregator
        else:
            raise ValueError('Unsupported aggregator type.')

        if aggregator in ['gcn', 'graphsage', 'lstm']:
            if aggregator == 'lstm':
                self.lstm = nn.LSTM(input_dim, input_dim, batch_first=True)

            # Update input dim
            input_dim = input_dim * 2 if aggregator in ['graphsage', 'lstm'] else input_dim
            self.linear = nn.Linear(input_dim, output_dim)

        elif aggregator in ['bi-interaction']:
            self.W_1 = nn.Linear(input_dim, output_dim)
            self.W_2 = nn.Linear(input_dim, output_dim)

        self.update = update

    def gates(self, relations):
        def func(edges):
            object_emb = edges.src['h']
            predicates = edges.data['type'] if self.relations else torch.zeros_like(edges.data['type'])
            subject_emb = edges.dst['h']
            if self.gate_type == 'concat':
                # gates = torch.bmm(torch.cat((subject_emb, object_emb), dim=-1).unsqueeze(1), relations[predicates])\
                #     .squeeze(1)
                gates = torch.einsum('bd,bdc->bc', torch.cat((subject_emb, object_emb), dim=-1),relations[predicates])
            else:
                subject_emb = torch.bmm(subject_emb.unsqueeze(1), relations[predicates]).squeeze(1)
                object_emb = torch.bmm(object_emb.unsqueeze(1), relations[predicates]).squeeze(1)
                gates = torch.mul(subject_emb, object_emb).sum(dim=-1)
            gates = self.gate_fn(gates)
            return {'gate': gates}

        return func

    def attention(self, relation_attention):
        def func(edges):
            object_emb = edges.src['h']
            predicates = torch.zeros_like(edges.data['type'])
            subject_emb = edges.dst['h']

            a = self.attention_fn(self.W_a(torch.cat((subject_emb, object_emb), dim=-1)))
            a = (relation_attention[predicates] * a).sum(-1)

            return {'attention': a}

        return func

    def lstm_reducer(self, msg, out):
        def func(nodes):
            # Random order of edges
            m = nodes.mailbox[msg]
            m = m[:, torch.randperm(m.shape[1])]

            return {out: self.lstm(m)[-1][0].squeeze(0)}  # Run lstm and select last hidden state

        return func

    def forward(self, g: dgl.DGLGraph, h: torch.Tensor, r: torch.Tensor, a: torch.Tensor=None):
        with g.local_scope():
            g.srcdata['h'], g.dstdata['h'] = expand_as_pair(h, g)

            flag = self.gate_type is not None

            if flag:
                g.apply_edges(self.gates(r))

            reduce = fn.mean
            if self.attention_type is not None:
                g.apply_edges(self.attention(a))
                g.edata['attention'] = ops.edge_softmax(g, g.edata['attention'])
                if flag:
                    g.edata['scale'] = torch.einsum('a,ab->ab', g.edata['attention'], g.edata['gate'])
                else:
                    g.edata['scale'] = g.edata['attention']

                reduce = fn.sum
            elif self.softmax and flag:
                g.edata['scale'] = ops.edge_softmax(g, g.edata['gate'])
                reduce = fn.sum
            elif flag:
                g.edata['scale'] = g.edata['gate']

            if flag:
                mfunc = fn.u_mul_e('h', 'scale', 'm')
            else:
                mfunc = fn.copy_u('h',  'm')

            if self.aggregator == 'lstm':
                reduce = self.lstm_reducer

            g.update_all(message_func=mfunc, reduce_func=reduce('m', 'h_N'))

            if self.aggregator == 'gcn':
                h = self.linear(g.dstdata['h_N'])
            elif self.aggregator in ['graphsage', 'lstm']:
                c = torch.cat((g.dstdata['h'], g.dstdata['h_N']), dim=-1)
                h = self.linear(c)
            elif self.aggregator == 'bi-interaction':
                h = self.W_1(g.dstdata['h'] + g.dstdata['h_N']) + \
                    self.W_2(g.dstdata['h'] * g.dstdata['h_N'])
            else:
                h = g.dstdata['h_N']

            if self.activation:
                h = self.activation_fn(h)

            if self.normalize:
                h = nn.functional.normalize(h)

            return h


class GInRec(nn.Module):
    def __init__(self, n_relations, entity_dim, user_fn, autoencoder_layer_dims,
                 dropouts, dimensions, gate_type='concat', aggregator='gcn',
                 attention='gatv2', attention_dim=32, device='cpu', relations=True, activation=False, vae=False,
                 kl_weight=0.01, nn_predictor_layers=None, tanh_range=False):
        super(GInRec, self).__init__()

        self.user_fn = user_fn
        self.n_entities = user_fn(0)
        self.n_layers = len(dimensions)

        # if light gcn keep dimensions throughout
        self.layer_dims = dimensions if aggregator != 'lightgcn' else [autoencoder_layer_dims[-1]]*self.n_layers
        self.device = device
        if isinstance(entity_dim, dict):
            self.autoencoder = nn.ModuleDict()
            for enum, dim in entity_dim.items():
                self.autoencoder[enum] = Autoencoder(dim, autoencoder_layer_dims, use_activation=activation,
                                                     use_vae=vae, kl_weight=kl_weight, tanh_range=tanh_range)
        else:
            self.autoencoder = Autoencoder(entity_dim, autoencoder_layer_dims, use_activation=activation, use_vae=vae,
                                           kl_weight=kl_weight, tanh_range=tanh_range)
        self.gate_type = gate_type

        # Entity dimensions are reduced after the autoencoder.
        input_dim = autoencoder_layer_dims[-1]

        if gate_type not in ['concat', 'inner_product', None]:
            raise ValueError('Invalid gate type.')

        if attention is not None:
            self.alpha_r = nn.Parameter(torch.Tensor(self.n_layers, n_relations, attention_dim))
        else:
            self.alpha_r = None

        self.W_r = nn.ParameterList() if gate_type is not None else None
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for output_dim, dropout in zip(self.layer_dims, dropouts):
            r_dim = input_dim * 2 if gate_type == 'concat' else input_dim

            if self.W_r is not None:
                W_r = nn.Parameter(torch.Tensor(n_relations, r_dim, input_dim))
                nn.init.xavier_uniform_(W_r, gain=nn.init.calculate_gain('relu'))
                self.W_r.append(W_r)

            # Add layers where the last layer skips activation
            self.layers.append(SimpleRecConv(input_dim, output_dim, gate_type=gate_type, aggregator=aggregator,
                                             attention=attention, attention_dim=attention_dim, relations=relations))

            self.dropouts.append(nn.Dropout(dropout))
            input_dim = output_dim

        self.predictor_layers = None
        self.pred_activation = nn.ReLU()
        if nn_predictor_layers is not None:
            self.predictor_layers = nn.ModuleList()
            in_dim = self.layer_dims[-1]
            for out_dim in nn_predictor_layers:
                self.predictor_layers.append(nn.Linear(in_dim, out_dim))
                in_dim = out_dim

        sigmoid = torch.nn.LogSigmoid()
        self.loss_fn = lambda pos, neg: - torch.mean(sigmoid(pos - neg))
        self.embeddings = None

    def embedder(self, blocks: List[DGLBlock], x: torch.FloatTensor):
        dstnodes = blocks[-1].dstnodes()

        embeddings = []
        for l, layer in enumerate(self.layers):
            x = self.dropouts[l](x)
            x = layer(blocks[l], x, self.W_r[l] if self.gate_type is not None else None,
                      self.alpha_r[l] if self.alpha_r is not None else None)

            embeddings.append(x[dstnodes])

        return torch.cat(embeddings, dim=-1)

    def predict(self, g: dgl.DGLGraph, embeddings):
        with g.local_scope():
            users, items = g.edges()
            user_emb, item_emb = embeddings[users], embeddings[items]

            if self.predictor_layers is not None:
                x = torch.cat([user_emb, item_emb], dim=-1)
                for i, layer in enumerate(self.predictor_layers, 1):
                    x = layer(x)

                    if i != len(self.predictor_layers):
                        x = self.pred_activation(x)
            else:
                x = (user_emb * item_emb).sum(dim=1)

            return x

    def inference(self, g: dgl.DGLGraph, embeddings, batch_size=128):
        if isinstance(embeddings, dict):
            embeddings = [self.autoencoder[enum.name].propagate(embeddings[enum.name], 'encode') for enum in sorted(FeatureEnum)
                          if enum.name in embeddings]
            embeddings = torch.cat(embeddings, dim=0)
        else:
            embeddings = self.autoencoder.propagate(embeddings, 'encode')

        all_emb = []
        for l, layer in enumerate(self.layers):
            next_embeddings = torch.zeros((g.number_of_nodes(), layer.output_dim), device=embeddings.device)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1, prefetch_edge_feats=['type'])
            dataloader = dgl.dataloading.NodeDataLoader(
                g, torch.arange(g.number_of_nodes()), sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False, device=self.device)

            # Within a layer, iterate over nodes in batches
            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0]
                nodes = block.srcdata[dgl.NID]
                next_embeddings[output_nodes] = layer(block, embeddings[nodes],
                                                      self.W_r[l] if self.W_r is not None else None,
                                                      self.alpha_r[l] if self.alpha_r is not None else None)

            embeddings = next_embeddings
            all_emb.append(embeddings)

        self.embeddings = torch.cat(all_emb, dim=-1)

    def forward(self, users, items, rank_all=False, apply_user_fn=True):
        if apply_user_fn:
            users = self.user_fn(users)

        users, items = users.to(self.device), items.to(self.device)

        user_embs = self.embeddings[users]
        item_embs = self.embeddings[items]

        if rank_all:
            predictions = torch.matmul(user_embs, item_embs.T)
        else:
            predictions = (user_embs * item_embs).sum(dim=1)

        return predictions

    def loss_ae(self, embeddings, input_nodes):
        if isinstance(embeddings, dict):
            encoded = []
            ae_loss = []
            for enum in sorted(FeatureEnum):
                enum = enum.name
                if enum in embeddings:

                    emb = embeddings[enum]
                    enc, decoded = self.autoencoder[enum](emb)
                    ae_loss.append(self.autoencoder[enum].loss(emb, decoded, enc).mean(dim=-1))
                    encoded.append(enc)

            encoded = torch.cat(encoded, dim=0)
            ae_loss = torch.cat(ae_loss)

        else:
            encoded, decoded = self.autoencoder(embeddings)
            ae_loss = self.autoencoder.loss(embeddings, decoded, encoded)

        encoded = encoded[input_nodes]
        ae_loss = ae_loss[input_nodes].mean()

        return encoded, ae_loss

    def loss(self, pos_graph, neg_graph, blocks, embeddings):
        graph_embeddings = self.embedder(blocks, embeddings)
        pos_preds, neg_preds = self.predict(pos_graph, graph_embeddings), self.predict(neg_graph, graph_embeddings)

        pos_preds, neg_preds = pos_preds.unsqueeze(0), neg_preds.view(-1, pos_preds.shape[0])
        cf_loss = self.loss_fn(pos_preds, neg_preds)

        return cf_loss, pos_preds > neg_preds

