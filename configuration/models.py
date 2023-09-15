from models.bpr.bpr_dgl_recommender import BPRDGLRecommender
from models.graphsage.graphsage_dgl_recommender import GraphSAGEDGLRecommender
from models.idcf.idcf_dgl_recommender import IDCFDGLRecommender
from models.igmc.igmc_dgl_recommender import IGMCDGLRecommender
from models.ginrec.ginrec_recommender import GInRecRecommender
from models.kgat.kgat_dgl_recommender import KGATDGLRecommender
from models.mock.mock_recommender import MockRecommender
from models.ngcf.ngcf_dgl_recommender import NGCFDGLRecommender
from models.pinsage.pinsage_recommender import PinSAGERecommender
from models.ppr.ppr_dgl_recommender import PPRDGLRecommender
from models.random.random_dgl_recommender import RandomDGLRecommender
from models.toppop.toppop_dgl_recommender import TopPopDGLRecommender



base_models = {
    'lightgcn': {
        'model': NGCFDGLRecommender,
        'graphs': ['cg_pos'],
        'use_cuda': True,
        'hyperparameters': {
            'learning_rates':  [1e-4, 0.005],
            'weight_decays': [1e-5, 100.],
            'dropouts': [0., 0.8]
        },
        'sherpa_types': {'learning_rates': 'continuous', 'dropouts': 'continuous',
                         'weight_decays': 'continuous'}
    },
    'ngcf': {
        'model': NGCFDGLRecommender,
        'graphs': ['cg_pos'],
        'use_cuda': True,
        'lightgcn': False,
        'hyperparameters': {
            'learning_rates':  [1e-4, 0.005],
            'weight_decays': [0., 100.],
            'dropouts': [0., 0.8]
        },
        'sherpa_types': {'learning_rates': 'continuous', 'dropouts': 'continuous',
                         'weight_decays': 'continuous'}
    },
    'ginrec': {
        'model': GInRecRecommender,
        'use_cuda': True,
        'graphs': ['ckg_rev'],
        'hyperparameters':  {
            'learning_rates': [0.01, 1e-5],
            'dropouts': [0, 0.5],
            'autoencoder_weights': [1e-5, 2.],
            'gate_types': ['concat', 'inner_product'],
            'weight_decays': [1e-8, .1],
            'aggregators': ['gcn', 'graphsage', 'bi-interaction', 'lightgcn'],
        },
        'sherpa_types': {'learning_rates': 'continuous', 'dropouts': 'continuous',
                         'autoencoder_weights': 'continuous', 'gate_types': 'choice',
                         'weight_decays': 'continuous', 'aggregators': 'choice'}
    },
    'ginrec-small': {
        'model': GInRecRecommender,
        'use_cuda': True,
        'graphs': ['ckg_rev'],
        'hyperparameters':  {
            'learning_rates': [0.01, 1e-5],
            'dropouts': [0, 0.5],
            'autoencoder_weights': [1e-5, 2.],
            'gate_types': ['concat', 'inner_product'],
            'weight_decays': [1e-8, .1],
            'aggregators': ['gcn', 'graphsage', 'bi-interaction', 'lightgcn'],
        },
        'sherpa_types': {'learning_rates': 'continuous', 'dropouts': 'continuous',
                         'autoencoder_weights': 'continuous', 'gate_types': 'choice',
                         'weight_decays': 'continuous', 'aggregators': 'choice'},
        # 'batch_size': 16,
        'activation': True
    },
    'bpr': {
        'model': BPRDGLRecommender,
        'use_cuda': True,
        'graphs': ['cg_pos'],
        'hyperparameters': {
            'learning_rates': [0.1, 0.001],
            'latent_factors': [8, 32],
        },
        'sherpa_types': {'learning_rates': 'continuous', 'latent_factors': 'discrete'}
    },
    'graphsage': {
        'model': GraphSAGEDGLRecommender,
        'use_cuda': True,
        'graphs': ['kg', 'cg_pos'],
        'hyperparameters': {
            'learning_rates': [0.01, 1e-4],
            'dropouts': [0., 0.5],
            'aggregators': ['mean', 'pool'],
        },
        'sherpa_types': {'learning_rates': 'continuous', 'dropouts': 'continuous',
                         'aggregators': 'choice'}
    },
    'kgat': {
        'model': KGATDGLRecommender,
        'use_cuda': True,
        'graphs': ['ckg_rev'],
        'hyperparameters': {
            'learning_rates': [0.05, 0.001],
            'dropouts': [0., 0.8],
            'weight_decays': [1e-5, 100.],
        },
        'sherpa_types': {'learning_rates': 'continuous', 'dropouts': 'continuous',
                         'weight_decays': 'continuous'}
    },
    'idcf': {
        'model': IDCFDGLRecommender,
        'use_cuda': True,
        'graphs': ['cg_pos'],
        'hyperparameters': {
            'attention_samples': [200, 500, 2000],
            'learning_rates': [0.1, 0.0001],
            'weight_decays': [1e-8, .2],
            'dims': [(32, [32, 32, 1]), (64, [64, 64, 1])]
        },
        'sherpa_types': {'learning_rates': 'continuous', 'dropouts': 'continuous',
                         'weight_decays': 'continuous', 'attention_samples': 'choice',
                         'dims': 'choice'}
    },
    'idcf-bpr': {
        'model': IDCFDGLRecommender,
        'use_cuda': True,
        'graphs': ['cg_pos'],
        'hyperparameters': {
            'attention_samples': [200, 500, 2000],
            'learning_rates': [0.1, 0.0001],
            'weight_decays': [1e-8, .2],
            'dims': [(32, [32, 32, 1]), (64, [64, 64, 1])]
        },
        'sherpa_types': {'learning_rates': 'continuous', 'dropouts': 'continuous',
                         'weight_decays': 'continuous', 'attention_samples': 'choice',
                         'dims': 'choice'},
        'use_bpr': True
    },
    'idcf-i': {
        'model': IDCFDGLRecommender,
        'use_cuda': True,
        'graphs': ['cg_pos'],
        'hyperparameters': {},
        'sherpa_types': {},
        'features': True
    },
    'ppr': {
        'model': PPRDGLRecommender,
        'graphs': ['ckg_pos'],
        'hyperparameters': {
            'alphas': [0.25, 0.45, 0.65, 0.85],
        },
        'gridsearch': True,
        'sherpa_types': {'alphas': 'choice'}
    },
    'pinsage-bpr': {
        'model': PinSAGERecommender,
        'use_cuda': True,
        'graphs': ['cg_pos'],
        'hyperparameters': {
            'learning_rates': [1.e-5, 0.1],
            'dropouts': [0., 0.8],
            'weight_decays': [1e-8, .1]
        },
        'sherpa_types': {'learning_rates': 'continuous', 'dropouts': 'continuous',
                         'weight_decays': 'continuous'},
        'bpr': True,
        'features': True
    },
    'pinsage': {
        'model': PinSAGERecommender,
        'use_cuda': True,
        'graphs': ['cg_pos'],
        'hyperparameters': {
            'learning_rates': [1.e-5, 0.1],
            'dropouts': [0., 0.8],
            'weight_decays': [1e-8, .1],
            'deltas': [0.01, 32.]
        },
        'sherpa_types': {'learning_rates': 'continuous', 'dropouts': 'continuous',
                         'weight_decays': 'continuous', 'deltas': 'continuous'},
        'features': True
    },
    'toppop': {
        'model': TopPopDGLRecommender,
        'graphs': ['cg_pos'],
        'hyperparameters': {}
    },
    'random': {
        'model': RandomDGLRecommender,
        'graphs': [],
        'hyperparameters': {}
    },
    'mock': {
        'model': MockRecommender,
        'use_cuda': True,
        'graphs': [],
        'hyperparameters':  {
            'learning_rates': [0.01, 1e-5],
            'dropouts': [0, 0.5],
            'autoencoder_weights': [0., 2.],
            'gate_types': ['concat', 'inner_product'],
            'weight_decays': [1e-8, .1],
            'aggregators': ['gcn', 'graphsage', 'bi-interaction', 'lightgcn'],
        },
        'sherpa_types': {'learning_rates': 'continuous', 'dropouts': 'continuous',
                         'autoencoder_weights': 'continuous', 'gate_types': 'choice',
                         'weight_decays': 'continuous', 'aggregators': 'choice'}
    }
}

grid_search = {
    'ginrec-s-vae': {
        'model': GInRecRecommender,
        'use_cuda': True,
        'graphs': ['ckg_rev'],
        'hyperparameters':  {
            'learning_rates': [0.0003],
            'dropouts': [0.06],
            'autoencoder_weights': [0.1],
            'gate_types': ['concat'],
            'weight_decays': [2.9e-06],
            'aggregators': ['lightgcn'],
        },
        'sherpa_types': {'learning_rates': 'choice', 'dropouts': 'choice',
                         'autoencoder_weights': 'choice', 'gate_types': 'choice',
                         'weight_decays': 'choice', 'aggregators': 'choice'},
        # 'batch_size': 16,
        'activation': True,
        'vae': True,
        'gridsearch': True
    },
    'igmc': {
        'model': IGMCDGLRecommender,
        'use_cuda': True,
        'graphs': ['cg_pos_rev'],
        'hyperparameters': {
            'learning_rates': [0.001],
            'learning_rate_factors': [0.1],
            'learning_rate_steps': [50],
            'hops': [1]
        },
        'sherpa_types': {'learning_rates': 'choice', 'learning_rate_factors': 'choice',
                         'learning_rate_steps': 'choice', 'hops': 'choice'},
        'gridsearch': True
    },'lightgcn-go': {
        'model': NGCFDGLRecommender,
        'use_cuda': True,
        'graphs': ['cg_pos'],
        'lightgcn': True,
        'hyperparameters': {
            'learning_rates':  [1e-3],
            'weight_decays': [1e-5],
            'batch_sizes': [2048],
            'dropouts': [0.1]
        },
        'sherpa_types': {'learning_rates': 'choice', 'dropouts': 'choice',
                         'weight_decays': 'choice', 'batch_sizes': 'choice'},
        'gridsearch': True
    },'idcf-book': {
        'model': IDCFDGLRecommender,
        'use_cuda': True,
        'graphs': ['cg_pos'],
        'hyperparameters': {
            'attention_samples': [500],
            'learning_rates': [0.0001],
            'weight_decays': [0.0004],
            'pretrain_learning_rates': [0.01],
            'dims': [(32, [32, 32, 1])]
        },
        'sherpa_types': {'learning_rates': 'continuous', 'dropouts': 'continuous',
                         'weight_decays': 'continuous', 'attention_samples': 'choice',
                         'dims': 'choice', 'pretrain_learning_rates': 'choice'},
        'gridsearch': True
    },
}

# All ablation studies must be run with --other_model.
layer_ablation = {
    'ginrec-l1': {
        'model': GInRecRecommender,
        'use_cuda': True,
        'graphs': ['ckg_rev'],
        'hyperparameters':  {},
        'sherpa_types': {},
        'ablation_parameter': {'layers': 1}
    },
    'ginrec-l2': {
        'model': GInRecRecommender,
        'use_cuda': True,
        'graphs': ['ckg_rev'],
        'hyperparameters':  {},
        'sherpa_types': {},
        'ablation_parameter': {'layers': 2}
    },
    'ginrec-l4': {
        'model': GInRecRecommender,
        'use_cuda': True,
        'graphs': ['ckg_rev'],
        'hyperparameters':  {},
        'sherpa_types': {},
        'ablation_parameter': {'layers': 4}
    }
}
autoencoder_ablation = {
    'ginrec-a0': {
        'model': GInRecRecommender,
        'use_cuda': True,
        'graphs': ['ckg_rev'],
        'hyperparameters':  {},
        'sherpa_types': {},
        'ablation_parameter': {'autoencoder_weights': 0}
    },
    'ginrec-a00001': {
        'model': GInRecRecommender,
        'use_cuda': True,
        'graphs': ['ckg_rev'],
        'hyperparameters':  {},
        'sherpa_types': {},
        'ablation_parameter': {'autoencoder_weights': 0.0001}
    },
    'ginrec-a0001': {
        'model': GInRecRecommender,
        'use_cuda': True,
        'graphs': ['ckg_rev'],
        'hyperparameters':  {},
        'sherpa_types': {},
        'ablation_parameter': {'autoencoder_weights': 0.001}
    },
    'ginrec-a001': {
        'model': GInRecRecommender,
        'use_cuda': True,
        'graphs': ['ckg_rev'],
        'hyperparameters':  {},
        'sherpa_types': {},
        'ablation_parameter': {'autoencoder_weights': 0.01}
    },
    'ginrec-a01': {
        'model': GInRecRecommender,
        'use_cuda': True,
        'graphs': ['ckg_rev'],
        'hyperparameters':  {},
        'sherpa_types': {},
        'ablation_parameter': {'autoencoder_weights': 0.1}
    },
    'ginrec-a05': {
        'model': GInRecRecommender,
        'use_cuda': True,
        'graphs': ['ckg_rev'],
        'hyperparameters':  {},
        'sherpa_types': {},
        'ablation_parameter': {'autoencoder_weights': 0.5}
    },
    'ginrec-a1': {
        'model': GInRecRecommender,
        'use_cuda': True,
        'graphs': ['ckg_rev'],
        'hyperparameters':  {},
        'sherpa_types': {},
        'ablation_parameter': {'autoencoder_weights': 1}
    },
    'ginrec-a2': {
        'model': GInRecRecommender,
        'use_cuda': True,
        'graphs': ['ckg_rev'],
        'hyperparameters':  {},
        'sherpa_types': {},
        'ablation_parameter': {'autoencoder_weights': 2}
    },
    'ginrec-vae': {
        'model': GInRecRecommender,
        'use_cuda': True,
        'graphs': ['ckg_rev'],
        'hyperparameters':  {'autoencoder_weights': [1e-5, 2.],
                             'dropouts': [0.0610116665135389], 'gate_types': ['concat'],
                             'learning_rates': [0.0003318008747161288], 'weight_decays': [2.969230908528221e-06],
                             'kl_weights': [0.01, 2.], 'aggregators': ['lightgcn']},
        'sherpa_types': {'learning_rates': 'choice', 'dropouts': 'choice',
                         'autoencoder_weights': 'continuous', 'gate_types': 'choice',
                         'weight_decays': 'choice', 'aggregators': 'choice', 'kl_weights': 'continuous'},
        'vae': True
    }
}
aggregator_ablation = {
    'ginrec-lstm': {
        'model': GInRecRecommender,
        'use_cuda': True,
        'graphs': ['ckg_rev'],
        'hyperparameters':  {},
        'sherpa_types': {},
        'ablation_parameter': {'aggregators': 'lstm'}
    },
    'ginrec-gcn': {
        'model': GInRecRecommender,
        'use_cuda': True,
        'graphs': ['ckg_rev'],
        'hyperparameters':  {},
        'sherpa_types': {},
        'ablation_parameter': {'aggregators': 'gcn'}
    },
    'ginrec-gs': {
        'model': GInRecRecommender,
        'use_cuda': True,
        'graphs': ['ckg_rev'],
        'hyperparameters':  {},
        'sherpa_types': {},
        'ablation_parameter': {'aggregators': 'graphsage'}
    },
    'ginrec-lightgcn': {
        'model': GInRecRecommender,
        'use_cuda': True,
        'graphs': ['ckg_rev'],
        'hyperparameters':  {},
        'sherpa_types': {},
        'ablation_parameter': {'aggregators': 'lightgcn'}
    },
    'ginrec-bi-interaction': {
        'model': GInRecRecommender,
        'use_cuda': True,
        'graphs': ['ckg_rev'],
        'hyperparameters':  {},
        'sherpa_types': {},
        'ablation_parameter': {'aggregators': 'bi-interaction'}
    },
}
gate_ablation = {
    # Do not train any edge weights and therefore no gates.
    'ginrec-no-g': {
        'model': GInRecRecommender,
        'use_cuda': True,
        'graphs': ['ckg_rev'],
        'hyperparameters':  {},
        'sherpa_types': {},
        'ablation_parameter': {'gate_types': None}
    },
    # Ignore relation types i.e. one weight or shared gated for all relations.
    'ginrec-no-r-g': {
        'model': GInRecRecommender,
        'use_cuda': True,
        'graphs': ['ckg_rev'],
        'hyperparameters':  {},
        'sherpa_types': {},
        'relations': False
    },
    # Use inner product.
    'ginrec-inner': {
        'model': GInRecRecommender,
        'use_cuda': True,
        'graphs': ['ckg_rev'],
        'hyperparameters':  {},
        'sherpa_types': {},
        'ablation_parameter': {'gate_types': 'inner_product'}
    }
}
graph_ablation = {
    # Use a bipartite graph instead of knowledge graph
    'ginrec-bipartite': {
        'model': GInRecRecommender,
        'use_cuda': True,
        'graphs': ['ckg_rev'],
        'hyperparameters':  {},
        'sherpa_types': {},
        'bipartite': True
    },
    'ginrec-bipartite-t': {
        'model': GInRecRecommender,
        'use_cuda': True,
        'graphs': ['ckg_rev'],
        'hyperparameters': {},
        'sherpa_types': {},
        'bipartite': True
    }
}
learned_ablation = {
    # Trainable user embeddings.
    'ginrec-u': {
        'model': GInRecRecommender,
        'use_cuda': True,
        'graphs': ['ckg_rev'],
        'hyperparameters':  {},
        'sherpa_types': {},
        'trainable_users': True
    },
    # Trainable item embeddings (also reduce initial dimensionality due to complexity)
    'ginrec-i': {
        'model': GInRecRecommender,
        'use_cuda': True,
        'graphs': ['ckg_rev'],
        'hyperparameters':  {},
        'sherpa_types': {},
        'features': False
    },
    'ginrec-ui': {
        'model': GInRecRecommender,
        'use_cuda': True,
        'graphs': ['ckg_rev'],
        'hyperparameters':  {},
        'sherpa_types': {},
        'trainable_users': True,
        'features': False
    },
    'ginrec-act': {
        'model': GInRecRecommender,
        'use_cuda': True,
        'graphs': ['ckg_rev'],
        'hyperparameters':  {
            'learning_rates': [0.01, 1e-5],
            'dropouts': [0, 0.5],
            'autoencoder_weights': [1e-5, 2.],
            'gate_types': ['concat', 'inner_product'],
            'weight_decays': [1e-8, .1],
            'aggregators': ['gcn', 'graphsage', 'bi-interaction', 'lightgcn'],
        },
        'sherpa_types': {'learning_rates': 'continuous', 'dropouts': 'continuous',
                         'autoencoder_weights': 'continuous', 'gate_types': 'choice',
                         'weight_decays': 'continuous', 'aggregators': 'choice'},
        'activation': True
    },
}

# Initial feature test
ginrec_features_ablation = {
    'ginrec-simple': {
        'model': GInRecRecommender,
        'use_cuda': True,
        'graphs': ['ckg_rev'],
        'hyperparameters':  {},
        'sherpa_types': {}
    }
}

# Bias test
no_bias_ablation = {
    'bpr-bias': {
        'model': BPRDGLRecommender,
        'use_cuda': True,
        'graphs': ['cg_pos'],
        'hyperparameters': {},
        'sherpa_types': {},
        'user_bias': False,
        'item_bias': False
    }
}

#lgcg
amazon_ablation = {
    'lightgcn-amazon': {
        'model': NGCFDGLRecommender,
        'graphs': ['cg_pos'],
        'use_cuda': True,
        'hyperparameters': {
            'learning_rates':  [0.001],
            'weight_decays': [1e-4],
            'dropouts': [0.],
            'batch_sizes': [8192]
        },
        'sherpa_types': {'learning_rates': 'choice', 'dropouts': 'choice',
                         'weight_decays': 'choice', 'batch_sizes': 'choice'},
        'gridsearch': True
    },
    'bpr-amazon': {
        'model': BPRDGLRecommender,
                 'use_cuda': True,
        'graphs': ['cg_pos'],
        'hyperparameters': {
            'learning_rates': [1.e-3],
            'latent_factors': [64],
            'num_negatives': [50],
            'weight_decays': [1.e-7]
        },
        'sherpa_types': {'learning_rates': 'choice', 'latent_factors': 'choice', 'num_negatives': 'choice',
                         'weight_decays': 'choice'},
        'gridsearch': True
    },
    'idcf-amazon': {
        'model': IDCFDGLRecommender,
        'use_cuda': True,
        'graphs': ['cg_pos'],
        'hyperparameters': {
            'attention_samples': [200],
            'learning_rates': [0.0001],
            'weight_decays': [0.001],
            'dims': [(64, [64, 64, 1])]
        },
        'sherpa_types': {'learning_rates': 'choice', 'dropouts': 'choice',
                         'weight_decays': 'choice', 'attention_samples': 'choice',
                         'dims': 'choice'},
        'gridsearch': True,
        'features': True
    }
}

# Update dictionary with ablation studies
dgl_models = {}
dgl_models.update(base_models)
dgl_models.update(grid_search)
dgl_models.update(layer_ablation)
dgl_models.update(autoencoder_ablation)
dgl_models.update(aggregator_ablation)
dgl_models.update(gate_ablation)
dgl_models.update(graph_ablation)
dgl_models.update(learned_ablation)
dgl_models.update(ginrec_features_ablation)
dgl_models.update(no_bias_ablation)
dgl_models.update(amazon_ablation)