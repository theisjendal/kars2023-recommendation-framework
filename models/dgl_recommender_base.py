import itertools
import os.path
import pickle
import time
from copy import deepcopy

import dgl
import numpy.random
import sherpa
import torch
from loguru import logger
from sherpa import Trial
from torch.utils.tensorboard import SummaryWriter

import random

import numpy as np

from datasets.feature_extractors.feature_extractor import get_feature_extractor
from shared.efficient_validator import Validator
from shared.experiments import Fold
from shared.meta import Meta
from shared.seed_generator import SeedGenerator
from shared.utility import get_feature_configuration, get_experiment_configuration


class RecommenderBase:
    def __init__(self, meta: Meta, seed_generator: SeedGenerator, use_cuda, graphs, infos, hyperparameters, fold: Fold,
                 sherpa_types, workers, parameter_path, writer_path, features=False, gridsearch=False, ablation_parameter=None,
                 feature_configuration=None, other_fold=None,):
        """
        Baseclass used by all methods. Contain base fit method for running methods.
        :param meta: Contains meta information about the dataset, e.g. users, items, entities.
        :param seed_generator: Used to generate and set seeds. Does not effect sherpa optimizer.
        :param use_cuda: To use cuda or not.
        :param graphs: Graphs to be used in training.
        :param infos: Information about the graphs.
        :param hyperparameters: Hyperparameters of the model.
        :param sherpa_types: Mapping of hyperparameters to sherpa types.
        :param workers: Number of workers used for sampling.
        :param parameter_path: Path to store model trials and similar.
        :param features: To use pre-calculated features or not.
        :param gridsearch: To use gridsearch or not.
        :param ablation_parameter: Used for ablation studies. Changes one parameter to another value.
        """
        self.name = ''
        self.meta = meta
        self.seed_generator = seed_generator
        self.use_cuda = use_cuda
        self.fold = fold
        self.graphs = graphs
        self.infos = infos
        self._hyperparameters = hyperparameters
        self.ablation_parameter = ablation_parameter
        self._sherpa_types = sherpa_types
        self.workers = workers
        self._other_fold = other_fold
        self._gridsearch = gridsearch
        self._eval_intermission = 20

        self.device = torch.device('cuda:0') if self.use_cuda else torch.device('cpu')

        self._random = random.Random(seed_generator.get_seed())
        self.require_features = features
        self._max_epochs = 1000
        self._min_epochs = 0
        self._features = None
        self._feature_configuration = feature_configuration
        self.parameter_path = parameter_path
        self.writer_path = writer_path
        self._model = None
        self._optimizer = None
        self._scheduler = None
        self._parameters = None
        self._state = None
        self._seed = seed_generator.get_seed()

        self._study = None
        self._sherpa_parameters = None
        self._trial = None
        self._epoch_resource = {}
        self._no_improvements = 0
        self._best_state = None
        self._best_score = -1
        self._best_epoch = 0
        self._early_stopping = 50
        self.summary_writer = None

        self.__start_time = time.time()

        sherpa.rng = numpy.random.RandomState(self._seed)
        self.set_seed()
        self._load()

    def _create_parameters(self):
        parameters = []
        for key, value in self._hyperparameters.items():
            t = self._sherpa_types[key]
            if t in ['continuous', 'discrete']:
                min_v = np.min(value)
                max_v = np.max(value)

                if key in ['learning_rates', 'weight_decays', 'autoencoder_weights', 'deltas', 'anchors']:
                    scale = 'log'
                else:
                    scale = 'linear'

                if t == 'continuous':
                    parameters.append(sherpa.Continuous(name=key, range=[min_v, max_v], scale=scale))
                else:
                    parameters.append(sherpa.Discrete(name=key, range=[min_v, max_v], scale=scale))
            elif t == 'choice':
                parameters.append(sherpa.Choice(name=key, range=value))
            else:
                raise ValueError(f'Invalid sherpa type or not implemented: {t}')

        self._sherpa_parameters = parameters

    def _parameter_combinations(self):
        self._create_parameters()
        if self._study is None:
            if self._gridsearch:
                algorithm = sherpa.algorithms.GridSearch()
            else:
                algorithm = sherpa.algorithms.SuccessiveHalving(
                    r=1, R=256, eta=4, s=0,  max_finished_configs=1
                )

            study = sherpa.Study(parameters=self._sherpa_parameters, algorithm=algorithm, lower_is_better=False,
                                 disable_dashboard=True)
            self._study = study
        return self._study

    def _create_model(self, trial):
        raise NotImplementedError

    def _epoch_range(self, trial):
        n_epochs = trial.parameters.get('resource', self._max_epochs)

        if not self._epoch_resource:
            self._epoch_resource[n_epochs] = 0
        elif n_epochs not in self._epoch_resource:
            m = max(self._epoch_resource)
            self._epoch_resource[n_epochs] = m + self._epoch_resource[m]
            if self._epoch_resource[m] == 0:
                self._epoch_resource[n_epochs] += self._min_epochs

        init_epoch = self._epoch_resource[n_epochs]
        end_epoch = n_epochs + init_epoch
        if init_epoch == 0:
            end_epoch += self._min_epochs

        # Set timer
        self.__start_time = time.time()

        return init_epoch, end_epoch

    def _on_epoch_end(self, trial, score, epoch, loss=None):
        logger.debug(f'Validation for {self.name}: {score}')
        if trial is not None:
            end_time = time.time()
            context = {'time': end_time - self.__start_time} if self._early_stopping > self._no_improvements else {}

            if loss:  # Assume loss is never zero.
                try:
                    loss = loss.item()  # try to convert, ignore if not of tensor type
                except AttributeError:
                    pass

                context['loss'] = loss

            self._study.add_observation(trial=trial, iteration=epoch,
                                        objective=score, context=context)
            self.__start_time = end_time

        if epoch < self._min_epochs:
            pass  # ensure model runs for at least min epochs before storing best state
        elif score > self._best_score:
            self._best_state = deepcopy(self._model.state_dict())
            self._best_score = score
            self._best_epoch = epoch
            self._no_improvements = 0
        else:
            self._no_improvements += 1

    def _save(self):
        """
        Saves current state and parameters, should be used after each tuning.
        :param score: score for state and parameters.
        """
        if self.parameter_path is not None:
            with open(f'{self.parameter_path}/parameters.states', 'wb') as f:
                state = {'study': self._study, 'resources': self._epoch_resource}
                pickle.dump(state, f)

    def _get_best_trial(self, load_state):
        result = self._study.get_best_result()
        self.set_parameters(result)
        if load_state:
            result['load_from'] = result.get('save_to', str(result.get('Trial-ID')))
        else:
            result['load_from'] = ""
        return Trial(result.pop('Trial-ID'), result)

    def _sherpa_load(self, trial):
        load_from = trial.parameters.get('load_from', '')

        if load_from != "":
            p = os.path.join(self.parameter_path, load_from) + '.trial'
            logger.info(f"Loading model from {p}")

            # Loading model
            checkpoint = torch.load(p, map_location=self.device)

            self._model.load_state_dict(checkpoint['model'])
            if self._optimizer is not None:
                self._optimizer.load_state_dict(checkpoint['optimizer'])
            if self._scheduler is not None:
                self._scheduler.load_state_dict(checkpoint['scheduler'])
            self._best_score = checkpoint['score']
            self._best_epoch = checkpoint.get('epoch', 0)
            self._best_state = checkpoint['state']
            self._no_improvements = checkpoint['iter']
        else:
            self._best_score = 0
            self._best_state = None
            self._no_improvements = 0
            self._best_epoch = 0

    def _sherpa_save(self, trial):
        torch.save({
            'model': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict() if self._optimizer is not None else None,
            'scheduler': self._scheduler.state_dict() if self._scheduler is not None else None,
            'score': self._best_score,
            'state': self._best_state,
            'epoch': self._best_epoch,
            'iter': self._no_improvements
        }, os.path.join(self.parameter_path, trial.parameters.get('save_to', str(trial.id))) + '.trial')

    def _load(self):
        """
        Loads best state and parameters from temporary directory if it exists.
        """

        fname = f'{self.parameter_path}/parameters.states'
        if os.path.isfile(fname):
            with open(fname, 'rb') as f:
                state = pickle.load(f)

            self._study = state.get('study', None)
            self._epoch_resource = state.get('resources', {})

    def _parameter_to_string(self, parameters):
        if isinstance(parameters, str):
            return parameters
        elif isinstance(parameters, tuple) or isinstance(parameters, int):
            return str(parameters)
        elif isinstance(parameters, dict):
            return {k: self._parameter_to_string(v) for k,v in parameters.items()}
        else:
            return f'{parameters:.2E}'

    def _get_ancestor(self, identifier):
        df = self._study.results
        df = df[df['Status'] == 'COMPLETED'].set_index('Trial-ID')
        t = df.loc[identifier]
        if t.load_from == '':
            return identifier
        else:
            return self._get_ancestor(int(t.load_from))

    def fit(self, validator: Validator):
        change = False
        for trial in self._parameter_combinations():
            logger.info(f"Trial:\t{trial.id}")
            if trial.parameters.get('load_from', '') != '':
                a = self._get_ancestor(int(trial.parameters['load_from']))
            else:
                a = trial.id
            self.summary_writer = SummaryWriter(log_dir=os.path.join(self.writer_path, str(a)))
            
            self.set_parameters(trial.parameters)  # Set parameters
            # UVA will create an error if the graph is already pinned
            for i, g in enumerate(self.graphs):
                if g.is_pinned():
                    g.unpin_memory_()

            self.set_seed()
            param_str = self._parameter_to_string(trial.parameters)
            logger.debug(f'{self.name} parameter: {param_str}')
            self._create_model(trial)  # initialize model with parameters given by trial

            first, final = self._epoch_range(trial)
            logger.info(f'Epochs {first} - {final}')

            self._fit(validator, first, final, trial=trial)  # train

            # save trial and state of study (sherpa study)
            self._sherpa_save(trial)
            self._save()
            change = True
            self.summary_writer.add_hparams(trial.parameters, {'score': self._best_score})
            self.summary_writer.close()

        trial = self._get_best_trial(load_state=change)

        # Run with ablation parameter
        if self.ablation_parameter:
            trial.parameters.update(self.ablation_parameter)

        logger.info(f'Using best parameters {trial.parameters}')
        self.set_parameters(trial.parameters)
        self._create_model(trial)
        first = self._epoch_range(trial)[-1] if change else 0

        # If no state is available or a change have occurred, find state.
        if change or self.get_state() is None:
            # Train in we can still train, i.e. both max epochs and early stopping criterion haven't been reached.
            if self._no_improvements < self._early_stopping and first < self._max_epochs:
                self.summary_writer = SummaryWriter(log_dir=os.path.join(self.parameter_path, str(trial.id)))
                self._fit(validator, first, self._max_epochs)

            self.set_state(self._best_state)

        # Should occur for all torch methods.
        if self.get_state() is not None:
            self._model.load_state_dict(self.get_state())

    def _fit(self, validator: Validator, first_epoch, final_epoch=1000, trial=None):
        raise NotImplementedError()

    def predict_all(self, users) -> np.array:
        """
        Score all items given users
        :param users: user ids to predict for
        :return: an matrix of user-item scores
        """
        raise NotImplementedError()

    def set_features(self, features):
        """
        Sets entity features either pretrained or extracted.
        :param features as ndarray, possibly memory mapped.
        :return: None
        """
        self._features = features

    def set_seed(self):
        """
        Seed all random generators, e.g., random, numpy.random, torch.random...
        """
        dgl.random.seed(self._seed)
        dgl.seed(self._seed)
        torch.random.manual_seed(self._seed)
        torch.manual_seed(self._seed)
        torch.cuda.manual_seed(self._seed)
        np.random.seed(self._seed)
        random.seed(self._seed)
        # torch.backends.cudnn.determinstic = True
        # torch.backends.cudnn.benchmark = False
        # torch.use_deterministic_algorithms(True)

    def set_parameters(self, parameters):
        """
        Set all model parameters, often hyperparameters.
        """
        raise NotImplementedError()

    def set_optimal_parameter(self, parameters):
        self._parameters = parameters

    def get_parameters(self) -> dict:
        """
        Get model parameters or hyperparameters.
        :return: Dictionary of parameters
        """
        return self._parameters

    def set_state(self, state):
        """
        Set the model state.
        :param state: State that a model can use for retraining or prediction.
        """
        self._state = state

    def get_state(self):
        """
        Return state of model.
        :return: state of model.
        """
        return self._state

    def get_features(self, feature_kwargs, extract_kwargs):
        if self._feature_configuration is not None:
            f_conf = get_feature_configuration(self._feature_configuration)
            f_conf.kwargs.update(feature_kwargs)
            anchor_path = os.path.join(self.parameter_path, 'anchor')

            os.makedirs(anchor_path, exist_ok=True)

            extension = '' if not feature_kwargs else '_' + '_'.join(map(str, itertools.chain(*feature_kwargs.items())))
            f_path = os.path.join(anchor_path, f_conf.name + extension + '.npy')
            if os.path.isfile(f_path):
                return np.load(f_path)
            else:
                warm_fold = self._other_fold if self._other_fold is not None else self.fold
                cold_fold = self.fold  # used for embeddings given anchors.
                d_path = self.fold.data_loader.path
                d_path = d_path.rsplit('/', 3)[0]

                f_extractor = get_feature_extractor(f_conf, self.use_cuda)
                features = f_extractor.extract_features(d_path, f_conf,
                                                        get_experiment_configuration(warm_fold.experiment.name),
                                                        get_experiment_configuration(cold_fold.experiment.name),
                                                        self.fold.name.split('_')[-1], **extract_kwargs)
                np.save(f_path, features)
                return features
        else:
            return self._features
