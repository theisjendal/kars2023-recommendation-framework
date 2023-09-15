import argparse
import random
from collections import defaultdict

import os

import json

import matplotlib
import numpy as np
import pickle
from matplotlib import cm, container
from matplotlib.lines import Line2D

from configuration.experiments import experiment_names
from configuration.models import dgl_models
from misc.user_bin_plot import replace_method, remove_legend_error, create_mean
from shared.utility import valid_dir, get_experiment_configuration
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--results', type=valid_dir, help='path to results')
parser.add_argument('--experiment', choices=experiment_names, type=str, help='name of experiment')
parser.add_argument('--models', nargs='+', choices=dgl_models.keys(), type=str, help='name of models')
parser.add_argument('--metric', choices=['ndcg', 'recall', 'cov', 'hr', 'precision'], type=str, help='metric to include')
parser.add_argument('--lim', action='store_true')
parser.add_argument('--ats', nargs='+', choices=[i for i in range(1,51)], type=int)


def load_results(results_path, experiment, models, metric, ats):
    results = defaultdict(dict)

    for model in models:
        loop = True
        fold = 0
        rs = []
        model_path = os.path.join(results_path, experiment.name, model)
        while loop:

            # If fold results does not exist assume end or last fold.
            try:
                for file in sorted(os.listdir(model_path)):
                    if file.startswith(f'fold_{fold}_{model}') and file.endswith(f'_metrics.pickle') \
                            and not ('study' in file or 'sampling' in file):
                        with open(os.path.join(model_path, file), 'rb') as f:
                            rs.append(pickle.load(f))
                        fold += 1
                else:
                    break
            except IOError:
                loop = False

        rs = create_mean(rs, [metric], ats)
        for at in ats:
            r = [f[metric][at] for f in rs]
            # mean = round(np.mean(r), 4)
            results[model][at] = np.mean(r)
            results[model][f'{at}_std'] = np.std(r)

    return results


def plot_one(results, ats, experiment):
    matplotlib.rc('font', **{'size': 14, 'weight': 'bold'})
    l = list(range(len(ats)))
    for (model, scores), marker, color in zip(results.items(), Line2D.markers.keys(), cm.rainbow(np.linspace(0, 1, 10))):
        r = [scores[at] for at in ats]
        rstd = [scores[f'{at}_std'] for at in ats]
        plt.errorbar(l, r, marker=marker, color=color, label=replace_method(model), yerr=rstd)

    plt.legend()
    plt.ylabel('NDCG', fontweight='bold')
    plt.xlabel('k', fontweight='bold')
    plt.xticks(l, ats)
    remove_legend_error(plt.gca(), loc='lower left')

    fname = f'plots/lineplot_{experiment.name}_{"_".join(map(str, ats))}.pdf'
    plt.savefig(fname, bbox_inches='tight')
    plt.show(bbox_inches='tight')


def run(results, experiment, models, metric, ats, lim):
    # Get experiments
    experiment = get_experiment_configuration(experiment)

    # Get results
    results = load_results(results, experiment, models, metric, ats)

    plot_one(results, ats, experiment)


if __name__ == '__main__':
    args = vars(parser.parse_args())
    run(**args)