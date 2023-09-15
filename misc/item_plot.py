import argparse
import random
from collections import defaultdict

import os

import matplotlib
import numpy as np
import pickle
from matplotlib import cm, container
from matplotlib.lines import Line2D

from configuration.experiments import experiment_names
from configuration.models import dgl_models
from misc.user_bin_plot import replace_method, remove_legend_error
from shared.utility import valid_dir, get_experiment_configuration
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--results', type=valid_dir, help='path to results')
parser.add_argument('--experiment', choices=experiment_names, type=str, help='name of experiment')
parser.add_argument('--models', nargs='+', choices=dgl_models.keys(), type=str, help='name of models')
parser.add_argument('--metric', type=str, help='metrics to include')
parser.add_argument('--at', default=20, type=int, help='value to use as k in @k')
parser.add_argument('--study', choices=['user_sparsity', 'user_popularity', 'user_test_popularity', 'user_group'],
					default='user_popularity', help='evaluation study to perform')
parser.add_argument('--lim', action='store_true')


def load_results(results_path, experiment, models, metric, at, study):
	results = dict()
	for model in models:
		loop = True
		fold = 0
		rs = []
		model_path = os.path.join(results_path, experiment.name, model)
		while loop:

			# If fold results does not exist assume end or last fold.
			try:
				for file in sorted(os.listdir(model_path)):
					if file.startswith(f'fold_{fold}_{model}') and file.endswith(f'_item.pickle'):
						with open(os.path.join(model_path, file), 'rb') as f:
							rs.append(pickle.load(f))
						fold += 1
				else:
					break
			except IOError:
				loop = False
				continue

		results[model] = rs[0]

	return results


def plot_one(results, study, lim, metric, at):
	matplotlib.rc('font', **{'size': 14, 'weight': 'bold'})

	labels = []
	n_bins = 4
	bin_range = int(100 / n_bins)
	for b in range(n_bins):
		range_str = f'{bin_range * b}-{bin_range * (b + 1)}'
		if b == 0:
			name = f'[{range_str}]'
		else:
			name = f'({range_str}]'

		labels.append(name)

	plt.xlabel('Item Popularity Group', fontweight='bold')
	ax = plt.gca()

	to_plot = []
	for method, r in results.items():
		bin_size = len(r) // n_bins
		bins = []
		diff = len(r) - bin_size * n_bins
		r = [s for i in sorted(r, key=lambda x: r.get(x)[0]) for s in r[i][1]]
		# r = {i: [a, sum(np.array(ranks) < at) / len(ranks)] for i, (a, ranks) in r.items()} # hitrate per item
		# r = [r[i][1] for i in sorted(r, key=lambda x: r.get(x)[0])]
		for i in range(n_bins):
			start = i * bin_size + (diff if i != 0 else 0)
			bins.append(r[start:(i + 1) * bin_size + diff])

		score = [[labels[i], np.mean(bins[i])] for i in range(n_bins)]
		score = list(zip(*score))
		stds = [np.std(bins[i]) for i in range(n_bins)]
		to_plot.append((score, method, stds))

	markers = list(Line2D.markers.keys())
	markers = [markers[i] for i in set(range(len(markers))).difference({0, 1, 5, 6})]
	for (p, method, stds), marker, color in zip(to_plot, markers, cm.rainbow(np.linspace(0, 1, 8))):
		ax.errorbar(*p, label=replace_method(method), color=color, capsize=10, marker=marker, yerr=stds)

	ax.legend()

	remove_legend_error(ax, loc='upper right')

	ax.set_ylabel(f'Rank (Lower Better)', fontweight='bold')
	# ax.set_ylabel(f'HR@{at}', fontweight='bold')
	# plt.yscale('log')
	if lim:
		ax.set_ylim(0.1, 0.4 if study == 'user_popularity' else 0.25)
		plt.margins(y=0)

	for i, l in enumerate(labels):
		if i % 2 != 0:
			labels[i] = ''
		plt.gca().set_xticklabels(labels)
	fname = f'plots/lineplot_item_{len(results)}_{n_bins}.pdf'
	plt.savefig(fname, bbox_inches='tight')
	plt.show()


def run(results, experiment, models, metric, at, study, lim):
	# Get experiments
	experiment = get_experiment_configuration(experiment)

	# Get results
	results = load_results(results, experiment, models, metric, at, study)

	plot_one(results, study, lim, metric, at)


if __name__ == '__main__':
	args = vars(parser.parse_args())
	run(**args)
