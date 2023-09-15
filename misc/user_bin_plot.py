import argparse
import random
from collections import defaultdict, Counter

import os

import matplotlib
import numpy as np
import pickle
from matplotlib import cm, container
from matplotlib.lines import Line2D

from configuration.experiments import experiment_names
from configuration.models import dgl_models
from shared.utility import valid_dir, get_experiment_configuration
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=valid_dir, help='path to dataset')
parser.add_argument('--results', type=valid_dir, help='path to results')
parser.add_argument('--experiment', choices=experiment_names, type=str, help='name of experiment')
parser.add_argument('--models', nargs='+', choices=dgl_models.keys(), type=str, help='name of models')
parser.add_argument('--metric', type=str, help='metrics to include')
parser.add_argument('--at', default=20, type=int, help='value to use as k in @k')
parser.add_argument('--study', choices=['user_sparsity', 'user_popularity', 'user_test_popularity', 'user_group'],
                    default='user_popularity', help='evaluation study to perform')
parser.add_argument('--lim', action='store_true')
parser.add_argument('--legend', action='store_true')

def _group_users(testing):
    n_bins = 4
    # nratings = [len(u.ratings) for u in testing]
    # percentage_step = 100 / n_bins
    # steps = [percentage_step*i for i in range(n_bins+1)]
    # pct = np.percentile(nratings, steps)
    # s = list(zip(nratings, testing))
    # bins = []
    # for i in range(n_bins):
    #     if i == 0:
    #         bins.append([u for r, u in s if r <= pct[i+1]])
    #     elif i+1 == n_bins:
    #         bins.append([u for r, u in s if pct[i] <= r])
    #     else:
    #         bins.append([u for r, u in s if pct[i] < r <= pct[i+1]])

    testing = sorted(testing, key=lambda x: (len(x.ratings), x.index))
    bin_size = len(testing) // n_bins
    bins = []
    diff = len(testing) - bin_size*n_bins
    for i in range(n_bins):
        start = i*bin_size + (diff if i != 0 else 0)
        bins.append(testing[start:(i+1)*bin_size+diff])

    return bins


def _group_users_popularity(testing, popular=2, use_testing=False):
    counts = Counter([i for u in testing for i, _ in (u.loo if use_testing else u.ratings)])

    n_items = len(counts)
    k = int(n_items * (popular / 100.))

    top_k = sorted(counts, key=counts.get, reverse=True)[:k]

    n_bins = 4
    percentage_step = 100 / n_bins
    bins = defaultdict(list)

    # Get bin number based on percentage, i.e., bin_fn(70) => 2
    for user in testing:
        n_r = len(user.ratings)  # number of ratings
        tk = len([i for i, _ in user.ratings if i in top_k])  # number of popularity biased ratings
        p = (tk / n_r) * 100
        for i in range(1, n_bins+1):
            bin_step = percentage_step*i
            if p <= bin_step:
                bins[i].append(user)
                break

    return [bins[i] for i in range(1, n_bins+1)]


def _group_users_group(testing):
    # Sort according to number of ratings
    testing = sorted(testing, key=lambda u: len(u.ratings))

    tot = 0
    accumulated = {u: (tot := tot + len(u.ratings)) for u in testing}

    n_bins = 4
    bin_size = tot // n_bins
    bins = []
    cur = 0
    bin_no = -1
    for user, acc in accumulated.items():
        if acc <= cur or bin_no == n_bins-1:
            bins[bin_no].append(user)
        else:
            bins.append([user])
            cur += bin_size
            bin_no += 1

    return bins, {u.index: len(u.ratings) for u in testing}


def create_mean(dictionaries, metrics, ats):

    results = []
    for dictionary in dictionaries:
        res = defaultdict(dict)
        for metric in metrics:
            for at in ats:
                if metric != 'cov':
                    res[metric][at] = np.mean([result[at] for result in dictionary[metric]])
                else:
                    res[metric][at] = dictionary[metric][at]

        results.append(res)

    return results


def replace_method(method):
    return method.replace('-s', '').replace('graphsage', 'gs-r').replace('lightgcn', 'lgcn')\
        .replace('bpr', 'bpr-mf').upper().replace('SIMPLEREC', 'SimpleRec').replace('TOPPOP', 'TopPop')\
        .replace('PINSAGE', 'PinSAGE').replace('-I', '').replace('SimpleRec-BIPARTITE', 'Bipartite')\
        .replace('GRAPHSAGE', 'GraphSAGE')


def replace_dataset(name):
    return name.replace("ml-mr-1m", "ML-S").capitalize()


def remove_legend_error(ax, loc='upper left'):
    handles, labels = ax.get_legend_handles_labels()

    new_handles = []

    for h in handles:
        #only need to edit the errorbar legend entries
        if isinstance(h, container.ErrorbarContainer):
            new_handles.append(h[0])
        else:
            new_handles.append(h)

    ax.legend(new_handles, labels, loc=loc)


def load_results(data_path, results_path, experiment, models, metric, at, study):
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for model in models:
        fold = 0
        rs = []
        model_path = os.path.join(results_path, experiment.name, model)
        for file in sorted(os.listdir(model_path)):
            if file.startswith(f'fold_{fold}_{model}') and file.endswith(f'_metrics.pickle') and 'study' not in file\
                    and 'sampling' not in file:
                with open(os.path.join(model_path, file), 'rb') as f:
                    rs.append(pickle.load(f))
                    break

        testing = pickle.load(open(os.path.join(data_path, experiment.dataset.name, experiment.name, f'fold_{fold}',
                                                'test.pickle'),'rb'))

        if study == 'user_sparsity':
        # Group by users if flag is true otherwise wrap testing set
            testing = _group_users(testing)
        elif study == 'user_popularity':
            testing = _group_users_popularity(testing)
        elif study == 'user_test_popularity':
            testing = _group_users_popularity(testing, use_testing=True)
        elif study == 'user_group':  # Same as KGAT
            testing, n_ratings = _group_users_group(testing)

        user_bin = {u.index: i for i, b in enumerate(testing) for u in b}
        user_rating = {u.index: len(u.ratings) for b in testing for u in b}

        for r in rs:
            r['group'] = [user_bin[u] for u in r['user']]

        n_bins = max(rs[0]['group']) + 1
        for i in range(n_bins):
            if metric == 'cov':
                cur_bin = [{metric: fold[metric][i]} for fold in rs]
            else:
                cur_bin = [{metric: [score for score, group in zip(fold[metric], fold['group']) if group == i]}
                           for fold in rs]
            r = create_mean(cur_bin, [metric], [int(at)])

            n_users = [len([g for g in fold['group'] if g == i]) for fold in rs]
            results[model][i]['n_users'] = np.mean(n_users)
            results[model][i]['n_users_std'] = np.std(n_users)

            r = [f[metric][at] for f in r]
            results[model][i][metric] = np.mean(r)
            results[model][i][f'{metric}_std'] = np.std(r)

            results[model][i]['max_rating'] = [max(user_rating[u] for u, g in zip(fold['user'], fold['group']) if g == i)
                                               for fold in rs]

    return results


def plot_one(results, experiment, study, lim, metric, at, use_legend):
    matplotlib.rc('font', **{'size': 14, 'weight': 'bold'})

    labels = []
    method, scores = next(iter(results.items()))
    n_bins = len(scores)
    bin_range = int(100 / n_bins)
    for b in range(n_bins):

        if study != 'user_group':
            range_str = f'{bin_range*b}-{bin_range*(b+1)}'
            if b == 0:
                name = f'[{range_str}]'
            else:
                name = f'({range_str}]'
        else:
            name = f'<={scores[b]["max_rating"][0]}'

        labels.append(name)

    names = study.split('_')
    if study != 'user_group':
        names += ['group']
    plt.xlabel(' '.join(names).title(), fontweight='bold')
    if study != 'user_sparsity':
        score = [[labels[i], scores[i]['n_users']] for i in range(n_bins)]
        score = list(zip(*score))  # transpose list
        plt.bar(*score, width=0.3, color='grey')  #, yerr=[scores[str(i)]['n_users_std'] for i in range(n_bins)])
        plt.ylabel('#Users', fontweight='bold')

        ax = plt.twinx()
    else:
        ax = plt.gca()

    to_plot = []
    for (method, scores), marker in zip(results.items(), random.sample(Line2D.markers.keys(), len(results))):

        score = [[labels[i], scores[i][metric]] for i in range(n_bins)]
        score = list(zip(*score))  # transpose list
        stds = [scores[i][f'{metric}_std'] for i in range(n_bins)]
        to_plot.append((score, method, stds))

    markers = list(Line2D.markers.keys())
    markers = [markers[i] for i in set(range(len(markers))).difference({0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 13,15,16})]
    for (p, method, stds), marker, color in zip(to_plot, markers, cm.tab10(np.linspace(0, 1, len(to_plot)))):
        ax.errorbar(*p, label=replace_method(method), color=color, capsize=10, marker=marker)

    if use_legend:
        ax.legend()

        if study == 'user_popularity':
            remove_legend_error(ax, loc='upper left')
        else:
            remove_legend_error(ax, loc='upper right')

    ax.set_ylabel(f'{metric.upper()}@{at}', fontweight='bold')

    if lim:
        ax.set_ylim(0.1, 0.4 if study == 'user_popularity' else 0.25)
        plt.margins(y=0)

    methods = '_'.join(sorted(results.keys()))
    fname = f'plots/lineplot_{experiment.name}_{study}_{methods}_{n_bins}.pdf'
    plt.savefig(fname, bbox_inches='tight')
    plt.show()


def run(dataset, results, experiment, models, metric, at, study, lim, legend):
    # Get experiments
    experiment = get_experiment_configuration(experiment)

    # Get results
    results = load_results(dataset, results, experiment, models, metric, at, study)

    plot_one(results, experiment, study, lim, metric, at, legend)


if __name__ == '__main__':
    args = vars(parser.parse_args())
    run(**args)