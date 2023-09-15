import argparse

import os
from collections import defaultdict

import json

import numpy as np
import pickle

from configuration.experiments import experiment_names
from configuration.models import dgl_models
from misc.user_bin_plot import replace_method, replace_dataset, create_mean
from shared.utility import valid_dir, get_experiment_configuration

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=valid_dir, help='path to datasets')
parser.add_argument('--results', type=valid_dir, help='path to results')
# parser.add_argument('--out_path', type=valid_dir, help='path to store results')
parser.add_argument('--experiments', nargs='+', choices=experiment_names, type=str, help='name of experiment')
parser.add_argument('--models', nargs='+', choices=dgl_models.keys(), type=str, help='name of models')
parser.add_argument('--metrics', nargs='+', type=str, help='metrics to include')
parser.add_argument('--at', default=20, type=int, help='value to use as k in @k')
parser.add_argument('--study', default='standard')


def newline():
    return '\n'


def new_table_line():
    return ' \\\\ \\hline' + newline()


def gen_title(experiments, metrics, at):
    title = ' & '

    # Add datasets
    title += ' & '.join([f'\\multicolumn{{{len(metrics)}}}{{c|}}{{{replace_dataset(experiment.dataset.name)}}}'
                          for experiment in experiments])
    title += new_table_line()

    # Add metrics:

    for e in experiments:
        for i, m in enumerate(metrics):
            if i+1 == len(metrics):
                m = m.replace('max_recall', 'recall')
                name = f'\\multicolumn{{1}}{{l|}}{{{m}@{at}}}'
            else:
                name = f'\\multicolumn{{1}}{{l}}{{{m}@{at}}}'

            title += f' & {name}'

    title += new_table_line()

    return title


def gen_row(model, results):
    # Insert metrics
    model = replace_method(model)
    row = ' & '.join([f'\\textbf{{{model}}}'] + [value for experiment, metric in results.items()
                                for _, value in metric.items()])
    row += new_table_line()
    return row


def gen_rows(models, results):
    rows = ''
    for model in models:
        rows += gen_row(model, results[model])

    return rows


def build_table(path, experiments, models, metrics, results, at):
    table = ''

    # Add begin
    table += '\\begin{table}[]' + newline()
    table += '\\begin{tabular}'

    ls = 'r|' + ''.join('r'*len(metrics) + '|' for _ in range(len(experiments)))

    table += f'{{{ls}}}'
    table += newline()

    # Get table
    table += gen_title(experiments, metrics, at)

    # Add rows
    table += gen_rows(models, results)

    # Add end
    table += '\\end{tabular}\n\\end{table}'
    return table


def load_results(results, experiments, models, metrics, at, study):
    res_rows = defaultdict(lambda: defaultdict(dict))
    values = defaultdict(lambda: defaultdict(dict))

    for model in models:
        for experiment in experiments:
            loop = True
            fold = 0
            rs = []
            while loop:
                study_name = '' if study == 'standard' else f'_{study}'
                path1 = os.path.join(results, experiment.name, model, f'fold_{fold}_{model}{study_name}'
                                                                               f'_metrics.pickle')
                path2 = os.path.join(results, experiment.name, model, f'fold_{fold}_ml_mr_1m_warm_start_metrics.pickle')

                if os.path.isfile(path1):
                    path = path1
                else:
                    path = path2

                # If fold results does not exist assume end or last fold.
                try:
                    with open(path, 'rb') as f:
                        rs.append(pickle.load(f))

                    fold += 1
                except IOError:
                    loop = False
                    continue

            rs = create_mean(rs, metrics, [at])

            for metric in metrics:
                r = [t[metric][at] for t in rs]
                mean = round(np.mean(r), 3)
                std = round(np.std(r), 3)
                res_rows[model][experiment][metric] = f'{mean:.3f}{{\\textpm}}{std:.3f}'

                values[experiment][metric][model] = mean

    # Assign ranks
    for experiment in experiments:
        for metric in metrics:
            dictionary = values[experiment][metric]
            for rank, model in enumerate(sorted(dictionary, key=dictionary.get, reverse=True)):
                string = res_rows[model][experiment][metric]
                if rank == 0:
                    res_rows[model][experiment][metric] = f'\\textbf{{{string}}}'
                elif rank == 1:
                    res_rows[model][experiment][metric] = f'\\underline{{{string}}}'
                elif rank == 2:
                    res_rows[model][experiment][metric] = f'*{string}'
                else:
                    break

    return res_rows


def run(path, results, experiments, models, metrics, at, study):
    assert not ('recall' in metrics and 'max_recall' in metrics), 'Use either recall or max recall, but not both.'
    # Get experiments
    experiments = [get_experiment_configuration(e) for e in experiments]

    # Get results
    results = load_results(results, experiments, models, metrics, at, study)

    # Build table
    table = build_table(path, experiments, models, metrics, results, at)

    # Save table
    print(table)


if __name__ == '__main__':
    args = vars(parser.parse_args())
    run(**args)