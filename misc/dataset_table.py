import argparse

import os
from collections import defaultdict

from configuration.datasets import dataset_names
from shared.utility import valid_dir, load_users, get_dataset_configuration

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=valid_dir, help='path to datasets')
parser.add_argument('--datasets', nargs='+', choices=dataset_names, type=str, help='name of experiment')
parser.add_argument('--extension', default='', type=str,
                    help='Extension of filenames (e.g., user pickle file)')


def rename_experiments():
    pass


def rename_models():
    pass


def newline():
    return '\n'


def new_table_line():
    return ' \\\\ \\hline' + newline()


def gen_title(data):
    title = ' & '

    # Add datasets
    title += ' & '.join([ f'\\multicolumn{{1}}{{c|}}{{{dataset}}}'
                          for dataset in data.keys()])
    title += new_table_line()

    return title


def gen_row(measure, data):
    # Insert metrics
    row = ' & '.join([measure] + [str(measures[measure]) for _, measures in data.items()])
    row += new_table_line()
    return row


def gen_rows(data):
    rows = ''
    measures = list(list(data.values())[0].keys())
    for m in measures:
        rows += gen_row(m, data)

    return rows


def build_table(data):
    table = ''

    # Add begin
    table += '\\begin{table}[]' + newline()
    table += '\\begin{tabular}'

    ls = 'r|' + ''.join('r' + '|' for _ in range(len(data)))

    table += f'{{{ls}}}'
    table += newline()

    # Get table
    table += gen_title(data)

    # Add rows
    table += gen_rows(data)

    # Add end
    table += '\\end{tabular}\n\\end{table}'
    return table


def load_datasets(path, datasets, extension):
    data = defaultdict(dict)
    for dataset in datasets:
        dataset_path = os.path.join(path, dataset.name)
        # Users
        users = load_users(dataset_path, extension if extension else None)
        n_users = len(users)
        data[dataset.name]['\\# Users'] = n_users

        # Items
        n_items = len({i for user in users for i, _ in user.ratings})
        data[dataset.name]['\\# Items'] = n_items

        # Ratings
        n_ratings = sum([len(user.ratings) for user in users])
        data[dataset.name]['\\# Ratings'] = n_ratings

        # Density
        n_nodes = n_users + n_items
        density = (2*n_ratings)/(n_nodes*(n_nodes-1))
        data[dataset.name]['Density'] = round(density, 4)

    return data


def run(path, datasets, extension):
    # Get experiments
    datasets = [get_dataset_configuration(e) for e in datasets]

    # Get data
    data = load_datasets(path, datasets, extension)

    # Build table
    table = build_table(data)

    # Save table
    print(table)


if __name__ == '__main__':
    args = vars(parser.parse_args())
    run(**args)



