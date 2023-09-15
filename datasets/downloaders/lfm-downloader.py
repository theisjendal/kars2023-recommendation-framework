# Mappings file: 0: item_id, 1: name, 2: item uri
# Feedback file: 0: user id, 1: item id, 2: rating (not available for lfm), 3: timestamp (only available for ML1)
# All: 0: user id, 1: item uri, 2: rating, 3: timestamp
import argparse

import os

import numpy as np
import pandas as pd

from datasets.downloaders.wikidata.queries import get_urls_from_dbpedia
from datasets.downloaders.wikimedia.queries import get_text_descriptions
from shared.utility import download_progress, valid_dir

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=valid_dir, default='../movielens',
                    help='path to store downloaded data')

MAPPING_URL = "https://raw.githubusercontent.com/D2KLab/entity2rec/master/datasets/LastFM/original/mappings.tsv"
FEEDBACK_URL = "https://raw.githubusercontent.com/D2KLab/entity2rec/master/datasets/LastFM/original/feedback.txt"
GRAPH_BASE_URL = "https://raw.githubusercontent.com/D2KLab/entity2rec/master/datasets/LastFM/graphs/"

RELATION_NAMES = ["dbo%3AassociatedBand.edgelist", "dbo%3AassociatedMusicalArtist.edgelist",
                  "dbo%3AbandMember.edgelist", "dbo%3AbirthPlace.edgelist", "dbo%3AformerBandMember.edgelist",
                  "dbo%3Agenre.edgelist", "dbo%3Ahometown.edgelist", "dbo%3Ainstrument.edgelist",
                  "dbo%3Aoccupation.edgelist", "dbo%3ArecordLabel.edgelist", "dct%3Asubject.edgelist"]

def download_files(path):
    download_progress(MAPPING_URL, path, 'mappings.tsv')
    download_progress(FEEDBACK_URL, path, 'feedback.csv')
    graph_path = os.path.join(path, 'graphs')
    os.makedirs(graph_path, exist_ok=True)
    for relation in RELATION_NAMES:
        download_progress(GRAPH_BASE_URL + relation, graph_path, relation.replace('%3A', ':'))


def download_text(path):
    # Get all entities available in graph
    entities = set()
    for rel in RELATION_NAMES:
        dfr = pd.read_csv(os.path.join(path, 'graphs', rel.replace('%3A', ':')), sep=' ', header=None,
                          names=['head', 'tail'])
        for col in dfr:
            entities.update(dfr[col].unique().tolist())

    # Get all entities with mapping (might not be in graph but only feedback
    dfm = pd.read_csv(os.path.join(path, 'mappings.tsv'), sep='\t', header=None, names=['itemId', 'name', 'uri'])
    items = set(dfm.uri.tolist())

    # Add items
    entities.update(items)

    # Get labels and wikipedia links
    header = True
    entity_path = os.path.join(path, 'entities.csv')

    if os.path.isfile(entity_path):
        df = pd.read_csv(entity_path)
        entities = entities.difference(df.uri.tolist())

    if entities:
        with open(entity_path, 'w') as f:
            for df in get_urls_from_dbpedia(entities):
                df.to_csv(f, header=header, index=False)
                header = False

    df = pd.read_csv(entity_path)
    uri_index = {uri: idx for idx, uri in df.uri.iteritems()}

    df_only_wiki = df[df.description.isna() & df.comment.isna() & df.wikilink.notna()]
    url_index = {url.split('?')[0].replace('http', 'https'): uri_index[uri] for uri, url in
                 zip(df_only_wiki.uri, df_only_wiki.wikilink)}

    # Query Wikipedia for entity descriptions
    for descriptions in get_text_descriptions(list(url_index.keys())):
        for url, desc in descriptions.items():
            df.loc[url_index[url], 'description'] = desc.get('WIKI_DESCRIPTION')

    # If no name try to use url instead.
    mask = df['name'].isna()
    df.at[mask, 'name'] = df[mask].uri.apply(lambda x: x.rsplit('/', 1)[-1].replace('_', ' '))

    df.drop(df[~df.uri.str.startswith('http://dbpedia.org/')].index)
    df = df[~df.uri.duplicated()]

    # If no name, description or comment, remove.
    df = df.replace('', np.NaN)
    df = df[df.name.notna() | df.description.notna() | df.comment.notna()]

    df.to_csv(entity_path, index=False)



def run(path):
    download_files(path)

    download_text(path)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args.path)