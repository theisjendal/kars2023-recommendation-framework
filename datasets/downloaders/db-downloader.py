import argparse
import os
from zipfile import ZipFile

import shutil

import pandas as pd
import requests

from datasets.downloaders.wikidata.queries import get_urls_from_dbpedia, get_entity_labels
from datasets.downloaders.wikimedia.queries import get_text_descriptions
from shared.utility import valid_dir, download_progress

URL = "https://docs.google.com/uc?export=download"
FILE_ID = '1FIbaWzP6AWUNG2-8q6SKQ3b9yTiiLvGW'

parser = argparse.ArgumentParser()
parser.add_argument('--out_path', nargs=1, type=valid_dir, default='../dbbook',
                    help='path to store downloaded data')


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def download_file_from_google_drive(destination):
    session = requests.Session()

    response = session.get(URL, params={'id': FILE_ID}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': FILE_ID, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, os.path.join(destination, 'dataset.zip'))


def unpack(path):
    with ZipFile(os.path.join(path, 'dataset.zip')) as z:
        for file in z.namelist():
            if file.startswith('datasetsτÜäσë»µ£¼/dbbook2014/') and file[-1] != '/':
                f_name = file.split('/', 1)[-1]
                os.makedirs(os.path.dirname(os.path.join(path, f_name)), exist_ok=True)
                with z.open(file) as fp_in, open(os.path.join(path, f_name), 'wb') as fp_out:
                    shutil.copyfileobj(fp_in, fp_out)


def download_entity_information(path):
    df_orig = pd.read_csv(os.path.join(path, 'dbbook2014/kg/e_map.dat'), sep='\t', header=None, names=['id', 'url'])
    # df_orig = df_orig[~df_orig.url.str.startswith('http://dbpedia.org/class/yago/')]  # Not useful

    # Get labels and wikipedia links
    header = True
    entity_path = os.path.join(path, 'db_entities.csv')
    with open(entity_path, 'w') as f:
        for df in get_urls_from_dbpedia(df_orig.url):
            df.to_csv(f, header=header, index=False)
            header = False

    # Get load df
    df = pd.read_csv(entity_path)
    df.drop_duplicates(inplace=True)

    # If uri is a wikidata uri, assign to column
    mask = df.uri.str.startswith('http://www.wikidata.org/entity/')
    df.loc[mask, 'wikidata'] = df[mask].uri

    f = df.groupby(['wikidata'])

    d = {uri: indices.index.tolist() for uri, indices in f}
    entities = df[df.uri.str.startswith('http://www.wikidata.org/entity/') | df.wikidata.notna()].wikidata.to_dict()
    for chunk in get_entity_labels(entities.values()):
        for _, (uri, name, description, wikilink) in chunk.iterrows():
            for index in d[uri]:
                if name is not None:
                    df.at[index, 'name'] = name
                if description is not None:
                    df.at[index, 'description'] = description
                if wikilink is not None:
                    df.at[index, 'wikilink'] = wikilink

    # Get uri to index mapping
    uri_index = {uri: idx for idx, uri in df.uri.iteritems()}

    # Get entities with no description and comment but has a wikipedia link.
    df_only_wiki = df[~df.description.notna() & ~df.comment.notna() & df.wikilink.notna()]

    url_index = {url.split('?')[0].replace('http', 'https'): uri_index[uri] for uri, url in zip(df_only_wiki.uri, df_only_wiki.wikilink)}

    # Query Wikipedia for entity descriptions
    for descriptions in get_text_descriptions(list(url_index.keys())):
        for url, desc in descriptions.items():
            df['description'][url_index[url]] = desc.get('WIKI_DESCRIPTION')

    print(f'Could not find names for {sum(df.name.isna())}')
    df = df[df.name.notna()]  # Could not find relevant information
    df = df.drop_duplicates()
    df.to_csv(os.path.join(path, 'entities.csv'), index=False)


if __name__ == '__main__':
    args = parser.parse_args()
    path = args.out_path

    # download_file_from_google_drive(path)
    #
    # unpack(path)

    download_entity_information(path)
