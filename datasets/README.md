# How to create the datasets
Install pv using `apt install pv` (deb) or `yay -S pv` (arch).

## TOC
* [Step 1: Download datasets](#step-1-download-datasets)
  * [Downloading the ML dataset](#downloading-the-ml-dataset)
  * [Downloading the AB dataset](#downloading-the-ab-dataset)
* [Step 2: Convert the datasets](#step-2-convert-the-datasets)
  * [Converting the ML dataset](#converting-the-ml-dataset)
  * [Converting the AB dataset](#converting-the-ab-dataset)
  * [Step 2.1: Creating the subsampled datasets](#step-21-creating-the-subsampled-datasets)
* [Step 3: Preprocessing the datasets](#step-3-preprocessing-the-datasets)
* [Step 4: Creating the experiments](#step-4-creating-the-experiments)
  * [Example with ml-mr-1m warm start](#example-with-ml-mr-1m-warm-start)
* [Step 5: Convert to DGL](#step-5-convert-to-dgl)
* [Step 6: Create features](#step-6-create-features)
  * [Standalone embeddings](#standalone-embeddings)
  * [The special complex embedding](#the-special-complex-embedding)
  * [Combined embeddings](#combined-embeddings)
* [Step 7: Create cold start](#step-7-create-cold-start)


## Step 1: Download datasets
Under the [downloads](downloaders) you will find all dataset downloaders. 

***IMPORTANT:*** Please note you ***must*** get permission to 
download the Amazon (find [here](http://jmcauley.ucsd.edu/data/amazon/index_2014.html)) and Movielens datasets (find 
[here](https://grouplens.org/datasets/movielens/)) ***before*** running the scripts. 

### Downloading the ML dataset
Create the following folders under the datasets folder: mindreader, movielens, and ml-mr.

CD into the [downloaders](downloaders) folder and run the following scripts in order:
1. [mr-downloader.py](downloaders/mr-downloader.py)
2. [ml-downloader.py](downloaders/ml-downloader.py)(*) -- run 2-3 times for complete coverage. 
3. [ml-mr-downloader.py](downloaders/ml-mr-downloader.py)CD into [converters](converters) and find run [experiment_to_dgl.py](converters/experiment_to_dgl.py) as:

*Note: Run the returned uris may differ from the dataset used in the paper as it quries wikidata directly.

### Downloading the AB dataset
Create the following folders under the datasets folder: amazon-book.

CD into the [downloaders](downloaders) folder and run the following scripts in order:
1. [ab-downloader-kgat.py](downloaders/ab-downloader-kgat.py)
2. [ab-downloader-og.py](downloaders/ab-downloader-og.py)

## Step 2: Convert the datasets
CD to the [converters](converters) directory.

### Converting the ML dataset
Run `python mr-converter.py --path ../ml-mr`.
To get textual descriptions for all items and entities run: `python mr-kg-text.py --dataset ml-mr`. It is 
located in the [downloaders](downloaders) folder.

### Converting the AB dataset
Run the following scripts in order:
1. [ab_convert_og.py](converters/ab_converter_og.py)
2. [ab_convert_kgat.py](converters/ab_converter_kgat.py)

### Step 2.1: Creating the subsampled datasets
For the ML dataset copy the dataset using `cp -r ml-mr ml-mr-1m` and for the AB dataset run `cp -r amazon-book amazon-book-s`.

## Step 3: Preprocessing the datasets
To preprocess the datasets we use the [preprocessor](preprocessors/preprocessor.py) script found in the 
[preprocessors](preprocessors) directory. 
The script takes two arguments a `--path` argument to the location of datasets (if following this guide it would be
the datasets directory) and a `--dataset` argument, taking one of the datasets found in 
[configuration/datasets](../configuration/datasets.py).

Simply CD to [preprocessors](preprocessors) run the script with all the dataset names, e.g. 
`preprocessor.py --path .. --dataset ml-mr-1m`. 

## Step 4: Creating the experiments
For each experiment run the [partitioner](partitioners/partitioner.py) script located under [partitioners](partitioners), except for the coldstart experiments, which will be handled seperately.

All experiments can be found in the [configuration/experiments](../configuration/experiments.py) file.

### Example with ml-mr-1m warm start
We assume you have followed all the previous steps and therefore have a folder named `ml-mr-1m` with users, relations, and entities.

Run `python partitioner.py --path .. --experiment ml_mr_1m_warm_start`

Run for ml_mr_1m_warm_start, ml_mr_warm_start, ab_warm_start, ab_full_warm_start

## Step 5: Convert to DGL
For all datasets and folds we create a `DGLGraph` for all the different types of graphs, e.g., KG and Bipartite. This script furthermore, creates two numpy files called items and labels, which is an ordered set of items and their labels used for efficient ndcg calculation during validation. There aren't as many negative items (unobserved) as in this format due to the matrix format. The validation is therefore a bit higher than the testing.

CD into [converters](converters) and find run [experiment_to_dgl.py](converters/experiment_to_dgl.py) as:

```
python experiment_to_dgl.py --path .. --experiment ml_mr_1m_warm_start
```

For the full datasets you can use the argument `--graphs_only` as you probably wont be training on that dataset.

## Step 6: Create features
Any inductive model require some initial embedding unless it is learned in some way. We have two types of features. Standalone and combined. In this guide we will start with the standalone features.

All features are located in [features.py](../configuration/features.py) in the [configuration](../configuration) 
folder. There are different feature extractors for different datasets and methods. The complete list can be found here:

| Model     | MovieLens      | Amazon Book              |
|-----------|----------------|--------------------------|
| SimpleRec | graphsage      | complex & graphsage_item |
| IDCF*     | key_user_state | key_user_state           |
| GraphSAGE | graphsage      | comsage**                |
| PinSAGE   | graphsage      | graphsage_item           |

(*) Only used during evaluation.
(**) A combined embedding

### Standalone embeddings
To create an embedding for an experiment simply run [feature_extractor.py](feature_extractors/feature_extractor.py) 
in the folder [feature_extractors](feature_extractors) as:

```bash
python feature_extractor.py --path .. --experiment ab_warm_start --feature_configuration graphsage
```
Note: some methods in the published results and datasets will use 'processed' which is just the legacy name before
multiple feature extraction methods were available. 

#### The special complex embedding
`complex` uses theeen library, sadly this program overrides the torch version used for in this code. The user should therefore create a virtual environment and install pykeen. Use the requirements file found in the [feature_extractors](feature_extractors) directory. 

After running the feature extractor with the in the virtual env, switch to the default environment and continue.

### Combined embeddings
In this we will use `comsage` as an example. `comsage` consists of the `complex` and `graphsage_item` embedding joined
as one. Each of the two embeddings have to be created as *standalone* embeddings before running this step.

To create it we run: 

```
python merge_features.py --path .. --experiment ab_warm_start --feature_configurations complex graphsage_item comsage
```

TODO: ML currently does not group items and entities as amazon book. This merging therefore does not work there.
Should be fixed at some point

## Step 7: Create cold start
The cold-start experiments uses the full dataset and a subsampled. E.g., the amazon-book dataset and the amazon-book-s dataset. 
In the datastructure we have the original user, item and entity identifyers. We can therefore map them.

In this example we use amazon book. We assume you have already created the experiments `ab_full_warm_start` 
and `ab_warm_start`.

CD into [mapping](mapping) and find run [cold_start_mapping.py](mapping/cold_start_mapping.py) as:

```
python cold_start_mapping.py --path .. --experiment_a ab_warm_start --experiment_b ab_full_warm_start --new_name ab_user_cold_start 
```

Afterwards, you should see a new folder in called ab_user_cold_start under the amazon-book dataset. The program takes the argument `--n_cold_users`, which is what was used to create the ML dataset with 1250 extra users. 

Simply copy the features extracted for ab_warm_start to the ab_user_cold_start directory. 
Note that if more items of entities exists or if the users require extracted features and the features have been scaled more needs to be done, which is not standardized in this framework. Basically, one should map the scale an unscaled version of the features in cold-start setting with the scaler used in the warm-start setting. Furthermore, if a model has been trained in the warm-start setting, it should be used for feature extraction in the warm-start setting.

For the ML and AB dataset, no items or entities are removed from the KG, and the features therefore remain the same. The cold-start features can therefore be copied from the warm-start setting. 

Therefore, simply run:
```
cp [path-to-warm]/*.npy [path-to-cold]
```
E.g. `cp amazon-book-s/ab_warm_start/*.npy amazon-book/ab_user_cold_start`.

