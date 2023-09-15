# GInRec: A Gated Architecture for Inductive Recommendation using Knowledge Graphs
## Table of content
1. [Installation](#installation)
2. [Running experiments](#running-experiments)
   1. [Training](#training)
   2. [Evaluation](#evaluation)
3. [Results](#results)
4. [General information about the framework](#general-information-about-the-framework)
5. [NOTICE](#notice)
   1. [Models](#models)
   2. [DGL](#dgl)

## Model
This project contains the model GInRec, and inductive recommender for both users and items.
![Go to images to find pdf full resulution version.](/images/model.png)

## Installation <a id="installation"/>
Python version. 3.8 all other requirements are located in requirements.txt. We highly recommend using a virtual 
environment.

In order to use the code you should first download datasets. As some query changing knowledge bases results are bound 
to differ. **Exact datasets and results will be uploaded after submission.** 

For local run, install all required packages (possibly install wheel before):
 ```
 pip3 install -r requirements.txt
 ```

## Running experiments<a id="running-experiments"/>
### Training<a id="training"/>
The methods can be run using:
```
python3 train/dgl_trainer.py --out_path ./results --data ./datasets --experiments ml_mr_1m_warm_start --include ginrec --test_batch 1024 --workers 1
```

For help with commands write:
```
python3 train/dgl_trainer.py --help
```

The method will automatically tune hyperparameters, otherwise the method can be set to gridsearch in the under 
configuration/models.py, where setting the keyword 'gridsearch' to `True` as in PPR. All models can be found in the 
configuration/models.py file, all datasets in the configuration/datasets.py file, and all experiments in the 
configuration/experiments.py file. Note that a dataset is a single instance of a dataset, while there might be multiple 
experiments for a single dataset. E.g., we have the full MovieLens dataset ml-mr and the subsampled MovieLens dataset 
ml-mr-1m. We then have multiple experiments for each of these datasets both warm and cold start. 

We can use the `--parameter` option to use a parameter tuned at another dataset. E.g, if we have tuned a model on 
the ml-mr-1m dataset and want to train on the ab dataset, 
use `--experiments ab_warm_start --parameter ml_mr_1m_warm_start`.

### Evaluation<a id="evaluation"/>
Evaluation requires two steps, first running evaluate/dgl_evaluator.py and then metric_calculator.py. Similarly to the 
trainer you can use the `--help` argument if in doubt. Examples:
```
--data ./datasets --results ./results --experiments ml_mr_1m_warm_start --include ginrec --test_batch 1024
```

#### Cold start evaluation
The ml-mr-1m dataset is a subset of the ml-mr dataset. It is therefore possible to train on the subset and evaluate on 
the full dataset. When running the evaluator script there are multiple arguments. For the coldstart we have: 

```
--state - use state from another experiment - useful for coldstart methods
--parameter - use parameters from another experiment - useful for training with pre-determined parameters
--parameter_usage - when parameter argument was used. 0) it was not passed during training (default) and 1) it was passed during training.
--other_model - use parameters from other method i.e. change one parameter
--other_model_usage - where other model was used: 0) location for parameters, 1) location of state, 2) both. E.g., if you want to load parameters from another model but use state from the model itself use 1.
```

For example, lets say we have tuned a model on the ml-mr-1m dataset, trained it on the ab-s dataset using those 
parameter (see last example of the training section) and want to evaluate in the coldstart setting of ab. We would then 
have to pass the arguments as:

```
--experiment ab_user_cold_start --state ab_warm_start --parameter ml_mr_1m_warm_start --parameter_usage 1
```

If we do not set parameter usage as `1` we might end up using the wrong state as it only uses the parameters for 
evaluation, e.g. tuned on the ml-mr-1m dataset and evaluating on the ml-mr dataset. We wouldn't need to retrain on the 
ml-mr dataset in the cold-start setting and therefore did not use the parameter option during training. 

The `other_model` and `other_model_usage` are similar but are primarily used for ablation studies. 

## General information about the framework<a id="general-information-about-the-framework"/>
Under configuration you will find all parameters for the models as well as dataset and experiment configuration. 
For datasets you can define a mapping of ratings, such as explicit to implicit and apply filters, such as minimal number 
of ratings using CountFilter:
```
ml_mr_1m = DatasetConfiguration('ml-mr-1m', lambda x: {x > 3: 1}.get(True, 0),
                                filters=[CountFilter(lambda c: c >= 5, sentiment=Sentiment.POSITIVE)],
                                max_users=12500)
```

All methods, experiments and datasets must be defined in these folders before use. All methods are located in the models 
folder and inherit from the RecommenderBase class in dgl_recommender_base.py

Look in datsets folder for the construction of datasets.

## NOTICE<a id="notice"/>
Section describing modifications to documents under the Apache License or for recognition.
Both IGMC and NGCF are originally under the Apache License.
### Models<a id="models"/>
#### IGMC
The IGMC model is a modified version of Based on https://github.com/LspongebobJH/dgl/blob/pr1815_igmc/examples/pytorch/igmc/
For download: https://github.com/LspongebobJH/dgl/archive/refs/heads/pr1815_igmc.zip

Specifically, it has been implemented with the RecommenderBase and uses a DGL collator for training.

#### KGAT
Heavily inspired by the work of jennyzhang0215 in https://github.com/jennyzhang0215/DGL-KGAT.

#### NGCF and LightGCN
Based on the example of NGCF in Based on https://github.com/dmlc/dgl/tree/master/examples/pytorch/NGCF modified to 
include LightGCN and using blocks.

### DGL<a id="dgl"/>
Under models/shared/dgl_dataloader.py, you will find changes of the DGL EdgeCollator classes and 
EdgeDataloader to skip certain graph building steps. 
