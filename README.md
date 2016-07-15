# avito-duplicate-ads-detection

code and solution for [kaggle: Avito Duplicate Ads Detection](https://www.kaggle.com/c/avito-duplicate-ads-detection) (team luoq)

## solution

Please read [solution.md]()

## environment setup

The base environment is linux with [Anaconda3](https://www.continuum.io/downloads)

A lot of extra libraries are needed to run this code, an incomprehensive list is
* python library
  * opencv3
  * imagehash
  * gensim
  * nltk
  * [pystemmer](https://github.com/snowballstem/pystemmer)
  * [python-Levenshtein](https://pypi.python.org/pypi/python-Levenshtein)
* xgboost with python interface
* mxnet with python interface

A GPU is highly recommended to run mxnet. It takes about 5 days to generate the features.

## how to run the code

1. extract data(except image) to data/data_files
2. cp config.example.json to config.json; change the config to match the data dir
3. change working dir to root of this repo
4. run [prepare_data.sh]() to generate features
5. run [leaderboad_solution.py]() to generate final solution
