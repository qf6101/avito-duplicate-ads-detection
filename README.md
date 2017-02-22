# kaggle competition solution: avito-duplicate-ads-detection

code and solution for [kaggle: Avito Duplicate Ads Detection](https://www.kaggle.com/c/avito-duplicate-ads-detection) (team luoq)

## detailed solution

Please read [solution.md](solution.md)

## slide

[A slide to discuss this solution](https://luoq.github.io/slide/avito-duplicate-ads-detection_review.html)

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
  * [datatrek](https://github.com/luoq/datatrek): some self made utility code
* xgboost with python interface
* mxnet with python interface

A GPU is highly recommended to run mxnet. It takes about 5 days to generate the features.

## how to run the code

1. extract data(except image) to data/data_files
2. cp config.example.json to config.json; change the config to match the data dir
3. change working dir to root of this repo
4. run [prepare_data.sh]() to generate features
5. run [leaderboad_solution.py]() to generate final solution
