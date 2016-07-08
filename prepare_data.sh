#!/usr/bin/env bash

#corpus feature
./script/text_preprocess.sh && python -m script.word_df
python -c 'from data.corpus_based import make_all; make_all()'

#image feature
(cd data && python image_item_pair.py)
./script/gen_image_histogram_feature.sh
./script/gen_image_hash_feature.sh
./script/gen_image_mxnet_feature.sh --gpu 0

#other data and feature
python -c "from data.pair_feature import *; from data.original import *"