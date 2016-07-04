#!/usr/bin/env bash

#corpus feature
./script/text_preprocess.sh && python -m script.word_df
python -c 'from data.corpus_based import make_all; make_all()'

#image feature
(cd data && python image_item_pair.py)
(cd script && ./gen_image_feature.sh)
./script/gen_image_hash_feature.sh

#other data and feature
python -c "from data.pair_feature import *; from data.original import *"