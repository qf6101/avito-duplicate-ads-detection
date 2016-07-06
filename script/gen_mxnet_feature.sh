#!/bin/bash
# This script is to generate image feature
# Prerequisite: data/data_files/image_itemPairs_train.csv and data/data_files/image_itemPairs_test.csv
# To generate them, execute: cd data && python image_item_pair.py

# Generate image feature in parallel
function work(){
 # arg_1: model selected: bn/v3/21k 
 csvcut -c 'index,images_array_1,images_array_2' | csvjson --stream \
 | python -m feature.mxnet_feature $*
}

# Generate image feature for training data set and testing data set
cat data/data_files/image_itemPairs_train.csv | work --model bn $* > data/data_files/mxnet_feature_train_bn.csv
cat data/data_files/image_itemPairs_test.csv | work --model bn $* > data/data_files/mxnet_feature_test_bn.csv
