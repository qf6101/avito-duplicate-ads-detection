#!/bin/bash

# argument_1: model selected: bn/21k/v3

# This script is to generate image feature
# Prerequisite: ../data/data_files/image_itemPairs_train.csv and ../data/data_files/image_itemPairs_test.csv
# To generate them, execute: cd ../data && python image_item_pair.py

# Generate image feature in parallel
function work(){
 # arg_1: model selected: bn/v3/21k 
 csvcut -c 'index,images_array_1,images_array_2' | csvjson --stream \
 | ../feature/mxnet_feature.py $1 
# | parallel --gnu -k --pipe -N 5  --jobs 4 --round ../feature/mxnet_feature.py
}

# Generate image feature for training data set and testing data set
# cat ../data/data_files/image_itemPairs_train.csv | work $1 > ../data/data_files/mxnet_feature_train_${1}.csv
sed '2,221835d' ../data/data_files/image_itemPairs_train.csv | work $1 >> ../data/data_files/mxnet_feature_train_${1}.csv
#cat ../data/data_files/image_itemPairs_train.csv | work > ../data/data_files/image_feature_train.csv
cat ../data/data_files/image_itemPairs_test.csv | work $1 > ../data/data_files/mxnet_feature_test_${1}.csv
