#!/bin/bash

# Prerequisite: ../data/data_files/image_itemPairs_train.csv and ../data/data_files/image_itemPairs_test.csv
# To generate them, execute: cd ../data && python image_item_pair.py

# Generate image feature in parallel
function work(){
pv --rate -i 5 \
 | csvcut -c 'index,images_array_1,images_array_2' | csvjson --stream \
  | parallel --gnu --linebuffer --pipe -N 10  --jobs 16 --round-robin python ../feature/image_feature.py

}

# Generate image feature for training data set and testing data set
cat ../data/data_files/image_itemPairs_train.csv | work > ../data/data_files/image_feature_train.jsonl
cat ../data/data_files/image_itemPairs_test.csv | work > ../data/data_files/image_feature_test.jsonl
# Sort image feature files (json format) and save to csv files
python ../feature/sort_image_feature.py -f ../data/data_files/image_feature_train.jsonl
python ../feature/sort_image_feature.py -f ../data/data_files/image_feature_test.jsonl