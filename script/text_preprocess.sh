#!/bin/bash

function work(){
pv --rate -i 5 \
 | csvcut -c 'itemID,title,description' | csvjson --stream \
  | parallel --gnu --linebuffer --pipe -N 10  --jobs 16 --round-robin ./feature/text_preprocess.py

}
cat ./data/data_files/ItemInfo_test.csv | work > ./data/data_files/ItemInfo_test_preprocessed.jsonl
cat ./data/data_files/ItemInfo_train.csv | work > ./data/data_files/ItemInfo_train_preprocessed.jsonl
