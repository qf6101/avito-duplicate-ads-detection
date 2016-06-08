#!/bin/bash

function work(){
pv --rate -i 5 \
 | csvcut -c 'itemID,title,description' | csvjson --stream \
  | parallel --gnu -k --pipe --block 10M  --jobs 4 ./feature/text_preprocess.py

}
cat ./data/data_files/ItemInfo_train.csv | work > ./data/data_files/ItemInfo_preprocessed.jsonl
cat ./data/data_files/ItemInfo_test.csv | work >> ./data/data_files/ItemInfo_preprocessed.jsonl
