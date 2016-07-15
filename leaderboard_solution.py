from data.original import item_pairs_train, item_pairs_test, item_info_train, item_info_test
from data.corpus_based import feature_nodes as corpus_based_feature_nodes
import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier
import pickle

# load simple features
from data.pair_feature import *
# load feature for categoryID
from data.dummy import dummy_features, categoryID_shuffle_features

# load mxnet features
def build_mxnet_header(model_names):
    header = ["line_num"]
    for name in model_names:
        header.extend('mxnet_{}_batch_{}_sim'.format(name, x) for x in  ['min', 'max', 'summean', 'mean'])
    return header

image_mxnet_features_train = pd.read_csv('data/data_files/image_mxnet_feature_train_bn.csv', names=build_mxnet_header(['bn']), index_col='line_num').sort_index()
image_mxnet_features_test = pd.read_csv('data/data_files/image_mxnet_feature_test_bn.csv', names=build_mxnet_header(['bn']), index_col='line_num').sort_index()

# load image histogram features
from feature.image_histogram_feature import header as image_histogram_feature_header
image_histogram_features_train = pd.read_csv('data/data_files/image_histogram_feature_train.csv', names = image_histogram_feature_header, index_col='index').sort_index()
image_histogram_features_test = pd.read_csv('data/data_files/image_histogram'
                                            '_feature_test.csv', names = image_histogram_feature_header, index_col='index').sort_index()
# load image hash features
from feature.image_hash_feature import headers as image_hash_feature_headers
image_hash_features_train = pd.read_csv('data/data_files/image_hash_feature_train.csv',
                                        names=image_hash_feature_headers, index_col='index').sort_index()
image_hash_features_test = pd.read_csv('data/data_files/image_hash_feature_test.csv',
                                       names=image_hash_feature_headers, index_col='index').sort_index()

# image features
image_features_train = pd.concat((image_histogram_features_train, image_hash_features_train, image_mxnet_features_train), axis=1)
image_features_test = pd.concat((image_histogram_features_test, image_hash_features_test, image_mxnet_features_test), axis=1)

# load corpus based features
corpus_based_features = []
for node in corpus_based_feature_nodes:
    corpus_based_features.append(node.get_data())
corpus_based_features = pd.concat(corpus_based_features, axis=1)

# split to train/test for some features
def split_feats(feats):
    n_train = 2991396
    feats_train = feats[:n_train]
    feats_test = feats[n_train:]
    feats_test.index = simple_features_test.index.copy()
    return feats_train, feats_test
corpus_based_features_train, corpus_based_features_test = split_feats(corpus_based_features)
dummy_features_train, dummy_features_test = split_feats(dummy_features)
categoryID_shuffle_features_train, categoryID_shuffle_features_test = split_feats(categoryID_shuffle_features)

# all feaures
features_train = pd.concat((
    simple_features_train,
    aggregation_features_train,
    title_features_train, description_features_train,
    ncd_features_train,
    image_features_train,
    corpus_based_features_train,
    dummy_features_train,
    categoryID_shuffle_features_train,
), axis=1)
features_test = pd.concat((
    simple_features_test,
    aggregation_features_test,
    title_features_test, description_features_test,
    ncd_features_test,
    image_features_test,
    corpus_based_features_test,
    dummy_features_test,
    categoryID_shuffle_features_test
), axis=1)

# weight sample by generationMethod
weight_map = {
    1: 3,
    2: 1,
    3: 1
}
weight = np.empty(item_pairs_train.shape[0])
generationMethod = item_pairs_train.generationMethod.values
for i in [1,2,3]:
    weight[generationMethod==i] = weight_map[i]


# train model and predict
model_name = 'leaderboard_solution'
model = XGBClassifier(learning_rate=0.05, max_depth=10, subsample=0.8, colsample_bytree=0.8, n_estimators=1000,
                      min_child_weight = 1,
                      nthread=32)
model.fit(features_train, item_pairs_train.isDuplicate, sample_weight=weight)
prob_test = model.predict_proba(features_test)[:,1]

pickle.dump(model, open(model_name+'.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
pd.DataFrame({'id': item_pairs_test['id'], 'probability': prob_test}).to_csv(model_name+'.csv', index=False)