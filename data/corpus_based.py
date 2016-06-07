import pickle
import numpy as np
import scipy.sparse as sp
import pandas as pd
import os
from . import data_file_dir,  generate_with_cache
from .original import item_pairs_train, item_pairs_test
from .item import item_id_to_index
import nltk
import re
from itertools import chain, islice

__all__ = ['title_idf_diff']

n_doc = 4659818


def gen_title_idf_diff():
    dfs = pickle.load(open((os.path.join(data_file_dir, 'df.pickle')), 'rb'))
    df = dfs[('word_ngram', False, 'title')]

    tmp = pd.read_pickle(os.path.join(data_file_dir, 'word_title_dtm.pickle'))
    dtm = tmp['dtm']
    words = tmp['words']

    stopwords = set(nltk.corpus.stopwords.words('russian'))

    selected_words = []
    selected = []

    for i, w in enumerate(words):
        if w not in stopwords and re.match('^[0-9\W_]*$', w) is None:
            selected_words.append(w)
            selected.append(i)
    dtm = dtm[:, selected]
    df = np.array([df[w] for w in selected_words])
    idf = np.log(n_doc) - np.log(df)

    dtm.data.fill(1)
    dtm = dtm * sp.diags(idf) #weight column by idf

    feats = {}
    for id1, id2 in chain(zip(item_pairs_train.itemID_1, item_pairs_train.itemID_2),
                          zip(item_pairs_test.itemID_1, item_pairs_test.itemID_2)):
        i1 = item_id_to_index[id1]
        i2 = item_id_to_index[id2]

        v1 = dtm[i1]
        v2 = dtm[i2]

        feats['title_cosine']  = v1.dot(v2)/(v1.dot(v1)*v2.dot(v2))**(1/2)

        diff = v1 - v2
        diff1 = diff.maximum(0).max()
        diff2 = -diff.minimum(0).min()

        feats['title_idf_diff_min'] = min(diff1, diff2)
        feats['title_idf_diff_max'] = max(diff1, diff2)

    feats = pd.DataFrame(feats, index=list(range(len(feats))))
    return feats

title_idf_diff = generate_with_cache('title_idf_diff', gen_title_idf_diff)