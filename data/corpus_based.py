import pickle
from collections import Counter
import numpy as np
import scipy.sparse as sp
import pandas as pd
import os
import nltk
import re
from itertools import chain, islice
import json

from . import data_file_dir,  generate_with_cache, get_data_file, get_cache_file
# from .original import item_pairs_train, item_pairs_test
# from .item import item_id_to_index

__all__ = ['title_idf_diff']

n_doc = 4659818

from datatrek.make import PickleNode, RootNode

preprocessed_text_file = get_data_file('ItemInfo_preprocessed.jsonl')
preprocessed_text = RootNode([preprocessed_text_file])
dfs_file = get_data_file('df.pickle')
dfs = RootNode([dfs_file])

def collect_tokens(x):
    res = []
    for y in x:
        res.extend(y)
    return res

class DocumentTermMatrix(PickleNode):
    def __init__(self, name, slot, lower=True):
        super(DocumentTermMatrix, self).__init__(get_cache_file(name+'.pickle'),
                                               [preprocessed_text, dfs])
        self.name = name
        self.slot = slot
        self.lower = lower

    def compute(self):
        dfs = pickle.load(open('./data/data_files/df.pickle', 'rb'))
        df = dfs[self.slot]

        source = self.slot[2]
        if 'stemmed' in self.slot[0]:
            source += '_stemmed'

        words = sorted(df.keys())
        if self.lower:
            words = sorted(set(map(str.lower, words)))
        word_to_index = dict(zip(words, range(len(words))))


        I = []
        J = []
        V = []
        for i, line in enumerate(open(preprocessed_text_file)):
            line = json.loads(line.rstrip())
            tokens = collect_tokens(line[source])
            if self.lower:
                tokens = list(map(str.lower, tokens))
            for w, c in Counter(tokens).items():
                I.append(i)
                J.append((word_to_index[w]))
                V.append(c)

        M = sp.csr_matrix((V, (I, J)))
        self.dtm = M
        self.words = words

    def decorate_data(self):
        return self.words, self.dtm

class WordFilter:
    stopwords = set(nltk.corpus.stopwords.words('russian'))
    @classmethod
    def contain_alphabet(cls, w, remove_stopwords=True):
        if remove_stopwords:
            if w in cls.stopwords:
                return False
            return re.match('^[0-9\W_]*$', w) is None


class DocumentTermMatricFilter(PickleNode):
    def __init__(self, name, src, word_filter):
        super(DocumentTermMatricFilter, self).__init__(get_cache_file(name+'.pickle'),
                                           [src])
        self.word_filter = word_filter
    def compute(self):
        words, dtm = self.dependencies[0].get_data()

        selected_words = []
        selected = []
        for i, w in enumerate(words):
            if self.word_filter(w):
                selected_words.append(w)
                selected.append(i)
        dtm = dtm[:, selected]
        words = selected_words

        self.dtm = dtm
        self.words = words
    def decorate_data(self):
        return self.words, self.dtm

title_word_dtm_0 = DocumentTermMatrix('title_word_dtm_0', slot=('word_stemmed_ngram', True, 'title'), lower=True)
title_word_dtm_1 = DocumentTermMatricFilter('title_word_dtm_1', title_word_dtm_0, WordFilter.contain_alphabet)


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

# title_idf_diff = generate_with_cache('title_idf_diff', gen_title_idf_diff)