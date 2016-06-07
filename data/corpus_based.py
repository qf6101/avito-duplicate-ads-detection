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

class RepresentationModel(PickleNode):
    def __init__(self, name, dtm_node, model, dtm_transformer=None):
        super(RepresentationModel, self).__init__(get_cache_file(name + '.pickle'), [dtm_node])
        self.name = name
        self.model = model
        self.dtm_transformer = dtm_transformer

    def compute(self):
        _, dtm = self.dependencies[0].get_data()
        if self.dtm_transformer:
            dtm = self.dtm_transformer.fit_transform(dtm)
        self.embedding = self.model.fit_transform(dtm)

    def decorate_data(self):
        return self.embedding, self.model

from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Binarizer
from sklearn.pipeline import Pipeline

title_word_dtm_0 = DocumentTermMatrix('title_word_dtm_0', slot=('word_stemmed_ngram', True, 'title'), lower=True)
title_word_dtm_1 = DocumentTermMatricFilter('title_word_dtm_1', title_word_dtm_0, WordFilter.contain_alphabet)
description_word_dtm_0 = DocumentTermMatrix('description_word_dtm_0', slot=('word_stemmed_ngram', True, 'description'), lower=True)
description_word_dtm_1 = DocumentTermMatricFilter('description_word_dtm_1', description_word_dtm_0, WordFilter.contain_alphabet)
title_word_lsa_1_0 = RepresentationModel('title_word_lsa_1_0', title_word_dtm_1,
                                         model = TruncatedSVD(n_components=500, n_iter=100, random_state=0),
                                         dtm_transformer=Binarizer(copy=False)
                                         )
title_word_lsa_1_1 = RepresentationModel('title_word_lsa_1_1', title_word_dtm_1,
                                         model = TruncatedSVD(n_components=500, n_iter=100, random_state=1),
                                         dtm_transformer=Pipeline([('binarizer', Binarizer(copy=False)),
                                                                   ('tfidf_transformer', TfidfTransformer())])
                                         )
description_word_lsa_1_0 = RepresentationModel('description_word_lsa_1_0', description_word_dtm_1,
                                         model = TruncatedSVD(n_components=500, n_iter=100, random_state=2),
                                         dtm_transformer=Binarizer(copy=False)
                                         )
description_word_lsa_1_1 = RepresentationModel('description_word_lsa_1_1', description_word_dtm_1,
                                         model = TruncatedSVD(n_components=500, n_iter=100, random_state=3),
                                         dtm_transformer=Pipeline([('binarizer', Binarizer(copy=False)),
                                                                   ('tfidf_transformer', TfidfTransformer())])
                                         )
title_word_nmf_1_0 = RepresentationModel('title_word_nmf_1_0', title_word_dtm_1,
                                         model = NMF(n_components=500, random_state=4),
                                         dtm_transformer=Binarizer(copy=False)
                                         )
title_word_nmf_1_1 = RepresentationModel('title_word_nmf_1_1', title_word_dtm_1,
                                         model = TruncatedSVD(n_components=500, random_state=5),
                                         dtm_transformer=Pipeline([('binarizer', Binarizer(copy=False)),
                                                                   ('tfidf_transformer', TfidfTransformer())])
                                         )
description_word_nmf_1_0 = RepresentationModel('description_word_nmf_1_0', description_word_dtm_1,
                                         model = NMF(n_components=500, random_state=4),
                                         dtm_transformer=Binarizer(copy=False)
                                         )
description_word_nmf_1_1 = RepresentationModel('description_word_nmf_1_1', description_word_dtm_1,
                                         model = TruncatedSVD(n_components=500, random_state=5),
                                         dtm_transformer=Pipeline([('binarizer', Binarizer(copy=False)),
                                                                   ('tfidf_transformer', TfidfTransformer())])
                                         )



models = [title_word_dtm_0, title_word_dtm_1, description_word_dtm_0, description_word_dtm_1,
          title_word_lsa_1_0, title_word_lsa_1_1, description_word_lsa_1_0, description_word_lsa_1_1,
          title_word_nmf_1_0, title_word_nmf_1_1, description_word_nmf_1_0, description_word_nmf_1_1,
          ]