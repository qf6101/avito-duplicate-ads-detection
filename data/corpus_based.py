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

from . import data_file_dir, generate_with_cache, get_data_file, get_cache_file

# from .original import item_pairs_train, item_pairs_test
# from .item import item_id_to_index

__all__ = ['title_idf_diff']

n_doc = 4659818

from datatrek.make import PickleNode, RootNode

preprocessed_text_file = get_data_file('ItemInfo_preprocessed.jsonl')
preprocessed_text = RootNode([preprocessed_text_file])
dfs_file = get_data_file('df.pickle')
dfs = RootNode([dfs_file])

item_pairs_train_file = get_cache_file('item_pairs_train.pickle')
item_pairs_test_file = get_cache_file('item_pairs_train.pickle')


class PairRelation(PickleNode):
    def __init__(self):
        super(PairRelation, self).__init__(get_cache_file('pair_relation.pickle'),
                                           [RootNode([item_pairs_train_file, item_pairs_test_file])])

    def compute(self):
        from .item import item_id_to_index
        from .original import item_pairs_train, item_pairs_test

        I = []
        J = []
        for a, b in chain(zip(item_pairs_train.itemID_1, item_pairs_train.itemID_2),
                          zip(item_pairs_test.itemID_1, item_pairs_test.itemID_2)):
            I.append(item_id_to_index[a])
            J.append(item_id_to_index[b])

        self.pairs = (I, J)

    def decorate_data(self):
        return self.pairs


pair_relation = PairRelation()


def collect_tokens(x):
    res = []
    for y in x:
        res.extend(y)
    return res


class DocumentTermMatrix(PickleNode):
    def __init__(self, name, slot, lower=True):
        super(DocumentTermMatrix, self).__init__(get_cache_file(name + '.pickle'),
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

    def decorate_data(self, matrix_only=False):
        if matrix_only:
            return self.dtm
        else:
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
        super(DocumentTermMatricFilter, self).__init__(get_cache_file(name + '.pickle'),
                                                       [src])
        self.name = name
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

    def decorate_data(self, matrix_only=False):
        if matrix_only:
            return self.dtm
        else:
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

    def decorate_data(self, matrix_only=False):
        if matrix_only:
            return self.embedding
        else:
            return self.embedding, self.model

from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Binarizer
l2_normalizer_inplace = Normalizer(norm='l2', copy=False)
binarizer = Binarizer()
tfidf_transformer = TfidfTransformer()

class CosineSimilarityFeature(PickleNode):
    def __init__(self, name, vec_models):
        self.vec_models = vec_models
        super(CosineSimilarityFeature, self).__init__(get_cache_file(name + '.pickle'), vec_models + [pair_relation])

    def compute(self):
        I, J = pair_relation.get_data()
        feats = {}
        for m in self.vec_models:
            feat_names = [m.name+'__'+version for version in ['tf', 'binary_tf', 'tfidf', 'binary_tfidf']]
            V0 = m.get_data(matrix_only=True)
            V1 = binarizer.fit_transform(V0)
            V2 = tfidf_transformer.fit_transform(V0)
            V3 = tfidf_transformer.fit_transform(V1)
            for k,V in zip(feat_names, [V0, V1, V2, V3]):
                V = l2_normalizer_inplace.fit_transform(V)
                if sp.issparse(V):
                    feats[k] = np.array((V[I].multiply(V[J])).sum(axis=1)).ravel()
                else:
                    feats[k] = (V[I] * V[J]).sum(axis=1)

        self.feats = pd.DataFrame(feats)

    def decorate_data(self):
        return self.feats


from sklearn.decomposition import TruncatedSVD, NMF

from sklearn.pipeline import Pipeline

title_word_dtm_0 = DocumentTermMatrix('title_word_dtm_0', slot=('word_stemmed_ngram', True, 'title'), lower=True)
title_word_dtm_1 = DocumentTermMatricFilter('title_word_dtm_1', title_word_dtm_0, WordFilter.contain_alphabet)
description_word_dtm_0 = DocumentTermMatrix('description_word_dtm_0', slot=('word_stemmed_ngram', True, 'description'),
                                            lower=True)
description_word_dtm_1 = DocumentTermMatricFilter('description_word_dtm_1', description_word_dtm_0,
                                                  WordFilter.contain_alphabet)
title_word_lsa_1_0 = RepresentationModel('title_word_lsa_1_0', title_word_dtm_1,
                                         model=TruncatedSVD(n_components=100, n_iter=20, random_state=0),
                                         dtm_transformer=Binarizer(copy=False)
                                         )
title_word_lsa_1_1 = RepresentationModel('title_word_lsa_1_1', title_word_dtm_1,
                                         model=TruncatedSVD(n_components=100, n_iter=20, random_state=1),
                                         dtm_transformer=Pipeline([('binarizer', Binarizer(copy=False)),
                                                                   ('tfidf_transformer', TfidfTransformer())])
                                         )
description_word_lsa_1_0 = RepresentationModel('description_word_lsa_1_0', description_word_dtm_1,
                                               model=TruncatedSVD(n_components=100, n_iter=20, random_state=2),
                                               dtm_transformer=Binarizer(copy=False)
                                               )
description_word_lsa_1_1 = RepresentationModel('description_word_lsa_1_1', description_word_dtm_1,
                                               model=TruncatedSVD(n_components=100, n_iter=20, random_state=3),
                                               dtm_transformer=Pipeline([('binarizer', Binarizer(copy=False)),
                                                                         ('tfidf_transformer', TfidfTransformer())])
                                               )
description_word_lsa_1_2 = RepresentationModel('description_word_lsa_1_2', description_word_dtm_1,
                                               model=TruncatedSVD(n_components=100, n_iter=20, random_state=3)
                                               )
description_word_lsa_1_3 = RepresentationModel('description_word_lsa_1_3', description_word_dtm_1,
                                               model=TruncatedSVD(n_components=100, n_iter=20, random_state=3),
                                               dtm_transformer=TfidfTransformer()
                                               )
title_word_nmf_1_0 = RepresentationModel('title_word_nmf_1_0', title_word_dtm_1,
                                         model=NMF(n_components=100, random_state=4),
                                         dtm_transformer=Binarizer(copy=False)
                                         )
title_word_nmf_1_1 = RepresentationModel('title_word_nmf_1_1', title_word_dtm_1,
                                         model=TruncatedSVD(n_components=100, random_state=5),
                                         dtm_transformer=Pipeline([('binarizer', Binarizer(copy=False)),
                                                                   ('tfidf_transformer', TfidfTransformer())])
                                         )
description_word_nmf_1_0 = RepresentationModel('description_word_nmf_1_0', description_word_dtm_1,
                                               model=NMF(n_components=100, random_state=4),
                                               dtm_transformer=Binarizer(copy=False)
                                               )
description_word_nmf_1_1 = RepresentationModel('description_word_nmf_1_1', description_word_dtm_1,
                                               model=TruncatedSVD(n_components=100, random_state=5),
                                               dtm_transformer=Pipeline([('binarizer', Binarizer(copy=False)),
                                                                         ('tfidf_transformer', TfidfTransformer())])
                                               )

models = [title_word_dtm_0, title_word_dtm_1, description_word_dtm_0, description_word_dtm_1,
          title_word_lsa_1_0, title_word_lsa_1_1,
          description_word_lsa_1_0, description_word_lsa_1_1, description_word_lsa_1_2, description_word_lsa_1_3
          # title_word_nmf_1_0, title_word_nmf_1_1, description_word_nmf_1_0, description_word_nmf_1_1,
          ]
cosine_similarity_features = CosineSimilarityFeature('cosine_similarity_features',
                                                     models)
feats = [cosine_similarity_features]