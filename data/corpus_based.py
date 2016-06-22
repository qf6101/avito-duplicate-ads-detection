import json
import pickle
import re
from collections import Counter, OrderedDict
from itertools import chain

import nltk
import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import get_data_file, get_cache_file

# from .original import item_pairs_train, item_pairs_test
# from .item import item_id_to_index

__all__ = ['title_idf_diff']

n_doc = 4659818

from datatrek.make import PickleNode, RootNode, VirtualNode

preprocessed_text_file = get_data_file('ItemInfo_preprocessed.jsonl')
preprocessed_text = RootNode([preprocessed_text_file])
dfs_file = get_data_file('df.pickle')
dfs = RootNode([dfs_file])

item_pairs_train_file = get_cache_file('item_pairs_train.pickle')
item_pairs_test_file = get_cache_file('item_pairs_train.pickle')

item_info = RootNode([get_cache_file('item_info_train.pickle'), get_cache_file('item_info_test.pickle')])

class ItemInfoColumn(PickleNode):
    def __init__(self, name, column):
        super(ItemInfoColumn, self).__init__(get_cache_file(name+'.pickle'), [item_info])
        self.name = name
        self.column = column

    def compute(self):
        from .item import item_info
        self.data = item_info[self.column].values

    def decorate_data(self):
        return self.data

price = ItemInfoColumn('price', 'price')


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
    def __init__(self, name, slot):
        super(DocumentTermMatrix, self).__init__(get_cache_file(name + '.pickle'),
                                                 [preprocessed_text, dfs])
        self.name = name
        self.slot = slot

    def compute(self):
        dfs = pickle.load(open('./data/data_files/df.pickle', 'rb'))
        df = dfs[self.slot]

        lower = self.slot[1]
        source = self.slot[2]
        if 'stemmed' in self.slot[0]:
            source += '_stemmed'

        words = sorted(df.keys())
        if lower:
            words = sorted(set(map(str.lower, words)))
        word_to_index = dict(zip(words, range(len(words))))

        I = []
        J = []
        V = []
        for i, line in enumerate(open(preprocessed_text_file)):
            line = json.loads(line.rstrip())
            tokens = collect_tokens(line[source])
            if lower:
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


from sklearn.feature_extraction.text import CountVectorizer


class DocumentTermMatrixFromWordCounter(PickleNode):
    def __init__(self, name, source, lower, counter_params):
        super(DocumentTermMatrixFromWordCounter, self).__init__(get_cache_file(name + '.pickle'),
                                                                [preprocessed_text])
        self.name = name
        self.source = source
        self.lower = lower
        self.counter_params = counter_params

    def _tokenizer(self, line):
        line = json.loads(line.rstrip())
        tokens = collect_tokens(line[self.source])
        if self.lower:
            tokens = list(map(str.lower, tokens))
        return tokens

    def compute(self):
        counter = CountVectorizer(tokenizer=self._tokenizer, **self.counter_params)
        self.dtm = counter.fit_transform(open(preprocessed_text_file))
        self.words = sorted(counter.vocabulary_)

    def decorate_data(self, matrix_only=False):
        if matrix_only:
            return self.dtm
        else:
            return self.words, self.dtm


class DocumentTermMatrixUnion(VirtualNode):
    def __init__(self, name, dependencies):
        super(DocumentTermMatrixUnion, self).__init__(dependencies)
        self.name = name

    def get_data(self, matrix_only=False):
        words = []
        dtms = []
        for src in self.dependencies:
            w, d = src.get_data()
            words.extend(w)
            dtms.append(d)
        dtm = sp.hstack(dtms)
        if matrix_only:
            return dtm
        else:
            return words, dtm


class WordFilter:
    stopwords = set(nltk.corpus.stopwords.words('russian'))

    @classmethod
    def none(cls, w):
        return True

    @classmethod
    def contain_alphabet(cls, w, remove_stopwords=True):
        if remove_stopwords:
            if w in cls.stopwords:
                return False
            return re.match('^[0-9\W_]*$', w) is None

    @classmethod
    def remove_stop_words(cls, w):
        return w not in cls.stopwords


import numbers


class DocumentTermMatricFilter(PickleNode):
    def __init__(self, name, src, word_filter, min_df=1):
        super(DocumentTermMatricFilter, self).__init__(get_cache_file(name + '.pickle'),
                                                       [src])
        self.name = name
        self.word_filter = word_filter
        self.min_df = min_df

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

        n_doc = dtm.shape[0]
        min_doc_count = (self.min_df
                         if isinstance(self.min_df, numbers.Integral)
                         else self.min_df * n_doc)
        if min_doc_count > 1:
            dtm_ = binarizer.fit_transform(dtm)
            dfs = np.asarray(dtm_.sum(axis=0)).ravel()
            selected = np.where(dfs >= min_doc_count)[0]
            dtm = dtm[:, selected]
            words = list(np.array(words)[selected])

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

from multiprocessing import Pool
from itertools import repeat


class VectorSimilarityFeatureBase(PickleNode):
    def compute(self):
        I, J = pair_relation.get_data()
        feats = OrderedDict()

        ## debug
        # for m in self.vec_models:
        #    feats.update(self.compute_for_model(m, I, J))
        pool = Pool(min(len(self.vec_models), 5))
        for res in pool.starmap(self.compute_for_model, zip(self.vec_models, repeat(I), repeat(J))):
            feats.update(res)

        self.feats = pd.DataFrame(feats)

    def decorate_data(self):
        return self.feats


class CosineSimilarityFeature(VectorSimilarityFeatureBase):
    def __init__(self, name, vec_models, add_variants=True):
        self.vec_models = vec_models
        super(CosineSimilarityFeature, self).__init__(get_cache_file(name + '.pickle'), vec_models + [pair_relation])
        self.add_variants = add_variants

    def compute_for_model(self, m, I, J):
        feats = OrderedDict()
        V0 = m.get_data(matrix_only=True)
        if self.add_variants:
            feat_names = [m.name + '__' + version for version in ['tf', 'binary_tf', 'tfidf', 'binary_tfidf']]
            V1 = binarizer.fit_transform(V0)
            V2 = tfidf_transformer.fit_transform(V0)
            V3 = tfidf_transformer.fit_transform(V1)
            Vs = [V0, V1, V2, V3]
        else:
            feat_names = [m.name]
            Vs = [V0]
        for k, V in zip(feat_names, Vs):
            V = l2_normalizer_inplace.fit_transform(V)
            if sp.issparse(V):
                feats[k] = np.array((V[I].multiply(V[J])).sum(axis=1)).ravel()
            else:
                feats[k] = (V[I] * V[J]).sum(axis=1)
        return feats


tfidf_transformer_nonorm = TfidfTransformer(norm=None)


class DiffTermIdfFeature(VectorSimilarityFeatureBase):
    def __init__(self, name, vec_models):
        self.vec_models = vec_models
        super(DiffTermIdfFeature, self).__init__(get_cache_file(name + '.pickle'), vec_models + [pair_relation])

    def compute_for_model(self, m, I, J):
        feats = OrderedDict()
        V = m.get_data(matrix_only=True)
        assert sp.issparse(V)
        V = binarizer.fit_transform(V)
        V = tfidf_transformer_nonorm.fit_transform(V)
        D = V[I] - V[J]
        d1 = np.array(D.maximum(0).max(axis=1).todense())
        d2 = np.array(-D.minimum(0).min(axis=1).todense())
        d = np.concatenate((d1, d2), axis=1)
        d.sort(axis=1)

        feats[m.name + '__' + 'max_disjoint_idf_min'] = d[:, 0]
        feats[m.name + '__' + 'max_disjoint_idf_max'] = d[:, 1]

        return feats

class PredictionFeature(PickleNode):
    def __init__(self, name, vec_model, model, y, y_transformer=None):
        super(PredictionFeature, self).__init__(get_cache_file(name+'.pickle'), [vec_model]+[y, pair_relation])
        self.name = name
        self.model = model
        self.y_transformer = y_transformer

    def compute(self):
        X = self.dependencies[0].get_data(matrix_only=True)
        y = self.dependencies[1].get_data()
        if self.y_transformer is not None:
            y = self.y_transformer(y)
        self.model.fit(X, y)
        self.prediction_ = self.predict(X)

        I, J = pair_relation.get_data()
        feats = OrderedDict()
        feats[self.name+'__1'] = self.prediction_[I]
        feats[self.name+'__2'] = self.prediction_[J]
        self.feats = feats

    def decorate_data(self, feature_only=True):
        if feature_only:
            return self.feats
        else:
            return self.feats, self.prediction_, self.model


from sklearn.decomposition import TruncatedSVD, NMF

from sklearn.pipeline import Pipeline

title_word_dtm_0 = DocumentTermMatrix('title_word_dtm_0', slot=('word_stemmed_ngram', True, 'title'))
title_word_dtm_1 = DocumentTermMatricFilter('title_word_dtm_1', title_word_dtm_0, WordFilter.contain_alphabet)
title_word_dtm_2 = DocumentTermMatrix('title_word_dtm_2', slot=('word_ngram', True, 'title'))
title_word_dtm_3 = DocumentTermMatricFilter('title_word_dtm_3', title_word_dtm_2, WordFilter.remove_stop_words)
title_word_dtm_4 = DocumentTermMatricFilter('title_word_dtm_4', title_word_dtm_0, WordFilter.remove_stop_words)

title_word_2gram_dtm_0 = DocumentTermMatrixFromWordCounter('title_word_2gram_dtm_0', source='title_stemmed', lower=True,
                                                           counter_params={'ngram_range': (2, 2)})
title_word_2gram_dtm_1 = DocumentTermMatricFilter('title_word_2gram_dtm_1', title_word_2gram_dtm_0, WordFilter.none,
                                                  min_df=3)

title_word_1_2gram_dtm_0 = DocumentTermMatrixUnion('title_word_1_2gram_dtm_0',
                                                   [title_word_dtm_4, title_word_2gram_dtm_1])

description_word_dtm_0 = DocumentTermMatrix('description_word_dtm_0', slot=('word_stemmed_ngram', True, 'description'))
description_word_dtm_1 = DocumentTermMatricFilter('description_word_dtm_1', description_word_dtm_0,
                                                  WordFilter.contain_alphabet)
description_word_dtm_2 = DocumentTermMatrix('description_word_dtm_2', slot=('word_ngram', True, 'description'))
description_word_dtm_3 = DocumentTermMatricFilter('description_word_dtm_3', description_word_dtm_2,
                                                  WordFilter.remove_stop_words)
description_word_dtm_4 = DocumentTermMatricFilter('description_word_dtm_4', description_word_dtm_0,
                                                  WordFilter.remove_stop_words)

def fillna_and_log(x):
    x = x.copy()
    x[np.isnan(x)] = 0
    return np.log(1+x)

from sklearn.linear_model import Lasso
description_word_dtm_0_predict_price = PredictionFeature('description_word_dtm_0_predict_price', description_word_dtm_0,
                                                         Lasso(random_state=123), price,
                                                         y_transformer=fillna_and_log)

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
title_word_nmf_0_0 = RepresentationModel('title_word_nmf_0_0', title_word_dtm_1,
                                         model=NMF(n_components=100, random_state=4),
                                         dtm_transformer=Binarizer(copy=False)
                                         )
title_word_nmf_0_1 = RepresentationModel('title_word_nmf_0_1', title_word_dtm_1,
                                         model=NMF(n_components=100, random_state=5, max_iter=30),
                                         dtm_transformer=Pipeline([('binarizer', Binarizer(copy=False)),
                                                                   ('tfidf_transformer', TfidfTransformer())])
                                         )
description_word_nmf_4_0 = RepresentationModel('description_word_nmf_4_0', description_word_dtm_1,
                                               model=NMF(n_components=100, random_state=6, max_iter=30),
                                               dtm_transformer=Binarizer(copy=False)
                                               )
description_word_nmf_4_1 = RepresentationModel('description_word_nmf_4_1', description_word_dtm_1,
                                               model=NMF(n_components=100, random_state=7, max_iter=30),
                                               dtm_transformer=Pipeline([('binarizer', Binarizer(copy=False)),
                                                                         ('tfidf_transformer', TfidfTransformer())])
                                               )

cosine_similarity_features = CosineSimilarityFeature('cosine_similarity_features',
                                                     [title_word_dtm_0, title_word_dtm_1, title_word_dtm_2,
                                                      title_word_dtm_3, title_word_dtm_4, description_word_dtm_0,
                                                      description_word_dtm_1,
                                                      description_word_dtm_2, description_word_dtm_3,
                                                      description_word_dtm_4, title_word_2gram_dtm_0,
                                                      title_word_2gram_dtm_1,
                                                      title_word_1_2gram_dtm_0,
                                                      ])

cosine_similarity_features_2 = CosineSimilarityFeature('cosine_similarity_features_2', [
    title_word_lsa_1_0, title_word_lsa_1_1,
    description_word_lsa_1_0, description_word_lsa_1_1,
    description_word_lsa_1_2, description_word_lsa_1_3,
    title_word_nmf_0_0, title_word_nmf_0_1, description_word_nmf_4_0, description_word_nmf_4_1

], add_variants=False)

diff_term_idf_features = DiffTermIdfFeature('diff_term_idf_features',
                                            [title_word_dtm_0, title_word_dtm_2, title_word_dtm_3, title_word_dtm_4,
                                             description_word_dtm_0, description_word_dtm_2, description_word_dtm_3,
                                             description_word_dtm_4]
                                            )

diff_term_idf_features_2 = DiffTermIdfFeature('diff_term_idf_features_2',
                                              [title_word_2gram_dtm_0, title_word_2gram_dtm_1])

feature_nodes = [cosine_similarity_features, cosine_similarity_features_2,
                 diff_term_idf_features, diff_term_idf_features_2, description_word_dtm_0_predict_price]


def make_all():
    for n in feature_nodes:
        n.make()
