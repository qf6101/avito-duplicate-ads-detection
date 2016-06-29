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
        super(ItemInfoColumn, self).__init__(get_cache_file(name + '.pickle'), [item_info])
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


import numbers


def _filter_min_df(words, dtm, min_df):
    n_doc = dtm.shape[0]
    min_doc_count = (min_df
                     if isinstance(min_df, numbers.Integral)
                     else min_df * n_doc)
    if min_doc_count > 1:
        dtm_ = binarizer.fit_transform(dtm)
        dfs = np.asarray(dtm_.sum(axis=0)).ravel()
        selected = np.where(dfs >= min_doc_count)[0]
        dtm = dtm[:, selected]
        words = list(np.array(words)[selected])
    return words, dtm


class SentenceRange(PickleNode):
    def __init__(self, name, slot):
        super().__init__(get_cache_file(name + '.pickle'),
                         [preprocessed_text, dfs])
        self.name = name
        self.slot = slot

    def compute(self):
        source = self.slot[2]
        if 'stemmed' in self.slot[0]:
            source += '_stemmed'

        self.sentence_range = [0]
        for line in open(preprocessed_text_file):
            line = json.loads(line.rstrip())
            self.sentence_range.append(self.sentence_range[-1] + len(line[source]))

    def decorate_data(self, as_slice=True):
        if as_slice:
            slices = []
            for i in range(len(self.sentence_range) - 1):
                slices.append(slice(self.sentence_range[i], self.sentence_range[i + 1]))
                return slices
        else:
            return self.sentence_range


class DocumentTermMatrix(PickleNode):
    def __init__(self, name, slot, sentence_as_doc=False, min_df=1):
        super(DocumentTermMatrix, self).__init__(get_cache_file(name + '.pickle'),
                                                 [preprocessed_text, dfs])
        self.name = name
        self.slot = slot
        self.min_df = min_df
        self.sentence_as_doc = sentence_as_doc

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

        def docs():
            if not self.sentence_as_doc:
                for line in open(preprocessed_text_file):
                    line = json.loads(line.rstrip())
                    tokens = collect_tokens(line[source])
                    yield tokens
            else:
                for line in open(preprocessed_text_file):
                    line = json.loads(line.rstrip())
                    yield from line[source]

        for i, tokens in enumerate(docs()):
            if lower:
                tokens = list(map(str.lower, tokens))
            for w, c in Counter(tokens).items():
                I.append(i)
                J.append((word_to_index[w]))
                V.append(c)

        dtm = sp.csr_matrix((V, (I, J)))

        words, dtm = _filter_min_df(words, dtm, self.min_df)

        self.dtm = dtm
        self.words = words

    def decorate_data(self, matrix_only=False):
        if matrix_only:
            return self.dtm
        else:
            return self.words, self.dtm


class DocumentTermMatrixFilter(PickleNode):
    def __init__(self, name, src, word_filter, min_df=1):
        super(DocumentTermMatrixFilter, self).__init__(get_cache_file(name + '.pickle'),
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

        words, dtm = _filter_min_df(words, dtm, self.min_df)

        self.dtm = dtm
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


from . import dir_here
import os


def get_word2vec_model_file(name):
    return os.path.join(dir_here, '../external_model/RusVectōrēs', name + '.model.bin.gz')


word2vec_model_web = RootNode([get_word2vec_model_file('web')])
word2vec_model_news = RootNode([get_word2vec_model_file('news')])
word2vec_model_ruscorpora = RootNode([get_word2vec_model_file('ruscorpora')])
word2vec_model_ruwikiruscorpora = RootNode([get_word2vec_model_file('ruwikiruscorpora')])

from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Binarizer

l2_normalizer = Normalizer(norm='l2', copy=True)
l1_normalizer = Normalizer(norm='l1', copy=True)
l2_normalizer_inplace = Normalizer(norm='l2', copy=False)
l1_normalizer_inplace = Normalizer(norm='l1', copy=False)
binarizer = Binarizer()
tfidf_transformer = TfidfTransformer()
import gensim


class Word2VecModel(PickleNode):
    def __init__(self, name, dtm_node, model_node):
        super(Word2VecModel, self).__init__(get_cache_file(name + '.pickle'), [dtm_node, model_node])
        self.name = name

    def compute(self):
        pass

    def get_data(self, matrix_only=True):
        model = gensim.models.Word2Vec.load_word2vec_format(self.dependencies[1].target_files[0], binary=True)
        words = np.array([x.split('_')[0] for x in model.index2word])
        V = pd.DataFrame(model.syn0, index=words)
        V = V.groupby(level=0).mean()
        words1 = np.array(V.index)

        words2, dtm = self.dependencies[0].get_data()
        words2 = np.array(words2)

        mask1 = np.in1d(words1, words2, assume_unique=True)
        mask2 = np.in1d(words2, words1, assume_unique=True)
        V = V.values[mask1, :]
        dtm = dtm[:, mask2]
        dtm = l1_normalizer_inplace.fit_transform(dtm)
        return dtm.dot(V)


from multiprocessing import Pool
from itertools import repeat


class VectorSimilarityFeatureBase(PickleNode):
    def compute(self):
        I, J = pair_relation.get_data()
        feats = OrderedDict()

        ## debug
        # for m in self.vec_models:
        #    feats.update(self.compute_for_model(m, I, J))
        if self.mp:
            pool = Pool(min(len(self.vec_models), 5))
            for res in pool.starmap(self.compute_for_model, zip(self.vec_models, repeat(I), repeat(J))):
                feats.update(res)
        else:
            for res in [self.compute_for_model(x, I, J) for x in self.vec_models]:
                feats.update(res)

        self.feats = pd.DataFrame(feats)

    def decorate_data(self):
        return self.feats


class CosineSimilarityFeature(VectorSimilarityFeatureBase):
    def __init__(self, name, vec_models, add_variants=True, mp=True):
        self.vec_models = vec_models
        super(CosineSimilarityFeature, self).__init__(get_cache_file(name + '.pickle'), vec_models + [pair_relation])
        self.add_variants = add_variants
        self.mp = mp

    def compute_for_model(self, m, I, J):
        feats = OrderedDict()
        V0 = m.get_data(matrix_only=True).copy()
        if self.add_variants:
            feat_names = [m.name + '__' + version + '__cosine' for version in
                          ['tf', 'binary_tf', 'tfidf', 'binary_tfidf']]
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
    def __init__(self, name, vec_models, mp=True):
        self.vec_models = vec_models
        super(DiffTermIdfFeature, self).__init__(get_cache_file(name + '.pickle'), vec_models + [pair_relation])
        self.mp = mp

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


class AggregatedVectorSimilarityFeatureBase(PickleNode):
    def __init__(self, name, vec_models, range, pre_normalized=False, dtm_transformer=None, dtm_transformer_name=''):
        self.vec_models = vec_models
        self.range = range
        self.pre_normalized = pre_normalized
        self.dtm_transformer = dtm_transformer
        self.dtm_transformer_name = dtm_transformer_name

        super().__init__(get_cache_file(name + '.pickle'), vec_models + [range, pair_relation])

    @staticmethod
    def _gen_aggregated_features(S):
        s0 = S.max(axis=1)
        s1 = S.max(axis=0)
        return [np.min(s0), np.max(s0), np.mean(s0), np.min(s1), np.max(s1), np.mean(s1)]

    def compute(self):
        I, J = pair_relation.get_data()
        slices = np.array(self.range.get_data())
        feats = []
        for model in self.vec_models:
            feats_per_model = []
            V = model.get_data(matrix_only=True).copy()

            if self.dtm_transformer is not None:
                V = self.dtm_transformer_name.fit_transform(V)
            if self.feature_name in ['cosine_similarity']:
                V = l2_normalizer_inplace.fit_transform(V)
            for rx, ry in zip(slices[I], slices[J]):
                S = self.similarity_matrix(V[rx], V[ry])
                if S.shape[0] > S.shape[1]:
                    S = S.T
                feats_per_model.append(self._gen_aggregated_features(S))
            feats.append(np.array(feats_per_model))
        columns = ['{}__{}__{}__{}_{}'.format(model.name, self.dtm_transformer_name, self.feature_name, i, suffix) for
                   model in self.vec_models
                   for i in
                   [0, 1] for suffix in ['min', 'max', 'mean']]
        return pd.DataFrame(np.hstack(feats),
                            columns=columns)

    def decorate_data(self):
        return self.feats


class AggregatedCosineSimilarityFeature(AggregatedVectorSimilarityFeatureBase):
    feature_name = 'cosine_similarity'

    @staticmethod
    def similarity_matrix(Va, Vb):
        # assume each row of Va, Vb as l2 norm 1
        return Va.dot(Vb.T)


class PredictionFeature(PickleNode):
    def __init__(self, name, vec_model, model, y, y_transformer=None, keep_true=False, true_name=None):
        super(PredictionFeature, self).__init__(get_cache_file(name + '.pickle'), [vec_model] + [y, pair_relation])
        self.name = name
        self.model = model
        self.y_transformer = y_transformer
        self.keep_true = keep_true
        self.true_name = true_name

    def compute(self):
        X = self.dependencies[0].get_data(matrix_only=True)
        y = self.dependencies[1].get_data()
        if self.y_transformer is not None:
            y = self.y_transformer(y)
        self.model.fit(X, y)
        self.prediction_ = self.model.predict(X)

        I, J = pair_relation.get_data()
        feats = OrderedDict()
        feats[self.name + '__1'] = self.prediction_[I]
        feats[self.name + '__2'] = self.prediction_[J]
        if self.keep_true:
            feats[self.true_name + '__1'] = y[I]
            feats[self.true_name + '__2'] = y[J]
        self.feats = pd.DataFrame(feats)

    def decorate_data(self, feature_only=True):
        if feature_only:
            return self.feats
        else:
            return self.feats, self.prediction_, self.model


from sklearn.decomposition import TruncatedSVD, NMF

from sklearn.pipeline import Pipeline

## dtm
title_word_dtm_0 = DocumentTermMatrix('title_word_dtm_0', slot=('word_stemmed_ngram', True, 'title'), min_df=3)
title_word_dtm_0_1 = DocumentTermMatrixFilter('title_word_dtm_0_1', title_word_dtm_0, WordFilter.remove_stop_words)
title_word_dtm_1 = DocumentTermMatrix('title_word_dtm_1', slot=('word_ngram', True, 'title'), min_df=3)
title_word_dtm_1_1 = DocumentTermMatrixFilter('title_word_dtm_1_1', title_word_dtm_1, WordFilter.remove_stop_words)

title_word_2gram_dtm_0 = DocumentTermMatrixFromWordCounter('title_word_2gram_dtm_0', source='title_stemmed', lower=True,
                                                           counter_params={'ngram_range': (2, 2)})
title_word_2gram_dtm_0_1 = DocumentTermMatrixFilter('title_word_2gram_dtm_0_1', title_word_2gram_dtm_0, WordFilter.none,
                                                    min_df=3)

description_word_dtm_0 = DocumentTermMatrix('description_word_dtm_0', slot=('word_stemmed_ngram', True, 'description'),
                                            min_df=3)
description_word_dtm_0_1 = DocumentTermMatrixFilter('description_word_dtm_0_1', description_word_dtm_0,
                                                    WordFilter.remove_stop_words)
description_word_dtm_1 = DocumentTermMatrix('description_word_dtm_1', slot=('word_ngram', True, 'description'),
                                            min_df=3)
description_word_dtm_1_1 = DocumentTermMatrixFilter('description_word_dtm_1_1', description_word_dtm_1,
                                                    WordFilter.remove_stop_words)

description_sentence_word_dtm_0 = DocumentTermMatrix('description_sentence_word_dtm_0',
                                                     slot=('word_stemmed_ngram', True, 'description_sentence'),
                                                     sentence_as_doc=True,
                                                     min_df=3)
description_sentence_word_dtm_0_1 = DocumentTermMatrixFilter('description_sentence_word_dtm_0_1',
                                                             description_sentence_word_dtm_0,
                                                             WordFilter.remove_stop_words)
description_sentence_word_dtm_1 = DocumentTermMatrix('description_sentence_word_dtm_1',
                                                     slot=('word_ngram', True, 'description_sentence'),
                                                     sentence_as_doc=True,
                                                     min_df=3)
description_sentence_word_dtm_1_1 = DocumentTermMatrixFilter('description_sentence_word_dtm_1_1',
                                                             description_sentence_word_dtm_1,
                                                             WordFilter.remove_stop_words)

description_sentence_range = SentenceRange('description_sentence_range',
                                           slot=('word_stemmed_ngram', True, 'description'))

description_sentence__binary__agg_cosine = AggregatedCosineSimilarityFeature(
    'description_sentence_word_dtm_0__binary__agg_cosine',
    [description_sentence_word_dtm_0, description_sentence_word_dtm_0_1, description_sentence_word_dtm_1,
     description_sentence_word_dtm_1_1],
    description_sentence_range,
    dtm_transformer=binarizer,
    dtm_transformer_name='binary_tf'
)

title_word_1_2gram_dtm_0 = DocumentTermMatrixUnion('title_word_1_2gram_dtm_0',
                                                   [title_word_dtm_0_1, title_word_2gram_dtm_0_1])
title_description_dtm_0 = DocumentTermMatrixUnion('title_description_dtm_0',
                                                  [title_word_dtm_0_1, title_word_2gram_dtm_0_1,
                                                   description_word_dtm_0_1])

## word2vec
title_word2vec_web = Word2VecModel('title_word2vec_web', title_word_dtm_0_1, word2vec_model_web)
description_word2vec_web = Word2VecModel('description_word2vec_web', description_word_dtm_0_1, word2vec_model_web)
title_word2vec_news = Word2VecModel('title_word2vec_news', title_word_dtm_0_1, word2vec_model_news)
description_word2vec_news = Word2VecModel('description_word2vec_news', description_word_dtm_0_1, word2vec_model_news)
title_word2vec_ruwikiruscorpora = Word2VecModel('title_word2vec_ruwikiruscorpora', title_word_dtm_0_1,
                                                word2vec_model_ruwikiruscorpora)
description_word2vec_ruwikiruscorpora = Word2VecModel('description_word2vec_ruwikiruscorpora', description_word_dtm_0_1,
                                                      word2vec_model_ruwikiruscorpora)
title_word2vec_ruscorpora = Word2VecModel('title_word2vec_ruscorpora', title_word_dtm_0_1, word2vec_model_ruscorpora)
description_word2vec_ruscorpora = Word2VecModel('description_word2vec_ruscorpora', description_word_dtm_0_1,
                                                word2vec_model_ruscorpora)

## lsa
title_word_lsa_0_1 = RepresentationModel('title_word_lsa_0_1', title_word_dtm_0_1,
                                         model=TruncatedSVD(n_components=100, n_iter=20, random_state=0),
                                         dtm_transformer=binarizer
                                         )
title_word_lsa_0_1_1 = RepresentationModel('title_word_lsa_0_1_1', title_word_dtm_0_1,
                                           model=TruncatedSVD(n_components=100, n_iter=20, random_state=1),
                                           dtm_transformer=Pipeline([('binarizer', binarizer),
                                                                     ('tfidf_transformer', TfidfTransformer())])
                                           )
description_word_lsa_0_1 = RepresentationModel('description_word_lsa_0_1', description_word_dtm_0_1,
                                               model=TruncatedSVD(n_components=100, n_iter=20, random_state=0),
                                               dtm_transformer=binarizer
                                               )
description_word_lsa_0_1_1 = RepresentationModel('description_word_lsa_0_1_1', description_word_dtm_0_1,
                                                 model=TruncatedSVD(n_components=100, n_iter=20, random_state=1),
                                                 dtm_transformer=Pipeline([('binarizer', binarizer),
                                                                           ('tfidf_transformer', TfidfTransformer())])
                                                 )

## nmf
title_word_nmf_0_1 = RepresentationModel('title_word_nmf_0_1', title_word_dtm_0_1,
                                         model=NMF(n_components=100, random_state=4, max_iter=30),
                                         dtm_transformer=binarizer
                                         )
title_word_nmf_0_1_1 = RepresentationModel('title_word_nmf_0_1_1', title_word_dtm_0_1,
                                           model=NMF(n_components=100, random_state=5, max_iter=30),
                                           dtm_transformer=Pipeline([('binarizer', binarizer),
                                                                     ('tfidf_transformer', TfidfTransformer())])
                                           )
description_word_nmf_0_1 = RepresentationModel('description_word_nmf_0_1', description_word_dtm_0_1,
                                               model=NMF(n_components=100, random_state=6, max_iter=30),
                                               dtm_transformer=binarizer
                                               )
description_word_nmf_0_1_1 = RepresentationModel('description_word_nmf_0_1_1', description_word_dtm_0_1,
                                                 model=NMF(n_components=100, random_state=7, max_iter=30),
                                                 dtm_transformer=Pipeline([('binarizer', binarizer),
                                                                           ('tfidf_transformer', TfidfTransformer())])
                                                 )


## prediction
def fillna_and_log(x):
    x = x.copy()
    x[np.isnan(x)] = 0
    return np.log(1 + x)


from sklearn.linear_model import SGDRegressor

title_word_1_2gram_dtm_0_predict_log_price = PredictionFeature('title_word_1_2gram_dtm_0_predict_log_price',
                                                               title_word_1_2gram_dtm_0,
                                                               SGDRegressor(penalty='elasticnet', l1_ratio=0.7,
                                                                            random_state=132, n_iter=20), price,
                                                               y_transformer=fillna_and_log, keep_true=True,
                                                               true_name='log_price')

title_description_dtm_0_predict_log_price = PredictionFeature('title_description_dtm_0_predict_log_price',
                                                              title_description_dtm_0,
                                                              SGDRegressor(penalty='elasticnet', l1_ratio=0.7,
                                                                           random_state=133, n_iter=30), price,
                                                              y_transformer=fillna_and_log)
## cosine similarity
cosine_similarity_features = CosineSimilarityFeature('cosine_similarity_features',
                                                     [title_word_dtm_0, title_word_dtm_1, title_word_dtm_0_1,
                                                      title_word_dtm_1_1, description_word_dtm_0,
                                                      description_word_dtm_1, description_word_dtm_0_1,
                                                      description_word_dtm_1_1, title_word_2gram_dtm_0,
                                                      title_word_2gram_dtm_0_1, title_word_1_2gram_dtm_0,
                                                      ], mp=False)

cosine_similarity_features_2 = CosineSimilarityFeature('cosine_similarity_features_2', [
    title_word_lsa_0_1, title_word_lsa_0_1_1,
    description_word_lsa_0_1, description_word_lsa_0_1_1,
    title_word_nmf_0_1, title_word_nmf_0_1_1,
    description_word_nmf_0_1, description_word_nmf_0_1_1
], add_variants=False, mp=False)

word2vec_cosine = CosineSimilarityFeature('word2vec_cosine',
                                          [title_word2vec_web, title_word2vec_news, title_word2vec_ruscorpora,
                                           title_word2vec_ruwikiruscorpora,
                                           description_word2vec_web, description_word2vec_news,
                                           description_word2vec_ruscorpora, description_word2vec_ruwikiruscorpora],
                                          add_variants=False, mp=False)

## diff max idf
diff_term_idf_features = DiffTermIdfFeature('diff_term_idf_features',
                                            [title_word_dtm_0, title_word_dtm_1, title_word_dtm_0_1,
                                             title_word_dtm_1_1, description_word_dtm_0,
                                             description_word_dtm_1, description_word_dtm_0_1,
                                             description_word_dtm_1_1, title_word_2gram_dtm_0, title_word_2gram_dtm_0_1]
                                            )

feature_nodes = [cosine_similarity_features, cosine_similarity_features_2,
                 diff_term_idf_features, title_word_1_2gram_dtm_0_predict_log_price,
                 title_description_dtm_0_predict_log_price, word2vec_cosine]


def make_all():
    for n in feature_nodes:
        n.make()
