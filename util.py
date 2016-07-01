import re

import numpy as np
import os
import pickle

def df_sample_n(data, n, seed=None):
    rng = np.random.RandomState(seed)
    N = data.shape[0]
    return data.iloc[rng.choice(range(N), n)]

def with_cache(cache_file, g):
    if os.path.exists(cache_file):
        return pickle.load(open(cache_file, 'rb'))
    else:
        res = g()
        pickle.dump(res, open(cache_file, 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)
        return res

def list_to_location_map(l):
    return dict(zip(l, range(len(l))))

from collections import OrderedDict

class DataFrameNDArrayWrapper:

    def __init__(self, df):
        self.d = df.values
        self.row_names = df.index.tolist()
        self.column_names = df.columns.tolist()
        self.row_name_to_loc = list_to_location_map(self.row_names)
        self.column_name_to_loc = list_to_location_map(self.column_names)

    def get_row_as_dict(self, row_name):
        i = self.row_name_to_loc[row_name]
        row = self.d[i]
        return OrderedDict(zip(self.column_names, row))

def word_ngrams(tokens, ngram_range, stop_words=None, binary=False):
    """Turn tokens into a sequence of n-grams after stop words filtering"""
    # handle stop words
    if stop_words is not None:
        tokens = [w for w in tokens if w not in stop_words]

    # handle token n-grams
    min_n, max_n = ngram_range
    if max_n != 1:
        original_tokens = tokens
        if binary:
            tokens = set()
        else:
            tokens = []
        n_original_tokens = len(original_tokens)
        for n in range(min_n,
                        min(max_n + 1, n_original_tokens + 1)):
            for i in range(n_original_tokens - n + 1):
                token = tuple(original_tokens[i: i + n])
                if binary:
                    tokens.add(token)
                else:
                    tokens.append(token)

    return tokens

_white_spaces = re.compile(r"\s\s+")
def char_ngrams(text_document, ngram_range, binary=False):
    """Tokenize text_document into a sequence of character n-grams"""
    # normalize white spaces
    text_document = _white_spaces.sub(" ", text_document)

    text_len = len(text_document)
    if binary:
        ngrams = set()
    else:
        ngrams = []
    min_n, max_n = ngram_range
    for n in range(min_n, min(max_n + 1, text_len + 1)):
        for i in range(text_len - n + 1):
            token = text_document[i: i + n]
            if binary:
                ngrams.add(token)
            else:
                ngrams.append(token)
    return ngrams

def binary_matrix_to_int(a):
    return a.dot(1 << np.arange(a.shape[1]))

def jaccard(a, b):
    a = set(a)
    b = set(b)
    n_intersection = len(a.intersection(b))
    n_union = len(a) + len(b) - n_intersection
    if n_union == 0:
        return 0.0
    else:
        return n_intersection / n_union