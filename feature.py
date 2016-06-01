import re

__all__ =  ['gen_simple_feature', 'gen_title_feature', 'gen_description_feature']

def jaccard(a, b):
    a = set(a)
    b = set(b)
    n_intersection = len(a.intersection(b))
    n_union = len(a) + len(b) - n_intersection
    if n_union == 0:
        return 0.0
    else:
        return n_intersection / n_union

from util import ngrams
def jaccard_ngram(a, b, n):
    return jaccard(ngrams(a, (n, n)), ngrams(b, (n, n)))


def gen_simple_feature(a, b):
    feats = {}
    for name in ['title', 'description', 'categoryID', 'locationID', 'metroID']:
        feats['same_' + name] = a[name] == b[name]
    feats['same_lat_lon'] = (a['lat'] == b['lat']) and (a['lon'] == b['lon'])

    price_diff = abs(a['price'] - b['price'])
    feats['price_diff_ratio'] = price_diff / (a['price'] + b['price'])

    feats['attrsJSON_key_jaccard'] = jaccard(a['attrsJSON'].keys(), b['attrsJSON'].keys())
    feats['attrsJSON_item_jaccard'] = jaccard(a['attrsJSON'].items(), b['attrsJSON'].items())

    return feats

def tokenize0(x):
    return x.split()

tokenizer1_pattern = re.compile("\w+")
def tokenize1(x):
    return tokenizer1_pattern.findall(x)

def gen_text_similarity_feature(sa, sb, prefix='', ngrams=[]):
    if not isinstance(sa, str) or not isinstance(sb, str):
        return {}
    feats = {}

    wa0 = tokenize0(sa)
    wb0 = tokenize0(sb)
    wa1 = tokenize1(sa)
    wb1 = tokenize1(sb)

    feats[prefix+'word0_jaccard'] = jaccard(wa0, wb0)
    feats[prefix+'word1_jaccard'] = jaccard(wa1, wb1)

    for n in ngrams:
        feats[prefix+'word0_jaccard_{}gram'.format(n)] = jaccard_ngram(wa0, wb0, n)
        feats[prefix+'word1_jaccard_{}gram'.format(n)] = jaccard_ngram(wa1, wb1, n)

    return feats

def gen_title_feature(a, b):
    return gen_text_similarity_feature(a['title'], b['title'], 'title_')

def gen_description_feature(a, b):
    return gen_text_similarity_feature(a['description'], b['description'],
                                       prefix='description_', ngrams=[2,3,4])
