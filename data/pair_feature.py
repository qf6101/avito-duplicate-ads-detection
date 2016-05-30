from .item import *
import pandas as pd

__all__ = ['gen_simple_feature', 'gen_features']

def jaccard(a, b):
    a = set(a)
    b = set(b)
    n_intersection = len(a.intersection(b))
    n_union = len(a) + len(b) - n_intersection
    if n_union == 0:
        return 0.0
    else:
        return n_intersection / n_union

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

def gen_features(g, pairs):
    X = []
    for i, j in zip(pairs['itemID_1'], pairs['itemID_2']):
        a = get_item(i)
        b = get_item(j)
        X.append(g(a, b))
    return pd.DataFrame(X)
