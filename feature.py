__all__ =  ['gen_simple_feature', 'gen_title_feature']

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

def tokenize(x):
    return x.split()

def gen_text_similarity_feature(sa, sb, prefix=''):
    if not isinstance(sa, str) or not isinstance(sb, str):
        return {}
    wa = tokenize(sa)
    wb = tokenize(sb)
    feats = {}
    feats[prefix+'word_jaccard'] = jaccard(wa, wb)
    return feats

def gen_title_feature(a, b):
    return gen_text_similarity_feature(a['title'], b['title'], 'title_')
