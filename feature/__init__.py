import re
from Levenshtein import jaro_winkler, distance as edit_distance

__all__ = ['gen_simple_feature', 'gen_title_feature', 'gen_description_feature', 'gen_ncd_feature']


from util import word_ngrams, char_ngrams, jaccard


def word_jaccard_ngram(a, b, n):
    return jaccard(word_ngrams(a, (n, n), binary=True), word_ngrams(b, (n, n), binary=True))


def char_jaccard_ngram(a, b, n):
    return jaccard(char_ngrams(a, (n, n), binary=True), char_ngrams(b, (n, n), binary=True))


def create_numeric_comparison(feats, a, b, name):
    # min(nan, 1) = 1
    feats[name + '_min'] = min(a, b)
    feats[name + '_max'] = max(a, b)
    feats[name + '_diff'] = abs(a - b)
    feats[name + '_diff_ratio'] = abs(a - b) / (a + b) if a + b > 0 else 0.0

def ncf(data_x, data_y, compressor ="lzma", compressor_kargs = None, precompressed = False):
    if compressor_kargs is None:
        compressor_kargs = {}

    if compressor == "zlib":
        #print "Using ZLIB"
        from zlib import compress
        from zlib import decompress
    elif compressor == "bz2":
        #print "Using BZIP"
        from bz2 import compress
        from bz2 import decompress
    elif compressor == "lzo":
        #print "Using LZO"
        from lzo import compress
        from lzo import decompress
    elif compressor == "lzma":
        #print "Using LZMA"
        from lzma import compress
        from lzma import decompress
    elif compressor == "snappy":
        #print "Using LZMA"
        from snappy import compress
        from snappy import decompress
    else:
        raise RuntimeError('no such compressor')
    if precompressed == 0:
        c_x = len(compress(data_x, **compressor_kargs))
        c_y = len(compress(data_y, **compressor_kargs))
        c_x_y = len(compress(data_x + b' ' + data_y, **compressor_kargs))
    else:
        c_x = len(data_x)
        c_y = len(data_y)
        c_x_y = len(compress(decompress(data_x) + b' ' + decompress(data_y), **compressor_kargs))
    ncd = (c_x_y - min(c_x, c_y)) / float(max(c_x, c_y))
    return ncd


def gen_simple_feature(a, b):
    feats = {}
    for name in ['title', 'description', 'price', 'categoryID', 'locationID', 'metroID']:
        feats['same_' + name] = a[name] == b[name]
    feats['same_lat_lon'] = (a['lat'] == b['lat']) and (a['lon'] == b['lon'])
    feats['location_distance'] = ((a['lat'] - b['lat']) ** 2 + (a['lon'] - b['lon']) ** 2) ** (1 / 2)

    create_numeric_comparison(feats, a['price'], b['price'], 'price')
    try:
        create_numeric_comparison(feats, len(a['title']), len(b['title']), 'title_length')
    except TypeError:
        pass
    try:
        create_numeric_comparison(feats, len(a['description']), len(b['description']), 'description_length')
    except TypeError:
        pass
    try:
        create_numeric_comparison(feats, len(a['images_array']), len(b['images_array']), 'images_count')
    except TypeError:
        pass

    feats['attrsJSON_key_jaccard'] = jaccard(a['attrsJSON'].keys(), b['attrsJSON'].keys())
    feats['attrsJSON_item_jaccard'] = jaccard(a['attrsJSON'].items(), b['attrsJSON'].items())

    return feats


def tokenize0(x):
    return x.split()


tokenizer1_pattern = re.compile("\w+")


def tokenize1(x):
    return tokenizer1_pattern.findall(x)


def gen_text_similarity_feature(sa, sb, prefix='', ngrams_word_jaccard=[],
                                use_char_ngram_jaccard=False, ngrams_char_jaccard=[3, 4, 5]):
    if not isinstance(sa, str) or not isinstance(sb, str):
        return {}
    feats = {}

    wa0 = tokenize0(sa)
    wb0 = tokenize0(sb)
    wa1 = tokenize1(sa)
    wb1 = tokenize1(sb)

    feats[prefix + 'word0_jaccard'] = jaccard(wa0, wb0)
    feats[prefix + 'word1_jaccard'] = jaccard(wa1, wb1)

    for n in ngrams_word_jaccard:
        feats[prefix + 'word0_jaccard_{}gram'.format(n)] = word_jaccard_ngram(wa0, wb0, n)
        feats[prefix + 'word1_jaccard_{}gram'.format(n)] = word_jaccard_ngram(wa1, wb1, n)

    if use_char_ngram_jaccard:
        for n in ngrams_char_jaccard:
            feats[prefix + 'char_jaccard_{}gram'.format(n)] = char_jaccard_ngram(sa, sb, n)

    feats[prefix + 'jw'] = jaro_winkler(sa, sb)
    feats[prefix + 'edit_distance_ratio'] = edit_distance(sa, sb) / (len(sa) + len(sb))

    return feats


def gen_title_feature(a, b):
    return gen_text_similarity_feature(a['title'], b['title'],
                                       prefix='title_')


def gen_description_feature(a, b):
    return gen_text_similarity_feature(a['description'], b['description'],
                                       prefix='description_', ngrams_word_jaccard=[2, 3, 4])

from collections import OrderedDict
def gen_ncd_feature(a, b, compressors=[('bz2', None), ('snappy', None)]):
    feats = OrderedDict()
    for k in ['title', 'description']:
        if isinstance(a[k], str) and isinstance(b[k], str):
            x = a[k].encode()
            y = b[k].encode()
        else:
            continue
        for method, kargs in compressors:
            feats[k+'_'+method+'_ncd'] = ncf(x, y, compressor = method, compressor_kargs = kargs)
    return feats

