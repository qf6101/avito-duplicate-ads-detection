import pickle
import scipy.sparse as sp
import json
from collections import Counter
from itertools import islice

def collect_tokens(x):
    res = []
    for y in x:
        res.extend(y)
    return res

if __name__ == '__main__':
    dfs = pickle.load(open('./data/data_files/df.pickle', 'rb'))
    word_tile_df = dfs[('word_ngram', False, 'title')]
    words = sorted(word_tile_df.keys())
    word_to_index = dict(zip(words, range(len(words))))

    I = []
    J = []
    V = []
    for i, line in enumerate(open('./data/data_files/ItemInfo_preprocessed.jsonl')):
        line = json.loads(line.rstrip())
        for w, c in Counter(collect_tokens(line['title'])).items():
            I.append(i)
            J.append((word_to_index[w]))
            V.append(c)

    M = sp.csr_matrix((V, (I, J)))

    pickle.dump({'dtm': M, 'words': words},
                open('./data/data_files/word_title_dtm.pickle', 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL
                )