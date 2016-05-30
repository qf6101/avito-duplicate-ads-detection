import pandas as pd
import os
import pickle
import json

__all__ = ['item_info_train', 'item_pairs_train',
           'item_info_test', 'item_pairs_test',
           'category_parent', 'location_region'
           ]

dir_here = os.path.dirname(os.path.realpath(__file__))
data_file_dir = os.path.join(dir_here, 'data_files')
cache_dir = os.path.join(dir_here, 'cache')


def with_cache(cache_file, g):
    if os.path.exists(cache_file):
        return pd.read_pickle(cache_file)
    else:
        res = g()
        pickle.dump(res, open(cache_file, 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)
        return res


def load_file_with_cache(name, reader, f):
    return with_cache(os.path.join(cache_dir, name + '.pickle'),
                      lambda: reader(os.path.join(data_file_dir, f)))


def read_item_info(f):
    res = pd.read_csv(f, index_col='itemID')
    res['images_array'] = res['images_array'].apply(lambda x: list(map(int, x.split(','))) if pd.notnull(x) else [])
    res['attrsJSON'] = res['attrsJSON'].apply(lambda x: json.loads(x) if isinstance(x, str) else {})

    return res


item_info_test = load_file_with_cache('item_info_test', read_item_info, 'ItemInfo_test.csv')
item_info_train = load_file_with_cache('item_info_train', read_item_info, 'ItemInfo_train.csv')
item_pairs_test = load_file_with_cache('item_pairs_test', pd.read_csv, 'ItemPairs_test.csv')
item_pairs_train = load_file_with_cache('item_pairs_train', pd.read_csv, 'ItemPairs_train.csv')
category_parent = load_file_with_cache('category_parent_dict', pd.read_csv, 'Category.csv')
location_region =load_file_with_cache('location_region', pd.read_csv, 'Location.csv')

