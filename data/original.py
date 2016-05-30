import pandas as pd
import json

from . import load_file_with_cache

__all__ = ['item_info_train', 'item_pairs_train',
           'item_info_test', 'item_pairs_test',
           'category_parent', 'location_region',
           ]


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
