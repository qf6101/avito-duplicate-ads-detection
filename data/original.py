import pandas as pd
import json

from . import load_file_with_cache

__all__ = ['item_info_train', 'item_pairs_train',
           'item_info_test', 'item_pairs_test',
           'category_parent', 'location_region',
           'get_item', 'show_item', 'show_item_pair',
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


import jinja2
from IPython.display import HTML

env = jinja2.Environment(loader=jinja2.FileSystemLoader('templates'))
item_info_template = env.get_template('item_info.html')
item_pair_template = env.get_template('item_pair.html')

def get_item(itemID):
    if itemID in item_info_test.index:
        item = item_info_test.loc[itemID]
    else:
        item = item_info_train.loc[itemID]
    return item

def show_item(itemID):
    item = get_item(itemID)
    return HTML(item_info_template.render(itemID=itemID, item=item))

def show_item_pair(itemID1, itemID2):
    item1 = get_item(itemID1)
    item2 = get_item(itemID2)
    return HTML(item_pair_template.render(itemID1=itemID1, item1=item1,
                                          itemID2=itemID2, item2=item2))

