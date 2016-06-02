from .item import item_info
from . import generate_with_cache
from collections import OrderedDict
from math import nan

__all__ = ['aggregation_feature_map', 'gen_aggregation_feature']

def gen_aggregation_feature_map():
    aggregation_feature_map = OrderedDict()
    for col in ['title', 'categoryID', 'locationID']:
        aggregation_feature_map[col] = OrderedDict([
            ('freq', item_info[col].value_counts(normalize=True).to_dict()),
            ('price_mean', item_info.groupby(col)['price'].mean().to_dict()),
            ('price_median', item_info.groupby(col)['price'].median().to_dict()),
            ('price_std', item_info.groupby(col)['price'].std().to_dict()),
        ])
    return aggregation_feature_map

aggregation_feature_map = generate_with_cache('aggregation_feature_map', gen_aggregation_feature_map)


def gen_aggregation_feature(a, b):
    feats = OrderedDict()

    name = 'categoryID'
    for k, m in aggregation_feature_map[name].items():
        feats[name+'_'+k] = m.get(a[name], nan)

    for name in ['title', 'locationID']:
        for k, m in aggregation_feature_map[name].items():
            feats[name+'_1_'+k] = m.get(a[name], nan)
            feats[name+'_2_'+k] = m.get(b[name], nan)

    return feats