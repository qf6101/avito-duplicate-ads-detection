from .original import item_pairs_train, item_pairs_test
from .item import item_info
from . import generate_with_cache
import pandas as pd
import numpy as np

def mergeLeftInOrder2(x, y, *args, **kargs):
    x = x.copy()
    x["Order"] = np.arange(len(x))
    z = pd.merge(x, y, *args, **kargs).sort("Order")
    return z.drop("Order", 1)

def gen_dummy_features():
    item_pairs = pd.concat((item_pairs_train, item_pairs_test))
    merged = mergeLeftInOrder2(item_pairs, item_info[['categoryID']], left_on='itemID_1', right_index=True)
    res = pd.get_dummies(merged[['categoryID']], columns=['categoryID'])
    return res

dummy_features = generate_with_cache('dummy_features', gen_dummy_features)
