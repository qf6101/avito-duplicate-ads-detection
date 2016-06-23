from .original import item_pairs_train, item_pairs_test
from .item import item_info
from . import generate_with_cache
import pandas as pd
import numpy as np

def mergeLeftInOrder2(x, y, *args, **kargs):
    x = x.copy()
    x["Order"] = np.arange(len(x))
    z = pd.merge(x, y, *args, **kargs).sort_values(by="Order")
    return z.drop("Order", 1)

def get_categoryID():
    item_pairs = pd.concat((item_pairs_train, item_pairs_test))
    merged = mergeLeftInOrder2(item_pairs, item_info[['categoryID']], left_on='itemID_1', right_index=True)
    return merged[['categoryID']]

def gen_dummy_features():
    res = pd.get_dummies(get_categoryID(), columns=['categoryID'])
    return res

def gen_categoryID_shuffle(seed=0, n_shuffle=20):
    categoryID = get_categoryID()
    category_id_uniq , tmp = np.unique(categoryID.categoryID.values, return_inverse=True)
    rng = np.random.RandomState(seed)
    for i in range(n_shuffle):
        ord = np.arange(len(category_id_uniq))
        rng.shuffle(ord)
        categoryID['categoryID_shuffle_'+str(i)] = ord[tmp]
    return categoryID

dummy_features = generate_with_cache('dummy_features', gen_dummy_features)
categoryID_shuffle_features = gen_categoryID_shuffle(seed=1324, n_shuffle=20)
