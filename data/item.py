import jinja2
from IPython.display import HTML
import pandas as pd

__all_ = ['get_item', 'show_item', 'show_item_pair', 'item_info', 'item_id_to_index']
from . import *

# object too big
# def gen_item_info_dict():
#     item_info_dict = item_info_train.to_dict('index')
#     item_info_dict.update(item_info_test.to_dict('index'))
#     return item_info_dict
#
# item_info_dict = generate_with_cache('item_info_dict.pickle', gen_item_info_dict)


from util import DataFrameNDArrayWrapper

item_info = pd.concat((item_info_train, item_info_test))
# item_info.sort_index(inplace=True)
item_info_ = DataFrameNDArrayWrapper(item_info)

item_id_to_index = dict(zip(item_info.index, range(item_info.shape[0])))

def get_item(itemID):
    return item_info_.get_row_as_dict(itemID)


env = jinja2.Environment(loader=jinja2.FileSystemLoader('templates'))
item_info_template = env.get_template('item_info.html')
item_pair_template = env.get_template('item_pair.html')


def show_item(itemID):
    item = get_item(itemID)
    return HTML(item_info_template.render(itemID=itemID, item=item))


def show_item_pair(itemID1, itemID2):
    item1 = get_item(itemID1)
    item2 = get_item(itemID2)
    return HTML(item_pair_template.render(itemID1=itemID1, item1=item1,
                                          itemID2=itemID2, item2=item2))
