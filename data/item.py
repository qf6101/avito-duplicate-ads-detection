import jinja2
from IPython.display import HTML
import pandas as pd
from .original import item_info_train, item_info_test

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

def image_location(images_dir, image_id):
    """Get image location from image ID"""
    first_index = str(int(int(image_id) % 100 / 10))
    second_index = str(int(image_id) % 100)
    return ''.join([images_dir, "/Images_", first_index, "/", second_index, "/", str(image_id).strip(), ".jpg"])



env = jinja2.Environment(loader=jinja2.FileSystemLoader('templates'))
env.globals.update(zip=zip)
item_info_template = env.get_template('item_info.html')
item_pair_template = env.get_template('item_pair.html')


def show_item(itemID, show_image=True, image_server='http://127.0.0.1:18002/'):
    item = get_item(itemID)
    if show_image:
        image_urls = [image_location(image_server, x) for x in item['images_array']]
        return HTML(item_info_template.render(itemID=itemID, item=item, show_image=show_image, image_urls=image_urls))
    else:
        return HTML(item_info_template.render(itemID=itemID, item=item, show_image=show_image, image_urls=None))

def show_item_pair(itemID1, itemID2, show_image=True, image_server='http://127.0.0.1:18002/'):
    item1 = get_item(itemID1)
    item2 = get_item(itemID2)
    image_urls1 = None
    image_urls2 = None
    if show_image:
        image_urls1 = [image_location(image_server, x) for x in item1['images_array']]
        image_urls2 = [image_location(image_server, x) for x in item2['images_array']]
    return HTML(item_pair_template.render(itemID1=itemID1, item1=item1,
                                          itemID2=itemID2, item2=item2,
                                          show_image=show_image, image_urls1=image_urls1, image_urls2=image_urls2
                                          ))
