import pandas as pd

__all__ = ['join_item_pair']


def join_image_item_pair(item_pair_file, item_info_file):
    """
    Expand item_pair with images_array_1 and images_array_2 using two merge operations
    :param item_pair_file: raw item pair file from kaggle site
    :param item_info_file: raw item info file form kaggle site
    :return: dataframe with columns of index, itemID_1, itemID_2, images_array_1, images_array_2
    """
    item_pairs = pd.read_csv(item_pair_file)
    item_infos = pd.read_csv(item_info_file)
    item_pairs = item_pairs.reset_index()
    item1_merged = pd.merge(item_pairs, item_infos, left_on='itemID_1', right_on='itemID')
    item1_merged = item1_merged[['index', 'itemID_1', 'itemID_2', 'images_array']]
    item12_merged = pd.merge(item1_merged, item_infos, left_on='itemID_2', right_on='itemID')
    item12_merged = item12_merged[['index', 'itemID_1', 'itemID_2', 'images_array_x', 'images_array_y']]
    item12_merged = item12_merged.rename(
        columns={'images_array_x': 'images_array_1', 'images_array_y': 'images_array_2'})
    item12_merged[['index', 'itemID_1', 'itemID_2']].astype(int)
    item12_merged[['images_array_1', 'images_array_2']].astype(str)
    item12_merged = item12_merged.sort_values(['index'])
    item12_merged = item12_merged.set_index('index')
    return item12_merged


if __name__ == '__main__':
    """
    Expand item pair for training data and testing data respectively
    """
    full_item_pair_train = join_image_item_pair('data_files/ItemPairs_train.csv', 'data_files/ItemInfo_train.csv')
    full_item_pair_train.to_csv('data_files/image_itemPairs_train.csv')
    full_item_pair_test = join_image_item_pair('data_files/ItemPairs_test.csv', 'data_files/ItemInfo_test.csv')
    full_item_pair_test.to_csv('data_files/image_itemPairs_test.csv')
