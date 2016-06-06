import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='image_feature.log',
                    filemode='a')

__all__ = ['image_location', 'batch_image_location', 'gen_image_feature']

# root of image locations
images_dir = "/srv/data/0/ticktock/competition/kaggle/avito-duplicate-ads-detection/data/images"


def image_location(image_id):
    """Get image location from image ID"""
    first_index = str(int(int(image_id) % 100 / 10))
    second_index = str(int(image_id) % 100)
    return ''.join([images_dir, "/Images_", first_index, "/", second_index, "/", str(image_id).strip(), ".jpg"])


def batch_image_location(image_ids):
    """Get images location from image IDs"""
    img_locs = []
    for img_id in image_ids:
        img_locs.append(image_location(img_id))
    return img_locs


def compare_images_from_minority(left_img_locs, right_img_locs, comp_func):
    """Compare image similarity from the minority side"""
    if len(left_img_locs) > len(right_img_locs):
        left_img_locs, right_img_locs = right_img_locs, left_img_locs
    batch_min_diff = sys.maxsize
    batch_max_diff = -sys.maxsize
    sum_diff = 0
    for l_img_loc in left_img_locs:
        img_diff = -sys.maxsize
        left_img = cv2.imread(l_img_loc)
        if left_img is None:
            logging.error("None image: >>>>>>>>" + l_img_loc)
            continue
        for r_img_loc in right_img_locs:
            right_img = cv2.imread(r_img_loc)
            if right_img is None:
                logging.error("None image: >>>>>>>>" + r_img_loc)
                continue
            diff = comp_func(left_img, right_img)
            if diff is not np.nan and img_diff < diff:
                img_diff = diff
        if batch_min_diff > img_diff:
            batch_min_diff = img_diff
        if batch_max_diff < img_diff:
            batch_max_diff = img_diff
        sum_diff += img_diff
    return batch_min_diff, batch_max_diff, sum_diff / len(left_img_locs)


def compare_images_with(left_img_locs, right_img_locs, comp_func):
    """Compare image similarity with provided compare function"""
    return compare_images_from_minority(left_img_locs, right_img_locs, comp_func)


def compare_images(left_img_locs, right_img_locs):
    """Compare image similarity with histogram difference (USE OTHER SIMILARITY MEASURES HERE)"""
    return compare_images_with(left_img_locs, right_img_locs, hist_diff)


def gen_image_feature(left_img_arrays, right_img_arrays):
    """Generate image feature for two images arrays"""
    if len(left_img_arrays) <= 0 or len(right_img_arrays) <= 0:
        return np.nan, np.nan, np.nan
    else:
        left_img_locs = batch_image_location(left_img_arrays.split(","))
        right_img_locs = batch_image_location(right_img_arrays.split(","))
        return compare_images(left_img_locs, right_img_locs)


def hist_diff(left_img, right_img):
    """Calculate histogram difference of two images"""
    try:
        l_img_hist = cv2.calcHist(left_img, [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        r_img_hist = cv2.calcHist(right_img, [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        l_img_hist = cv2.normalize(l_img_hist, l_img_hist).flatten()
        r_img_hist = cv2.normalize(r_img_hist, r_img_hist).flatten()
        return cv2.compareHist(l_img_hist, r_img_hist, cv2.HISTCMP_CORREL)
    except:
        return np.nan


if __name__ == '__main__':
    """ Generate image feature in parallel
        Input is from stdin, output is to stdout
    """
    import sys
    import json
    from collections import OrderedDict

    jsonify = lambda x: json.dumps(x, ensure_ascii=False)

    for line in sys.stdin:
        line = json.loads(line.rstrip())
        index = int(line['index'])
        left_item = line['images_array_1']
        right_item = line['images_array_2']
        min_hist_diff, max_hist_diff, avg_hist_diff = gen_image_feature(left_item, right_item)
        res = OrderedDict([
            ('index', index),
            ('min_hist_diff', min_hist_diff),
            ('max_hist_diff', max_hist_diff),
            ('avg_hist_diff', avg_hist_diff),
        ])
        print(jsonify(res))
