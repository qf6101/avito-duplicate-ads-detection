import sys
import matplotlib.pyplot as plt
import cv2

__all__ = ['image_location', 'batch_image_location', 'show_images', 'compare_images']

images_dir = "/srv/data/0/ticktock/competition/kaggle/avito-duplicate-ads-detection/data/images"


def image_location(image_id):
    first_index = str(int(image_id % 100 / 10))
    second_index = str(image_id % 100)
    return ''.join([images_dir, "/Images_", first_index, "/", second_index, "/", str(image_id), ".jpg"])


def batch_image_location(image_ids):
    img_locs = []
    for img_id in image_ids:
        img_locs.append(image_location(img_id))
    return img_locs


def show_images(img_locs):
    for loc in img_locs:
        image = plt.imread(loc)
        plt.imshow(image)
        plt.axis('off')
        plt.show()


def compare_images_from_left(left_img_locs, right_img_locs, comp_func):
    batch_min_hist_diff = sys.maxsize
    batch_max_hist_diff = -sys.maxsize
    sum_hist_diff = 0
    for l_img_loc in left_img_locs:
        max_hist_diff = -sys.maxsize
        l_img = cv2.imread(l_img_loc)
        l_img_hist = cv2.calcHist(l_img, [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        l_img_hist = cv2.normalize(l_img_hist, l_img_hist).flatten()
        for r_img_loc in right_img_locs:
            r_img = cv2.imread(r_img_loc)
            r_img_hist = cv2.calcHist(r_img, [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            r_img_hist = cv2.normalize(r_img_hist, r_img_hist).flatten()
            hist_diff = comp_func(l_img_hist, r_img_hist)
            if max_hist_diff < hist_diff:
                max_hist_diff = hist_diff
        if batch_min_hist_diff > max_hist_diff:
            batch_min_hist_diff = max_hist_diff
        if batch_max_hist_diff < max_hist_diff:
            batch_max_hist_diff = max_hist_diff
        sum_hist_diff += max_hist_diff
    return batch_min_hist_diff, batch_max_hist_diff, sum_hist_diff / len(left_img_locs)


def compare_images_with(left_img_locs, right_img_locs, comp_func):
    l_min, l_max, l_avg = compare_images_from_left(left_img_locs, right_img_locs, comp_func)
    r_min, r_max, r_avg = compare_images_from_left(right_img_locs, left_img_locs, comp_func)
    return (l_min + r_min) / 2, (l_max + r_max) / 2, (l_avg + r_avg) / 2


def compare_images(left_img_locs, right_img_locs):
    return compare_images_with(left_img_locs, right_img_locs,
                               lambda left, right: cv2.compareHist(left, right, cv2.HISTCMP_CORREL))
