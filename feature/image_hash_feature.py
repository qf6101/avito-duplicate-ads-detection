import logging

import imagehash
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import pairwise_distances

from util import binary_matrix_to_int, jaccard
from .util import image_path
logger = logging.getLogger('image_hash_feature')


def whash_(image):
    ''' bypass assert for small image '''
    try:
        return imagehash.whash(image)
    except AssertionError:
        return imagehash.ImageHash(np.zeros((8,8), dtype=bool))

hash_names = ['ahash', 'phash', 'dhash', 'whash']
hash_functions = [imagehash.average_hash, imagehash.phash, imagehash.dhash, whash_]


def dhash(image_id):
    try:
        image = Image.open(image_path(image_id))
    except OSError as e:
        logger.warn('cannot find image {}'.format(image_id))
        return None
    else:
        return np.stack(f(image).hash.ravel() for f in hash_functions)


def build_headers():
    headers = ['index']
    for hash_name in hash_names:
        headers.append('image_{}_jaccard'.format(hash_name))
        headers.extend(['image_{}_hamming_{}_{}'.format(hash_name, x, y) for x in [0, 1] for y in ['min', 'max', 'mean']])
    return headers
headers = build_headers()


def pair_features(hashes1, hashes2):
    feats = [jaccard(binary_matrix_to_int(hashes1), binary_matrix_to_int(hashes2))]

    D = pairwise_distances(hashes1, hashes2, metric='hamming')
    if D.shape[0] > D.shape[1]:
        D = D.T
    if D.shape[0] == 0 or D.shape[1] == 0:
        feats.extend([np.nan] * 6)
    else:
        s0 = D.min(axis=1)
        s1 = D.max(axis=0)
        feats.extend([s0.min(), s0.max(), s0.mean(), s1.min(), s1.max(), s1.mean()])
    return feats


def _collect_hashes(I):
    hashes = list(filter(lambda x: x is not None, (dhash(i) for i in I)))
    if len(hashes) == 0:
        return None
    else:
        return np.stack(hashes)


def gen_features(I1, I2):
    n_features = 7 * len(hash_functions)
    if len(I1) == 0 or len(I2) == 0:
        return [np.nan] * n_features
    else:
        H1 = _collect_hashes(I1)
        H2 = _collect_hashes(I2)
        if H1 is not None and H2 is not None:
            features = []
            for x, y in zip(np.transpose(H1, (1, 0, 2)), np.transpose(H2, (1, 0, 2))):
                features.extend(pair_features(x, y))
            return features
        else:
            return [np.nan] * n_features


def parse_image_array(x):
    if len(x) == 0:
        return []
    else:
        return list(map(int, x.split(', ')))


if __name__ == '__main__':
    import sys
    import json

    for line in sys.stdin:
        try:
            line = json.loads(line.rstrip())
            index = line['index']
            image_ids_1 = parse_image_array(line['images_array_1'])
            image_ids_2 = parse_image_array(line['images_array_2'])
            print(index + ',' + ','.join(map(str, gen_features(image_ids_1, image_ids_2))))
        except Exception as e:
            logger.exception(e)
