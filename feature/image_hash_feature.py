from PIL import Image
import imagehash
from util import binary_matrix_to_int, jaccard
from config import config
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
images_dir = config['image_root']

import logging
logger = logging.getLogger('image_hash_feature')

def image_path(image_id):
    """
    Get image location from image ID
    """
    first_index = str(int(int(image_id) % 100 / 10))
    second_index = str(int(image_id) % 100)
    return ''.join([images_dir, "/Images_", first_index, "/", second_index, "/", str(image_id).strip(), ".jpg"])

def dhash(image_id):
    try:
        image = Image.open(image_path(image_id))
    except OSError as e:
        logger.warn('cannot find image {}'.format(image_id))
        return None
    else:
        return imagehash.phash(image).hash.ravel()


headers = ['index', 'image_dhash_jaccard'] + ['image_dhash_hamming_{}_{}'.format(x,y) for x in [0,1] for y in ['min', 'max', 'mean']]
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
        return np.vstack(hashes)

def gen_features(I1, I2):
    if len(I1) == 0 or len(I2) == 0:
        return [np.nan] * 7
    else:
        H1 = _collect_hashes(I1)
        H2 = _collect_hashes(I2)
        if H1 is not None and H2 is not None:
            return pair_features(H1, H2)
        else:
            return [np.nan] * 7

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
            print(index+','+','.join(map(str, gen_features(image_ids_1, image_ids_2))))
        except Exception as e:
            logger.exception(e)