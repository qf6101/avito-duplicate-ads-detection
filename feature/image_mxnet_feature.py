#!/usr/bin/env python

import cv2
import numpy as np
import logging
import sys
import mxnet as mx
from config import config
from skimage.color import gray2rgb
from .util import image_path

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='mxnet_feature.log',
                    filemode='a')

logger = logging.getLogger("image_mxnet_feature")

# __all__ = [ 'mxnet_model_parent_dir', 'mxnet_model_dir_prefix', 'mxnet_mean_img_path', 'init_models', 'batch_image_mxnet_feature', 'compare_images_batch', 'cos_sim']



# root of image locations
images_dir = config['image_root']

# 模型文件放置的顶级目录
mxnet_model_parent_dir = config['mxnet_model_root']

# 模型文件名称作为键，模型文件的目录、模型前缀和epoch数量作为值
#mxnet_model_dir_prefix = {"bn" : ("inception-bn", "Inception_BN", 39)}

#mxnet_model_dir_prefix = {"bn" : ("inception-bn", "Inception_BN", 39), 
#                  "v3" : ("inception-v3", "Inception-7", 1), 
#                  "21k" : ("inception-21k", "Inception", 9)}


global LINE_BATCH_SIZE
global NUMPY_BATCH_SIZE
global GPU
USE_MP = False


def init_models(mxnet_model_parent_dir,mxnet_model_dir_prefix, mxnet_mean_img_path):
    models_dict = {}
    mean_img_dict = {}
    #遍历所有的模型
    for (name,(dir, prefix, num_epoch)) in mxnet_model_dir_prefix.items() :
        model = mx.model.FeedForward.load(mxnet_model_parent_dir + "/" + dir + "/" + prefix, num_epoch, ctx=GPU, numpy_batch_size=NUMPY_BATCH_SIZE)
        # get internals from model's symbol
        internals = model.symbol.get_internals()
        # get feature layer symbol out of internals
        fea_symbol = internals["global_pool_output"]
        # Make a new model by using an internal symbol. We can reuse all parameters from model we trained before
        # In this case, we must set ```allow_extra_params``` to True
        # Because we don't need params from FullyConnected symbol
        feature_extractor = mx.model.FeedForward(ctx=GPU, symbol=fea_symbol, numpy_batch_size=NUMPY_BATCH_SIZE,
                                         arg_params=model.arg_params, aux_params=model.aux_params,
                                         allow_extra_params=True)
        models_dict[name] = feature_extractor
        if name in mxnet_mean_img_path.keys() :
            mean_img_dict[name] = mx.nd.load(mxnet_model_parent_dir + "/" + dir + "/" + mxnet_mean_img_path.get(name))["mean_img"]
    return (models_dict, mean_img_dict)

#@profile
def preprocess_image(path, mean_img, method, show_img=False):
    """
    预处理图像
    path: 图像位置
    mean_img: 均值图像
    method: 使用哪种预处理方式
    """
    # load image
    img = cv2.imread(path)
    if img is None:
        return np.zeros((1, 3, 224, 224))
    img = img[:,:,[2,1,0]]
    # 判断图片是否是灰度图
    if (len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1) ):
        img = gray2rgb(img)
    # we crop image from center
    short_egde = min(img.shape[:2])
    yy = int((img.shape[0] - short_egde) / 2)
    xx = int((img.shape[1] - short_egde) / 2)
    crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
    
    #@profile
    def precess_helper(resize_l, resize_r) :
        # resize to 224, 224
        resized_img = cv2.resize(crop_img, (resize_l, resize_r))
        # convert to numpy.ndarray
        sample = resized_img
        # swap axes to make image from (224, 224, 4) to (3, 224, 224)
        sample = np.swapaxes(sample, 0, 2)
        sample = np.swapaxes(sample, 1, 2)
        return sample
    
    if method == "bn":
        sample = precess_helper(224, 224)
        # sub mean 
        normed_img = sample - mean_img.asnumpy()
        return normed_img.reshape([1, 3, 224, 224])
    elif method == "v3" :
        sample = precess_helper(299, 299)
        # sub mean
        normed_img = sample - 128.
        normed_img /= 128.
        return normed_img.reshape([1, 3, 299, 299])
    elif method == "21k" :
        sample = precess_helper(224, 224)
        # sub mean 
        normed_img = sample - 117.
        return normed_img.reshape([1, 3, 224, 224])
    else:
        raise Exception('Model Error', method)


from multiprocessing import Pool
from itertools import repeat

pool = Pool(8)

#@profile
def batch_image_mxnet_feature(img_ids, models, means):
    """
    批量获得图像特征
    """
    # {{}}
    result_dict = dict( (img_id, {}) for img_id in img_ids )
    paths = batch_image_location(img_ids)
    for model_name in models.keys():
        # 获得所有的图像预处理数据
        if USE_MP:
            img_sample = pool.starmap(preprocess_image, zip(paths, repeat(means.get(model_name)), repeat(model_name), repeat(False)))
        else:
            img_sample = []
            for path in paths:
                img_sample.append(preprocess_image(path, means.get(model_name), model_name, False))
        samples = np.vstack(img_sample)
        global_pooling_feature = models.get(model_name).predict(samples)
        result = []
        for i in range(len(paths)):
            img_id = img_ids[i]
            result_dict[img_id][model_name] =  global_pooling_feature[i,:,0,0]
    return result_dict

# 批量获得图片路径
def batch_image_location(image_ids):
    """
    Get images location from image IDs
    """
    img_locs = [image_path(img_id) for img_id in image_ids]
    return img_locs

#@profile
def batch_img_pair_mxnet_features(img_pairs, models, img_means):
    """
    批量获得特征
    img_pairs: 图片对列表  [([1,2,3],[4,5,6]), ([1,2,3],[4,5,6])], 一些元组的集合，元组第一个是左边，第二个是右边
    """
    img_ids = sum([ a + b for a,b in img_pairs], [])
    if len(img_ids) == 0:
        return None
    else:
        return batch_image_mxnet_feature(img_ids, models, img_means)
    

def compare_images_from_minority(img_features_l, img_features_r, comp_func):
    """
    对一对图像进行相似度计算
    输入是mxnet的特征
    """
    #保持长度短的在左边
    if len(img_features_l) > len(img_features_r):
        img_features_l, img_features_r = img_features_r, img_features_l
    
    # 计算图片平均的相似度
    img_feature_acc_l = np.zeros_like(img_features_l[0])
    img_feature_acc_r = np.zeros_like(img_features_r[0])
    for i in img_features_l : img_feature_acc_l += i
    for i in img_features_r : img_feature_acc_r += i
    img_feature_acc_l /= len(img_features_l)  
    img_feature_acc_r /= len(img_features_r) 
    batch_mean_sim = comp_func(img_feature_acc_l, img_feature_acc_r)
    
    # 最小的相似度     
    batch_min_sim = sys.maxsize
    # 最大的相似度
    batch_max_sim = -sys.maxsize
    # 相似度的和
    sum_sim = 0
    
    for img_f_l in img_features_l:
        # 图片相似度，选择右边和当前图片相似度最接近的作为这个值
        img_sim = -sys.maxsize
        for img_f_r in img_features_r:
            sim = comp_func(img_f_l, img_f_r)
            if sim is not np.nan and img_sim < sim:
                img_sim = sim
        if batch_min_sim > img_sim:
            batch_min_sim = img_sim
        if batch_max_sim < img_sim:
            batch_max_sim = img_sim
        sum_sim += img_sim
    return [batch_min_sim, batch_max_sim, sum_sim / len(img_features_l),batch_mean_sim ]

#@profile
def compare_images_batch(img_ids_pairs, models, means, comp_func):
    """
    获得最终的相似度
    """
    # 获得有哪些特征类型，如bn、v3、21k，进行排序，确保知道输出的情况
    feature_types = [ key for key in models.keys()]
    feature_types.sort()
    # 获得所有图片ID对应的特征
    img_features = batch_img_pair_mxnet_features(img_ids_pairs, models, means)
    # 整理为训练样本那样的特征对，如左边图片对应的特征列表和右边图片对应的特征列表 [([左边的特征列表],[右边的特征列表]),([],[])]
    result = []
    for img_ids_pair in img_ids_pairs:
        if len(img_ids_pair[0]) <=0 or len(img_ids_pair[1]) <=0:
            result.append(dict( (x, [np.nan] * 4) for x in feature_types ))
        else :    
            # 获得各自商品的图片特征, 结果如 [{"bn":feature, "21k":feature}]
            img_feature_l = [ img_features[img_id_l] for img_id_l in img_ids_pair[0] ]
            img_feature_r = [ img_features[img_id_r] for img_id_r in img_ids_pair[1] ]
            img_sim = {}
            for feature_type in feature_types:
                # 获得当前特征方式的所有特征
                img_feature_l_t = [ feature[feature_type] for feature in img_feature_l ]
                img_feature_r_t = [ feature[feature_type] for feature in img_feature_r ]
                sims = compare_images_from_minority(img_feature_l_t, img_feature_r_t, comp_func)
                img_sim[feature_type] = sims
            result.append(img_sim)
    return result

def cos_sim(v1, v2):
    vv1 = np.asarray(v1)
    vv2 = np.asarray(v2)
    return vv1.dot(vv2) / (np.linalg.norm(vv1, 2) * np.linalg.norm(vv2, 2))

def parse_int_list(x):
    return list(map(int, [x for x in x.split(', ') if len(x)>0]))

# CSV header
def build_header(model_names):
    header = ["line_num"]
    for name in model_names:
        header.extend('mxnet_{}_batch_{}_sim'.format(name, x) for x in  ['min', 'max', 'summean', 'mean'])
    return header

if __name__ == '__main__':
    """ Generate image feature in parallel
        Input is from stdin, output is to stdout
    """
    import json
    import argparse

    mxnet_mean_img_path = {"bn" : "mean_224.nd"}
    mxnet_model_dir_prefix_all = {"bn" : ("inception-bn", "Inception_BN", 39),
        "v3" : ("inception-v3", "Inception-7", 1),
        "21k" : ("inception-21k", "Inception", 9)}
    parser = argparse.ArgumentParser("generate pairwise mxnet similarity")
    parser.add_argument('--model', help='mxnet model name', choices=mxnet_model_dir_prefix_all.keys(), required=True)
    parser.add_argument('--gpu', help='gpu', type=int, required=True)
    parser.add_argument('--line-batch-size', default=1, type=int, help='how many pairs in a batch')
    parser.add_argument('--numpy-batch-size', default=5, type=int, help='numpy batch size in mxnet')

    args = parser.parse_args()
    GPU = mx.gpu(args.gpu)
    LINE_BATCH_SIZE = args.line_batch_size
    NUMPY_BATCH_SIZE = args.numpy_batch_size

    model_selection = args.model
    mxnet_model_dir_prefix = {model_selection : mxnet_model_dir_prefix_all[model_selection]}
    # 获得所有的模型
    (models, means) = init_models(mxnet_model_parent_dir, mxnet_model_dir_prefix, mxnet_mean_img_path)
    model_names = list(models.keys())
    # 通过排序，来进行对应
    model_names.sort()

    #jsonify = lambda x: json.dumps(x, ensure_ascii=False)
    # 每次处理多少行的数据
    batch_line_size = LINE_BATCH_SIZE
    #f = open('test.txt')
    line_num = 0
    while True:
        line_count = 0
        img_ids_pairs = []
        index_process = []
        line_num_start = line_num
        # 需要防止的错误类型：
        # 1. 读错误，利用行号来记录哪些行被读成功，没有读成功的不参加处理，程序运行结束后统一处理
        # 2. 预测错误，通过标记为“ERROR”来表示
        for line in sys.stdin:
            try:
                # 行号增加放在开头，无论是否错误都会增加
                line_num += 1
                line = json.loads(line.rstrip())
                index = line['index']
                img_ids_pairs.append((parse_int_list(line['images_array_1']), parse_int_list(line['images_array_2'])))
                line_count += 1
                # 读错误，这个就不会加入
                index_process.append(index)
                if line_count == batch_line_size:
                    break
            except Exception as e:
                logger.exception(e)
                logger.error(str(index) + ": IndexLineReadError")
        
        # 没有可读的行时，进行退出程序
        if line_num_start == line_num:
            break
        else:
            try:
                result = compare_images_batch(img_ids_pairs, models, means, cos_sim)
                for i in range(len(result)) :
                    sim_score = []
                    for name in model_names: sim_score += result[i][name]
                    print(str(index_process[i]) + "," + ','.join([ str(score) for score in sim_score ]))
            except Exception as e:
                logger.exception(e)
                logger.error(",".join([str(num) for num in index_process]) + ": IndexLineFeatureError")
