import os
import pickle

dir_here = os.path.dirname(os.path.realpath(__file__))
data_file_dir = os.path.join(dir_here, 'data_files')
cache_dir = os.path.join(dir_here, 'cache')


def with_cache(cache_file, g):
    if os.path.exists(cache_file):
        return pickle.load(open(cache_file, 'rb'))
    else:
        res = g()
        pickle.dump(res, open(cache_file, 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)
        return res


def load_file_with_cache(name, reader, f):
    return with_cache(os.path.join(cache_dir, name + '.pickle'),
                      lambda: reader(os.path.join(data_file_dir, f)))


from .original import *