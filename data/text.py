from feature.text import preprocess_sentence, preprocess
from . import item_info, generate_with_cache
import pandas as pd

def gen_preprocessed_title(n_jobs=16, item_info=item_info):
    from multiprocessing import Pool
    pool = Pool(n_jobs)
    raw = item_info['title']
    raw = raw[raw.notnull()]
    title, title_stemmed = zip(*pool.map(preprocess_sentence, raw))
    df = pd.DataFrame({
        'title': title,
        'title_stemmed': title_stemmed,
    }, index=raw.index)

    return df

def gen_preprocessed_description(n_jobs=16, item_info=item_info):
    from multiprocessing import Pool
    pool = Pool(n_jobs)
    description, description_stemmed = zip(*pool.map(preprocess, item_info['description']))
    df = pd.DataFrame({
        'description': description,
        'description_stemmed': description_stemmed,
    }, index=item_info.index)

    return df


preprocessed_title = generate_with_cache('preprocessed_title', gen_preprocessed_title)
preprocessed_description = generate_with_cache('preprocessed_description', gen_preprocessed_description)
