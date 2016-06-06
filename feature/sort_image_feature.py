import pandas as pd
import json


def sort_image_feature(feature_file):
    """Sort image feature file (json format) by index and save to csv file"""
    X = []
    with open(feature_file) as file:
        for line in file:
            line = json.loads(line)
            X.append(line)
    image_feature = pd.DataFrame(X)
    image_feature['index'].astype(int)
    image_feature = image_feature.set_index(keys=['index'])
    image_feature = image_feature.sort_index()
    image_feature.to_csv(feature_file.replace(".jsonl", "") + ".csv")


if __name__ == '__main__':
    """Sort image feature file and save to csv file"""
    import argparse
    # The image feature file name is in arguments
    arg_parser = argparse.ArgumentParser(description='sort image feature')
    arg_parser.add_argument('-f', help='json format image feature file', action='store', dest='jsonfile')
    args = arg_parser.parse_args()
    sort_image_feature(args.jsonfile)
