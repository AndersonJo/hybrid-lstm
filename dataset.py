import os
import re

import pandas as pd


def load_wiki_traffic_dataset(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError('file not found')
    train = pd.read_csv('/dataset/web-traffic-forecast/train_1.csv').fillna(0.)
    train.columns = list(train.columns[:1]) + list(range(1, len(train.columns[1:]) + 1))

    r = re.compile('(?P<page>.+)_(?P<country>[a-z]+)\.'
                   '(?P<project>[a-z]+)\.org_'
                   '(?P<access>\w*-?\w*)_(?P<agent>\w+)')

    train = pd.concat((train['Page'].str.extract(r, expand=False), train.ix[:, train.columns != 'Page']), axis=1)
    return train
