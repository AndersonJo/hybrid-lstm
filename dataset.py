import os
import re

import pandas as pd


def load_wiki_traffic_dataset(train_path: str, test_path: str):
    if not os.path.exists(train_path):
        raise FileNotFoundError('Train dataset not found')
    if not os.path.exists(test_path):
        raise FileNotFoundError('Test dataset not found')

    train_dataset = pd.read_csv(train_path).fillna(0.)
    test_dataset = pd.read_csv(test_path)

    # Preprocessing Train Dataset
    # train_dataset.columns = list(train_dataset.columns[:1]) + list(range(1, len(train_dataset.columns[1:]) + 1))

    train_regex = re.compile('(?P<page>.+)_(?P<country>[a-z]+)\.'
                             '(?P<project>[a-z]+)\.org_'
                             '(?P<access>\w*-?\w*)_(?P<agent>\w+)')
    train_dataset = pd.concat((train_dataset['Page'].str.extract(train_regex, expand=True),
                               train_dataset.ix[:, train_dataset.columns != 'Page']), axis=1)

    # Preprocessing Test Dataset
    test_regex = re.compile('(?P<name>.+)_(?P<date>\d{4}-\d{2}-\d{2})')
    test_dataset = pd.concat([test_dataset['Page'].str.extract(test_regex, expand=True), test_dataset['Id']], axis=1)
    return train_dataset, test_dataset
