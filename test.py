from dataset import load_wiki_traffic_dataset


def test_load_dataset():
    file_path = '/dataset/web-traffic-forecast/train_1.csv'
    train = load_wiki_traffic_dataset(file_path)

    assert train[train.isnull().any(axis=1)].shape[0] == 0
