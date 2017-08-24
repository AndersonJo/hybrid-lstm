from dataset import load_wiki_traffic_dataset


def test_load_dataset():
    train_path = '/dataset/web-traffic-forecast/train_1.csv'
    test_path = '/dataset/web-traffic-forecast/key_1.csv'
    train = load_wiki_traffic_dataset(train_path, test_path)

    assert train[train.isnull().any(axis=1)].shape[0] == 0
