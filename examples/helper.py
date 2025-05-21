import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def load_iris(labels=[]):
    # TODO: Check if data is there/add error handling
    data = pd.read_csv('../../data/iris/iris.data', header=None)

    data = data.values

    X = np.array(data[:, :-1], dtype=np.float32)
    y = np.array(data[:, -1])

    if len(labels) != 0:
        mask = np.isin(y, labels)
        y = y[mask]
        X = X[mask]


    y = y.reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(y)
    y = encoder.transform(y)

    return X, y


def load_glioma(exclude_non_binary_features = False):
    # TODO: Check if data is there/add error handling
    data = pd.read_csv('../../data/glioma/TCGA_InfoWithGrade.csv')
    data = data.values

    if exclude_non_binary_features:
        X1 = np.array(data[:, 1:2], dtype=np.float32)
        X2 = np.array(data[:, 4:], dtype=np.float32)
        X = np.hstack((X1, X2))
    else:
        X = np.array(data[:, 1:], dtype=np.float32)

    y = np.array(data[:, 0])


    y = y.reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(y)
    y = encoder.transform(y)

    return X, y


def load_wdbc(labels=[]):
    # TODO: Check if data is there/add error handling
    data = pd.read_csv('../../data/wdbc/wdbc.data', header=None)

    data = data.values

    X = np.array(data[:, 2:], dtype=np.float32)
    y = np.array(data[:, 1])

    if len(labels) != 0:
        mask = np.isin(y, labels)
        y = y[mask]
        X = X[mask]


    y = y.reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(y)
    y = encoder.transform(y)

    return X, y


def load_mushroom():
    # TODO: Check if data is there/add error handling
    data = pd.read_csv('../../data/mushroom/agaricus-lepiota.data', header=None)
    data = data.values

    X = np.array(data[:, 1:])
    y = np.array(data[:, 0])

    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(X)
    X = encoder.transform(X)

    y = y.reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(y)
    y = encoder.transform(y)

    return X, y


def split_data(X, y, seed, test_size=0.2):

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=test_size, random_state=seed)

    return {"X": X_train_full, "y": y_train_full}, {"X": X_train, "y": y_train}, {"X": X_val, "y": y_val}, {"X": X_test, "y": y_test}

def load_dataset(dataset):
    """
    Load the specified dataset.

    eters:
    - dataset (str): The name of the dataset ("iris", "mnist", "glioma", "covertype").

    Returns:
    - X: Features of the dataset.
    - y: One-hot encoded labels of the dataset.
    """
    loaders = {
        "iris": load_iris,
        "glioma": load_glioma,
        "wdbc": load_wdbc,
        "mushroom": load_mushroom,
    }

    if dataset not in loaders:
        raise ValueError(f"Unsupported dataset: {dataset}. Supported datasets are: {list(loaders.keys())}")

    return loaders[dataset]()



def normalise_input(X, mean, std):
    return (X-mean)/std