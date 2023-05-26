from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


class Circles(object):
    def __init__(self):
        self.X, self.labels = make_circles(n_samples=300, noise=0.1, random_state=5622, factor=0.6)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.labels,
                                                                                test_size=0.3, random_state=5622)


class DataBlobs:
    def __init__(self, centers, std=1.75):
        self.X, self.labels = make_blobs(n_samples=300, n_features=2, cluster_std=std, centers=centers,
                                         shuffle=False, random_state=5622)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.labels,
                                                                                test_size=0.3, random_state=5622)


class DataMoons(object):
    def __init__(self):
        self.X, self.labels = make_moons(n_samples=300, noise=0.05, shuffle=False, random_state=5622)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.labels,
                                                                                test_size=0.3, random_state=5622)


import os
import pickle
import numpy as np
import pandas as pd
import json
import random
from sklearn.model_selection import train_test_split

current_folder = os.path.dirname(os.path.abspath(__file__))


class Concrete(object):
    def __init__(self):
        rawdata = pd.read_csv('data/Concrete_Data.csv').to_numpy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(rawdata[:, :-1], rawdata[:, -1],
                                                                                test_size=0.2, random_state=5622)


class Digits(object):

    def __init__(self):
        loaded = np.load(os.path.join(current_folder, "mnist.npz"))
        self.images = images = loaded["images"].reshape(-1, 28 * 28)
        self.labels = labels = loaded["labels"]
        train_size = 1000
        valid_size = 500
        test_size = 500
        self.X_train, self.y_train = images[:train_size], labels[:train_size]
        self.X_valid, self.y_valid = images[train_size: train_size + valid_size], labels[
                                                                                  train_size: train_size + valid_size]
        self.X_test, self.y_test = (images[train_size + valid_size:train_size + valid_size + test_size],
                                    labels[train_size + valid_size: train_size + valid_size + test_size])


class BinaryDigits:
    """
    Class to store MNIST data for images of 9 and 8 only
    """

    def __init__(self):
        loaded = np.load(os.path.join(current_folder, "mnist.npz"))
        images = loaded["images"].reshape(-1, 28 * 28)
        labels = loaded["labels"]
        labels = labels % 2
        train_size = 1000
        valid_size = 500
        test_size = 500

        self.X_train, self.y_train = images[:train_size], labels[:train_size]
        self.X_valid, self.y_valid = images[train_size: train_size + valid_size], labels[
                                                                                  train_size: train_size + valid_size]
        self.X_test, self.y_test = (images[train_size + valid_size:train_size + valid_size + test_size],
                                    labels[train_size + valid_size: train_size + valid_size + test_size])


class IMDB:
    """
    Class to store IMDB dataset
    """

    def __init__(self):
        with open(os.path.join(current_folder, "movie_review_data.json")) as f:
            self.data = data = json.load(f)
        X = [d['text'] for d in data['data']]
        y = [d['label'] for d in data['data']]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, shuffle=True,
                                                                                random_state=42)
