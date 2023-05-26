from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
import os
from sklearn.model_selection import train_test_split
import json

current_folder = os.path.dirname(os.path.abspath(__file__))


class Circles(object):
    def __init__(self):
        self.X, self.labels = make_circles(n_samples=400, noise=0.1, random_state=5622, factor=0.8)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.labels,
                                                                                test_size=0.3, random_state=5622)


class DataBlobs:
    def __init__(self, centers, std=1.75):
        self.X, self.labels = make_blobs(n_samples=400, n_features=2, cluster_std=std, centers=centers,
                                         shuffle=False, random_state=5622)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.labels,
                                                                                test_size=0.3, random_state=5622)


class DataMoons(object):
    def __init__(self):
        self.X, self.labels = make_moons(n_samples=400, noise=0.05, shuffle=False, random_state=5622)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.labels,
                                                                                test_size=0.3, random_state=5622)


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
