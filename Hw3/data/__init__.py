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
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(rawdata[:,:-1], rawdata[:,-1], test_size=0.2, random_state=5622)

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
