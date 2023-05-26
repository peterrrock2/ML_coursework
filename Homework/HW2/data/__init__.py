#import os
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class BikeSharing(object):
    def __init__(self):
        rawdata = pd.read_csv('data/uci_bike_day.csv').drop(['registered', 'casual', 'dteday'], axis=1).to_numpy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(rawdata[:,:-1], rawdata[:,-1], test_size=0.2, random_state=5622)