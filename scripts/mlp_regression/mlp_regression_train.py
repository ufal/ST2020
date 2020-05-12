#!/usr/bin/env python3
#coding: utf-8

# Rudolf Rosa

import sys
import csv

from sklearn.preprocessing import OneHotEncoder

import logging
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)

# TODO encode GPS latitude longitude (bucketed? float?)
# TODO encode countrycodes (set)
# TODO handle rare values for family (if count < N, replace with RARE)
# TODO handle rare values for genus (if count < N, replace with FAMILY_family)
feats_remove = {'','wals_code','name','latitude','longitude','countrycodes', 'ACCURACY'}

# load training data 
train_data = list()
train_labels = list()
with open('../../data/train_regression.csv') as train:
#with open('../../data/train_regression_10000.csv') as train:
    d2 = csv.DictReader(train)
    for dictline in d2:
        train_line = list()
        for feat in dictline:
            if feat not in feats_remove:
                train_line.append(dictline[feat])        
        train_data.append(train_line)
        train_labels.append(float(dictline['ACCURACY']))

# sklearn.preprocessing.OneHotEncoder
one_hotter = OneHotEncoder(handle_unknown='ignore')
one_hotter.fit(train_data)
logging.debug(one_hotter.categories_)

train_data_onehot = one_hotter.transform(train_data)
logging.debug(train_data_onehot)

# sklearn.neural_network.MLPRegressor

from sklearn.neural_network import MLPRegressor
regressor = MLPRegressor(verbose=True, early_stopping=True)
regressor.fit(train_data_onehot, train_labels)

# save trained model
import pickle
with open('../models/mlpr.onehot_encoder', 'wb') as f:
    pickle.dump(one_hotter, f)
with open('../models/mlpr.model', 'wb') as f:
    pickle.dump(regressor, f)



# apply to dev data (map 'nan' to '', map unknown to '')


