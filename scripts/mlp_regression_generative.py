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

# TODO encode GPS
# TODO encode countries
feats_remove = {'','wals_code','name','latitude','longitude','countrycodes'}
#feats_fixed = {'','wals_code','name','latitude','longitude','genus','family','countrycodes'}

# load training data 
train_data = list()
#with open('../data/train_regression.csv') as train:
with open('../data/train_regression_100.csv') as train:
    d2 = csv.DictReader(train)
    for dictline in d2:
        train_data.append(list(dictline.values()))


# sklearn.preprocessing.OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(train_data)
logging.info(enc.categories_)

train_data_onehot = enc.transform(train_data)
logging.info(train_data_onehot)

# sklearn.neural_network.MLPRegressor

# save trained model

# apply to dev data (map 'nan' to '', map unknown to '')


