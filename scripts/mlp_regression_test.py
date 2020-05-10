#!/usr/bin/env python3
#coding: utf-8

# Rudolf Rosa

# artificial setup: predict accuracies

import sys
import csv

from sklearn.preprocessing import OneHotEncoder

import logging
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)

feats_remove = {'','wals_code','name','latitude','longitude','countrycodes', 'ACCURACY'}

# load test data 
languages = list()
test_data = list()
test_labels = list()
with open('../data/train_regression_tail.csv') as test:
    d2 = csv.DictReader(test)
    for dictline in d2:
        test_line = list()
        for feat in dictline:
            if feat not in feats_remove:
                test_line.append(dictline[feat])        
        test_data.append(test_line)
        test_labels.append(float(dictline['ACCURACY']))
        languages.append(dictline['name'])

# load models
import pickle
with open('../models/mlpr.onehot_encoder', 'rb') as f:
    one_hotter = pickle.load(f)
with open('../models/mlpr.model', 'rb') as f:
    regressor = pickle.load(f)

# apply
test_data_onehot = one_hotter.transform(test_data)
predictions = regressor.predict(test_data_onehot)

for true, pred, lang in zip(test_labels, predictions, languages):
    print(abs(true-pred), true, pred, lang)


# apply to dev data (map 'nan' to '', map unknown to '')


