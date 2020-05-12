#!/usr/bin/env python3
#coding: utf-8

# Rudolf Rosa

# Mesure accuracies of gold correct data

import sys
import csv

from sklearn.preprocessing import OneHotEncoder

import logging
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)

feats_remove = {'','wals_code','name','latitude','longitude','countrycodes'}
feats_fixed = {'family', 'genus'}

# load test data 
test_data = list()
with open('../data/dev_y.csv') as test:
    d2 = csv.DictReader(test)
    for dictline in d2:
        test_data.append(dictline)
    feats_all = d2.fieldnames


# load models

#M='mlpr'
M='mlpr_full'

import pickle
with open('../models/'+M+'.onehot_encoder', 'rb') as f:
    one_hotter = pickle.load(f)
with open('../models/'+M+'.model', 'rb') as f:
    regressor = pickle.load(f)

feat_all_values = one_hotter.categories_


# convert to onehotter input format and convert ot onehot
def onehot(option, one_hotter):
    line = list()
    for feat in option:
        if feat not in feats_remove:
            if option[feat] == 'nan':
                line.append('')
            else:
                line.append(option[feat])
    return one_hotter.transform([line])

# apply
preds_sum = 0
for line in test_data:
    line_onehot = onehot(line, one_hotter)
    prediction = regressor.predict(line_onehot)[0]
    logging.info('Predicted accuracy for {} is {}'.format(
                line['name'], prediction))
    preds_sum += prediction

logging.info('Average predicted accuracy {}'.format(
                (preds_sum/len(test_data))))
