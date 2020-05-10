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

feats_remove = {'','wals_code','name','latitude','longitude','countrycodes'}
feats_fixed = {'family', 'genus'}

# load test data 
test_data = list()
with open('../data/dev_x.csv') as test:
    d2 = csv.DictReader(test)
    for dictline in d2:
        test_data.append(dictline)
    feats_all = d2.fieldnames


# load models
import pickle
with open('../models/mlpr.onehot_encoder', 'rb') as f:
    one_hotter = pickle.load(f)
with open('../models/mlpr.model', 'rb') as f:
    regressor = pickle.load(f)

feat_all_values = one_hotter.categories_

# for each ? values, expand to all possible values for the feat
# include all info
def generate_options(line):
    return generate_options_recursively(line, 0)

def generate_options_recursively(line, position):
    # First generate the continuation (head recursion)
    if position < len(line)-1:
        further_options = generate_options_recursively(line, position+1)
    else:
        further_options = [[]]
    
    output = list()
    feat = feats_all[position]
    if line[feat] == '?':
        all_values = feat_all_values[position-len(feats_remove)]  # the removed feats are not encoded
        for value in all_values:
            for further in further_options:
                option = [value, *further]
                output.append(option)
    else:
        for further in further_options:
            option = [line[feat], *further]
            output.append(option)
    return output

# convert to onehotter input format and convert ot onehot
def onehot(options, one_hotter):
    options_transformed = list()
    for option in options:
        line = list()
        for feat in option:
            if feat not in feats_remove:
                if option[feat] == 'nan':
                    line.append('')
                else:
                    line.append(option[feat])
        options_transformed.append(line)
    return one_hotter.transform(options_transformed)

def find_max(options, predictions):
    best_option = options[0]
    best_prediction = predictions[0]
    for option, prediction in zip(options, predictions):
        if prediction > best_prediction:
            best_prediction = prediction
            best_option = option
    return best_option, best_prediction


# apply
output = list()
for line in test_data:
    logging.info('Predicting values for {}'.format(line['name']))
    all_options = generate_options(line)
    logging.info('Generated {} options'.format(len(all_options)))
    all_options_onehot = onehot(all_options, one_hotter)
    predictions = regressor.predict(all_options_onehot)
    best_option, best_prediction = find_max(all_options, predictions)
    logging.info('Selected option with predicted accuracy {}'.format(
                best_prediction))
    output.append(best_option)

with open('output', 'w') as out:
    outwriter = csv.DictWriter(out, feats_all)
    outwriter.writeheader()
    for line in output:
        outwriter.writerow(line)

