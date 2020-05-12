#!/usr/bin/env python3
#coding: utf-8

# Rudolf Rosa

import sys
import csv

from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPRegressor

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

gold_data = list()
with open('../data/dev_y.csv') as test:
    d2 = csv.DictReader(test)
    for dictline in d2:
        gold_data.append(dictline)

# load models

#M='mlpr'
M='mlpr_full'

import pickle
with open('../models/'+M+'.onehot_encoder', 'rb') as f:
    one_hotter = pickle.load(f)
with open('../models/'+M+'.model', 'rb') as f:
    regressor = pickle.load(f)

feat_all_values = one_hotter.categories_

# How many random options to generate
N=10000

# TODO this VASTLY subexplores the space; it is reasonable to first take random
# options, and then to take a hillclimbing approach to improving the best
# option

# TODO measure search error versus prediction error to find best params

def generate_options(line):
    options = list()
    for _ in range(N):
        options.append(generate_random_option(line))
    return options

import random

# randomly fill each '?'
def generate_random_option(line):
    new_line = line.copy()
    for position in range(len(line)):
        feat = feats_all[position]
        if line[feat] == '?':
            all_values = feat_all_values[position-len(feats_remove)]  # the removed feats are not encoded
            line[feat] = random.choice(all_values)
    return new_line

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
preds_sum = 0
for line, goldline in zip(test_data, gold_data):
    logging.info('Predicting values for {}'.format(line['name']))
    all_options = generate_options(line)
    all_options.append(goldline)
    #logging.info('Generated {} options'.format(len(all_options)))
    all_options_onehot = onehot(all_options, one_hotter)
    predictions = regressor.predict(all_options_onehot)
    best_option, best_prediction = find_max(all_options, predictions)
    logging.info('Selected option with predicted accuracy {}'.format(
                best_prediction))
    preds_sum += best_prediction
    output.append(best_option)

with open('output', 'w') as out:
    outwriter = csv.DictWriter(out, feats_all)
    outwriter.writeheader()
    for line in output:
        outwriter.writerow(line)

logging.info('Average predicted accuracy {}'.format(
                (preds_sum/len(test_data))))
