#!/usr/bin/env python3
#coding: utf-8

# Rudolf Rosa

import sys
from collections import defaultdict
import csv
import random

# how many times to sample a feature vector for each language
N=100

# label for the accuracy field
ACC='ACCURACY'

import logging
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG)
    #level=logging.WARN)

# feat -> count
gold_data = list()
feat_values = defaultdict(set)
with open('../data/train_y.csv') as train:
    d2 = csv.DictReader(train)
    for dictline in d2:
        gold_data.append(dictline)
        for feat in dictline:
            value = dictline[feat]
            if value:
                feat_values[feat].add(value)

feats_all = list(dictline.keys())

# because random.choice cannot work with sets :-/
feat_values_tuples = dict()
for feat in feat_values:
    feat_values_tuples[feat] = tuple(feat_values[feat])

#for feat in allfeats:
#    print(feat, ':', ' '.join(sorted(list(feat_values[feat]))))

# ignore features that should stay fixed
feats_fixed = {'','wals_code','name','latitude','longitude','genus','family','countrycodes'}
feats_use = set(feats_all)
feats_use.difference_update(feats_fixed)

# add a feature for the accuracy of the generated lines
feats_extended = feats_all.copy()
feats_extended.append(ACC)

# construct output
output = list()

# put there the gold data
for dictline in gold_data:
    dictline[ACC] = 1.0
    output.append(dictline)

# repeatedly sample and score random data
for _ in range(N):
    for dictline in gold_data:
        sampledline = dictline.copy()
        incorrect_count = 0
        feats_filled = [feat if feat for feat in feats_use]
        total_count = len(feats_filled)
        # How many feature values to change
        change_count = random.randoint(0, total_count)
        # Which features to change
        feats_change = random.sample(feats_filled, change_count)
        for feat in feats_change:
            # Change the feature
            true_value = dictline[feat]
            sampled_value = random.choice(feat_values_tuples[feat])
            if sampled_value != true_value:
                incorrect_count += 1
                sampledline[feat] = sampled_value
        # NOTE: An incorrect line for one language may be a correct line for a
        # different language (even one from the same family or even genus).
        # I ignore that at the moment, hoping that the same line will simply
        # appear multiple times in the training data with different scores.
        # Some accuracies are thus underestimates (and symmetrically some high
        # accuracy lines would be low accuracy lines for another language, so
        # these are overestimates).
        # Alternatively, we might go through whole train to find the highest
        # accuracy for the changed line and use that as the gold accuracy?
        sampledline[ACC] = 1 - incorrect_count/total_count
        output.append(sampledline)
        
# produce output

random.shuffle(output)

with open('../data/train_regression.csv', 'w') as out:
    outwriter = csv.DictWriter(out, feats_extended)
    outwriter.writeheader()
    for dictline in output:
        outwriter.writerow(dictline)

