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
        correct = 0
        total = 0
        feats_filled = [feat if feat for feat in feats_use]
        change_count = random.randoint(0, len(feats_filled))
        feats_change = random.sample(feats_filled, change_count)
        for feat in feats_use:
            true_value = dictline[feat]
            if true_value:
                # TODO need to often change only a few values -- first sample
                # the number of values to change, then change only some of
                # them? Or for each, first throw a dice to decide whether to
                # change it?
                # TODO !!!!!! ??????
                # Probably I want uniform ditribution of the proportion of
                # correct values.
                # It is probably not hard to distinguish really bad lines, as
                # the baselines are around 50%, so we need to focus on the
                # quite good and really good lines and learn to distinguish
                # those well.
                # (But we also need to see some bad lines to know what they
                # look like.)
                # We could generate ALL options but that would lead to too
                # large training data (and too slow training), and also such
                # distro would have way too many bad lines (we probably don!t
                # need so many).
                if random.choice((True,False)):
                    sampled_value = random.choice(feat_values_tuples[feat])
                    sampledline[feat] = sampled_value
                    total += 1
                    if sampled_value == true_value:
                        correct += 1
        sampledline[ACC] = correct/total
        output.append(sampledline)
        
# produce output

random.shuffle(output)

with open('../data/train_regression.csv', 'w') as out:
    outwriter = csv.DictWriter(out, feats_extended)
    outwriter.writeheader()
    for dictline in output:
        outwriter.writerow(dictline)

