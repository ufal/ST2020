#!/usr/bin/env python3
#coding: utf-8

import sys
from collections import defaultdict

import logging
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.WARN)

CODE = 0
NAME = 1
LAT = 2
LON = 3
GENUS = 4
FAM = 5
COUNTRY = 6
FEATS = 7

# lang family -> lang code -> feature -> value
data = defaultdict(dict)

allfeats = set()
allfamilies = set()

with open('data/train.csv') as train:
    train.readline()  # skip header
    for line in train:
        fields = line.rstrip().split('\t')
        data[fields[FAM]][fields[CODE]] = dict()
        langdict = data[fields[FAM]][fields[CODE]]
        feats = fields[FEATS].split('|')
        for feat in feats:
            #logging.info(feat)
            name, value = feat.split('=', 1)
            langdict[name] = value
            allfeats.add(name)

allfamilies = set(data.keys())

# family -> feature -> value -> count
counts = defaultdict(dict)
for feat in allfeats:
    counts[0][feat] = defaultdict(int)
for family in allfamilies:
    for feat in allfeats:
        counts[family][feat] = defaultdict(int)
    for lang in data[family]:
        for feat in data[family][lang]:
            value = data[family][lang][feat]
            counts[family][feat][value] += 1
            counts[0][feat][value] += 1

# precompute maxima
# family -> feat -> maxvalue
maxes = defaultdict(dict)
# all languages
logging.info('Most frequent values across all languages, with counts')
for feat in allfeats:
    maxvalue = max(counts[0][feat], key=counts[0][feat].get)
    maxes[0][feat] = maxvalue
    maxcount = counts[0][feat][maxvalue]
    logging.info(feat + ': ' + maxvalue + ' ' + str(maxcount))
# per family
for family in allfamilies:
    for feat in allfeats:
        #logging.debug(family + " " + feat)
        if counts[family][feat]:
            maxvalue = max(counts[family][feat], key=counts[family][feat].get)
            assert counts[family][feat][maxvalue] != 0
        else:
            maxvalue = maxes[0][feat]
        maxes[family][feat] = maxvalue

# predict and measure accuracies
correct = 0
total = 0

with open('data/dev.csv') as dev:
    dev.readline()  # skip header
    for line in dev:
        fields = line.rstrip().split('\t')
        feats = fields[FEATS].split('|')
        for feat in feats:
            name, value = feat.split('=', 1)
            #logging.debug(fields[NAME] + " " + name)
            family = fields[FAM]
            if family in maxes:
                maxvalue = maxes[family][name]
            else:
                maxvalue = maxes[0][name]
            total += 1
            if maxvalue == value:
                correct += 1

print('Total', 'Correct', 'Accuracy', sep='\t')
print(total, correct, (correct/total*100), sep='\t')

