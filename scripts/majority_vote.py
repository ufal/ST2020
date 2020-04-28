#!/usr/bin/env python3
#coding: utf-8

# Rudolf Rosa

import sys
from collections import defaultdict
import csv


import logging
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG)
    #level=logging.WARN)

CODE = 1
FAM = 6

# lang family -> lang code -> feature -> value
data = defaultdict(dict)

with open('../data/train_y.csv') as train:
    d2 = csv.DictReader(train)
    for dictline in d2:
        code = dictline['wals_code']
        fam = dictline['family']
        data[fam][code] = dictline

allfeats = list(dictline.keys())
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
            if value:
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

# predict

with open('../data/dev_x.csv') as dev, open('output', 'w') as output:
    d2 = csv.DictReader(dev)
    outwriter = csv.DictWriter(output, allfeats)
    outwriter.writeheader()
    for dictline in d2:
        family = dictline['family']
        for feat in dictline:
            if dictline[feat] == '?':
                # predict
                if family in maxes:
                    maxvalue = maxes[family][feat]
                    logging.debug('Predicting {} for {} based on {}'.format(maxvalue, feat, family))
                else:
                    maxvalue = maxes[0][feat]
                    logging.debug('Predicting {} for {}'.format(maxvalue, feat))
                dictline[feat] = maxvalue
        outwriter.writerow(dictline)

