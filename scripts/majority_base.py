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

# feat -> count
counts = defaultdict(dict)
with open('../data/train_y.csv') as train:
    d2 = csv.DictReader(train)
    for dictline in d2:
        for feat in dictline:
            value = dictline[feat]
            if value:
                logging.debug('Feature {} has value {}'.format(feat, value))
                counts[feat][value] = counts[feat].get(value, 0) + 1

allfeats = list(dictline.keys())

# precompute maxima
# feat -> maxvalue
maxes = defaultdict(dict)
logging.info('Most frequent values across all languages, with counts')
for feat in allfeats:
    maxes[feat] = max(counts[feat], key=counts[feat].get)

# predict
with open('../data/dev_x.csv') as dev, open('output', 'w') as output:
    d2 = csv.DictReader(dev)
    outwriter = csv.DictWriter(output, allfeats)
    outwriter.writeheader()
    for dictline in d2:
        for feat in dictline:
            if dictline[feat] == '?':
                # predict
                dictline[feat] = maxes[feat]
                logging.debug('Predicting {} for {}'.format(dictline[feat], feat))
        outwriter.writerow(dictline)

