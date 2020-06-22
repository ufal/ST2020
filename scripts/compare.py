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

dev_x = csv.DictReader(open(sys.argv[1]))
dev_y = csv.DictReader(open(sys.argv[2]))
out_1 = csv.DictReader(open(sys.argv[3]))
out_2 = csv.DictReader(open(sys.argv[4]))

# dict[feat] = count
total = defaultdict(int)
correct_1 = defaultdict(int)
correct_2 = defaultdict(int)
correct_any = defaultdict(int)
better_1 = defaultdict(int)
better_2 = defaultdict(int)

def percent(share, total):
    if total > 0:
        return '{:.2f}%'.format(share*100/total)
    else:
        return '--'

for line_x, line_y, line_1, line_2 in zip(dev_x, dev_y, out_1, out_2):
    for feat in line_x:
        if line_x[feat] == '?':
            is_correct_1 = line_y[feat] == line_1[feat]
            is_correct_2 = line_y[feat] == line_2[feat]
            
            total[feat] += 1
            correct_1[feat] += is_correct_1
            correct_2[feat] += is_correct_2
            correct_any[feat] += (is_correct_1 or is_correct_2)
            better_1[feat] += (is_correct_1 and not is_correct_2)
            better_2[feat] += (is_correct_2 and not is_correct_1)

print('Feature', 'Acc 1', 'Acc 2', 'Acc *', '1 > 2', 'Support', sep='\t')
for feat in total:
    print(feat,
            percent(correct_1[feat], total[feat]),
            percent(correct_2[feat], total[feat]),
            percent(correct_any[feat], total[feat]),
            percent(better_1[feat], (better_1[feat] + better_2[feat])),
            better_1[feat] + better_2[feat],
            sep='\t')

print('TOTAL',
        percent(sum(correct_1.values()), sum(total.values())),
        percent(sum(correct_2.values()), sum(total.values())),
        percent(sum(correct_any.values()), sum(total.values())),
        percent(sum(better_1.values()), sum(better_1.values()) + sum(better_2.values())),
        sum(better_1.values()) + sum(better_2.values()),
        sep='\t')
