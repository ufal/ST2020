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
out_1_scores = csv.DictReader(open(sys.argv[4]))
out_2 = csv.DictReader(open(sys.argv[5]))
out_2_scores = csv.DictReader(open(sys.argv[6]))

# dict[feat] = count
total = defaultdict(int)
correct_1 = defaultdict(int)
correct_2 = defaultdict(int)
correct_any = defaultdict(int)
better_1 = defaultdict(int)
better_2 = defaultdict(int)
correct_1_scores = list()
incorrect_1_scores = list()
correct_2_scores = list()
incorrect_2_scores = list()

def percent(share, total):
    if total > 0:
        return '{:.2f}%'.format(share*100/total)
    else:
        return '--'

def tdp(number):
    return '{:.2f}'.format(number)

for line_x, line_y, line_1, line_1_score, line_2, line_2_score in zip(
        dev_x, dev_y, out_1, out_1_scores, out_2, out_2_scores):
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

            score_1 = float(line_1_score[feat])
            score_2 = float(line_2_score[feat])
            if is_correct_1:
                correct_1_scores.append(score_1)
            else:
                incorrect_1_scores.append(score_1)
            if is_correct_2:
                correct_2_scores.append(score_2)
            else:
                incorrect_2_scores.append(score_2)

print()
print('Feature', 'Acc 1', 'Acc 2', 'Acc *', '1 > 2', 'Support', sep='\t')
for feat in total:
    print(feat,
            percent(correct_1[feat], total[feat]),
            percent(correct_2[feat], total[feat]),
            percent(correct_any[feat], total[feat]),
            percent(better_1[feat], (better_1[feat] + better_2[feat])),
            better_1[feat] + better_2[feat],
            sep='\t')

print()
print('Feature', 'Acc 1', 'Acc 2', 'Acc *', '1 > 2', 'Support', sep='\t')
print('TOTAL',
        percent(sum(correct_1.values()), sum(total.values())),
        percent(sum(correct_2.values()), sum(total.values())),
        percent(sum(correct_any.values()), sum(total.values())),
        percent(sum(better_1.values()), sum(better_1.values()) + sum(better_2.values())),
        sum(better_1.values()) + sum(better_2.values()),
        sep='\t')

print()
print('Setup', 'Eval', 'Min', 'Max', 'Avg', sep='\t')

print('Dan',
        'Incorr.',
        tdp(min(incorrect_1_scores)),
        tdp(max(incorrect_1_scores)),
        tdp(sum(incorrect_1_scores)/len(incorrect_1_scores)),
        sep='\t')
print('Dan',
        'Correct',
        tdp(min(correct_1_scores)),
        tdp(max(correct_1_scores)),
        tdp(sum(correct_1_scores)/len(correct_1_scores)),
        sep='\t')
print('Martin',
        'Incorr.',
        tdp(min(incorrect_2_scores)),
        tdp(max(incorrect_2_scores)),
        tdp(sum(incorrect_2_scores)/len(incorrect_2_scores)),
        sep='\t')
print('Martin',
        'Correct',
        tdp(min(correct_2_scores)),
        tdp(max(correct_2_scores)),
        tdp(sum(correct_2_scores)/len(correct_2_scores)),
        sep='\t')

# TODO correlation?

