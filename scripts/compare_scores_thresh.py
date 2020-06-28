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
thresh_1 = float(sys.argv[7])
thresh_2 = float(sys.argv[8])

# dict[feat] = count
total = 0
correct_1 = 0
correct_2 = 0
correct_any = 0
correct_thresh = 0
better_1 = 0
better_2 = 0
correct_1_scores = list()
incorrect_1_scores = list()

def percent(share, total):
    if total > 0:
        return '{:.2f}'.format(share*100/total)
    else:
        return '--'

def tdp(number):
    return '{:.3f}'.format(number)

for line_x, line_y, line_1, line_1_score, line_2, line_2_score in zip(
        dev_x, dev_y, out_1, out_1_scores, out_2, out_2_scores):
    for feat in line_x:
        if line_x[feat] == '?':
            is_correct_1 = line_y[feat] == line_1[feat]
            is_correct_2 = line_y[feat] == line_2[feat]
            score_1 = float(line_1_score[feat])
            score_2 = float(line_2_score[feat])
            
            if score_1 > thresh_1 and score_2 < thresh_2:
                # choosing Dan
                is_correct_thresh = is_correct_1
                choose_1 = True
            else:
                # choosing Martin
                is_correct_thresh = is_correct_2
                choose_1 = False

            total += 1
            correct_1 += is_correct_1
            correct_2 += is_correct_2
            correct_any += (is_correct_1 or is_correct_2)
            correct_thresh += is_correct_thresh
            
            if is_correct_1 and not is_correct_2:
                better_1 += 1
            if is_correct_2 and not is_correct_1:
                better_2 += 1
 
            if choose_1:
                if is_correct_thresh:
                    correct_1_scores.append(score_1)
                else:
                    incorrect_1_scores.append(score_1)

def tuple2first(tup):
    return tup[0]


dan_sc_bw = list()
for score in correct_1_scores:
    dan_sc_bw.append((score, True))
for score in incorrect_1_scores:
    dan_sc_bw.append((score, False))
dan_sc_bw.sort(key=tuple2first, reverse=True)

print()
print('Dan', 'Better?', '% better', sep='\t')
better_count = 0
total_count = 0
for score, better in dan_sc_bw:
    total_count += 1
    better_count += better
    print(tdp(score), better, percent(better_count, total_count), sep='\t')



print()
print('Feature', 'Acc 1', 'Acc 2', 'Acc *', 'Acc thresh', sep='\t')
print('TOTAL',
        percent(correct_1, total),
        percent(correct_2, total),
        percent(correct_any, total),
        percent(correct_thresh, total),
        sep='\t')
print(total,
        correct_1,
        correct_2,
        correct_any,
        correct_thresh,
        sep='\t')

print()
print('Setup', 'Eval', 'Min', 'Max', 'Avg', sep='\t')

print('Dan if used',
        'Incorr.',
        tdp(min(incorrect_1_scores)),
        tdp(max(incorrect_1_scores)),
        tdp(sum(incorrect_1_scores)/len(incorrect_1_scores)),
        sep='\t')
print('Dan if used',
        'Correct',
        tdp(min(correct_1_scores)),
        tdp(max(correct_1_scores)),
        tdp(sum(correct_1_scores)/len(correct_1_scores)),
        sep='\t')


