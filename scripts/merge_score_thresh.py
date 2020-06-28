#!/usr/bin/env python3
#coding: utf-8

# Rudolf Rosa

import sys
import csv
from collections import OrderedDict

import logging
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)

dev_x = csv.DictReader(open(sys.argv[1]))
out_1 = csv.DictReader(open(sys.argv[2]))
out_1_scores = csv.DictReader(open(sys.argv[3]))
out_2 = csv.DictReader(open(sys.argv[4]))
out_2_scores = csv.DictReader(open(sys.argv[5]))
thresh_1 = float(sys.argv[6])
thresh_2 = float(sys.argv[7])

# DESCRIPTION
# Uses out_1 if score_1 > thresh_1 and score_2 < thresh_2
# Otherwise uses out_2

count_1 = 0
count_1_diff = 0
count_2 = 0
count_2_diff = 0

output = list()
for line_x, line_1, line_1_score, line_2, line_2_score in zip(
        dev_x, out_1, out_1_scores, out_2, out_2_scores):
    outline = OrderedDict()
    for feat in line_x:
        # feature to predict
        if line_x[feat] == '?':
            # get scores
            score_1 = float(line_1_score[feat])
            score_2 = float(line_2_score[feat])
            # decide which to choose
            if score_1 > thresh_1 and score_2 < thresh_2:
                # choosing Dan
                outline[feat] = line_1[feat]
                count_1 += 1
                count_1_diff += line_1[feat] != line_2[feat]
            else:
                # choosing Martin
                outline[feat] = line_2[feat]
                count_2 += 1
                count_2_diff += line_1[feat] != line_2[feat]
        else:
            # feature not to predict
            outline[feat] = line_x[feat]
    # output line
    output.append(outline)

# write output
outwriter = csv.DictWriter(sys.stdout, output[0].keys())
outwriter.writeheader()
for outline in output:
    outwriter.writerow(outline)

logging.info('Used {} times input 1 ({} times when differed from input 2)'.format(
            count_1, count_1_diff))
logging.info('Used {} times input 2 ({} times when differed from input 1)'.format(
            count_2, count_2_diff))

