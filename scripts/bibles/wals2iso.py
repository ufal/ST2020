#!/usr/bin/env python3
#coding: utf-8

# Rudolf Rosa

import sys
import csv

wals2iso = dict()
with open('data/wals.tsv') as wals:
    for line in wals:
        wals_code, iso_code = line.split()[:2]
        if iso_code:
            wals2iso[wals_code] = iso_code
    
with open('data/dev_y.csv') as dev:
    d2 = csv.DictReader(dev)
    for dictline in d2:
        wals_code = dictline['wals_code']
        if wals_code in wals2iso:
            print(wals2iso[wals_code])

