#!/bin/bash

set -o xtrace


python3 evaluate_from_csv.py \
    --input_file ../data/dev_x.csv \
    --output_file $1 \
    --golden_file ../data/dev_y.csv

