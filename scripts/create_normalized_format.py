import re
import argparse
import os 

import pandas as pd
import numpy as np

WALS_CODE = 0
NAME = 1
LATITUDE = 2
LONGTITUDE = 3
GENUS = 4
FAMILY = 5
COUNTRY_CODES = 6
FEATURES = 7

NUMBER_OF_ALWAYS_FILLED = 7


def parse_header(header):
    return header.split()[:-1]

def get_all_possible_features(lines):
    all_features = set()
    for line in lines:
        features_start = [m.start() for m in re.finditer('\t', line)] # get the position of tab character
        features = line[features_start[COUNTRY_CODES]+1:].split('|') # split the features after countries column
        for feature in features:
            all_features.add(feature.split('=')[0]) # choose only text up until the first =
    return all_features

def feature_int_map(all_features):
    """
    Creates maping of feature to int and int to feature
    """
    feature_to_int = {}
    int_to_feature = []
    i = 0
    for feature in all_features:
        feature_to_int[feature] = i
        i += 1
        int_to_feature.append(feature)
    return int_to_feature, feature_to_int

def convert_line(line, features_to_int):
    """
    Takes non header line and creates new vector of all ordered columns by features_to_int
    """
    features_start = [m.start() for m in re.finditer('\t', line)] # finds positions of all tabs
    main_features = line[:features_start[COUNTRY_CODES]].split('\t') # splits the line all up to tab after country codes
    result = [np.nan] * (len(features_to_int) + len(main_features)) # fill the array with nan values
    
    for i in range(len(main_features)): # fill the first values
        result[i] = main_features[i]
    

    features = line[features_start[COUNTRY_CODES]+1:].split('|') # split the features by |
    for feature in features:
        equal_symbol = feature.find('=') # finds the position of = in features
        key, value = feature[:equal_symbol], feature[equal_symbol+1:] # splits the feature into key, value
        index = len(main_features) + features_to_int[key] # moves the index so it doesn't overlap with previously filled values such as wals code, etc...
        result[index] = value
    return result
        

def parse_sigtyp_format(file_path, given_header=None):
    """
    Converts the SIGTYP format to matrix and header
    """
    with open(file_path, 'r') as file:
        lines = file.readlines() 
    lines = [i.rstrip() for i in lines]
    header = parse_header(lines[0])
    
    result = []
    if given_header is None: 
        all_features = get_all_possible_features(lines[1:])
    else:
        all_features = given_header[7:] # if the header is already given, take only the features
    
        
    int_to_feature, features_to_int = feature_int_map(all_features)

    # iterate over all non header lines
    for line in lines[1:]: 
        result.append(convert_line(line, features_to_int))

    # creates header
    for feature in int_to_feature:
        header.append(feature)

    return result, header

def fill_randomly_missing_values(data):
    """
    Randomly fills the mask to data.
    """
    random_missing = np.vectorize(lambda x: x if np.random.uniform() < 0.5 else "?" ) # random masking
    mask = (data != 'nan') # find where the data are nan
    mask[:,:7] = False # we don't want to mask first 7 columns
    data[mask] = random_missing(data[mask]) # fill the values
    return data

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="../data/train.csv", help="Path to train file.")
    parser.add_argument("--dev", type=str, default="../data/dev.csv", help="Path to dev file.")
    parser.add_argument("--output_dir", type=str, default="../data/", help="Path to output folder.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    args = parser.parse_args()

    # Fix random seeds and threads
    np.random.seed(args.seed)

    train, header = parse_sigtyp_format(args.train)
    dev, _ = parse_sigtyp_format(args.dev, header) # we pass the header since we want to have same ordered header as in train values

    # create expected output
    train_y = pd.DataFrame(data=train, columns=header)
    dev_y = pd.DataFrame(data=dev, columns=header)

    # randomly mask the expected values
    train_X = fill_randomly_missing_values(np.array(train))
    dev_X = fill_randomly_missing_values(np.array(dev))

    # creates the input data
    train_X = pd.DataFrame(data=train_X, columns=header)
    dev_X = pd.DataFrame(data=dev_X, columns=header)

    # saves all to the csv
    train_X.to_csv(os.path.join(args.output_dir, 'train_x.csv'))
    train_y.to_csv(os.path.join(args.output_dir, 'train_y.csv'))

    dev_X.to_csv(os.path.join(args.output_dir, 'dev_x.csv'))
    dev_y.to_csv(os.path.join(args.output_dir, 'dev_y.csv'))