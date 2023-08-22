"""Functions for indexing data using train/test pairs and preparing k-fold cross validation train/validation pairs using for splitting"""

import random
import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import RepeatedKFold



def data_indexing(train_pairs,test_pairs,x_all,x_drug,y_series):
    """Using train and test pairs, splits omics, drug and target value data by train and test data samples"""  
    # standardise y_series
    y_series_st = scipy.stats.zscore(y_series) # standardisation using z-score normalisation
    # test train selection
    xo_train, xo_test = x_all.loc[train_pairs], x_all.loc[test_pairs] 
    xd_train, xd_test = x_drug.loc[train_pairs], x_drug.loc[test_pairs] #drug dfs
    y_train, y_test = y_series_st[train_pairs], y_series_st[test_pairs] #target series

    return xo_train, xd_train, y_train, xo_test, xd_test, y_test



def crossval_pairs(train_2_pairs_set,eval_pairs_set,index):
    """Set training and eval pairs for cross-val sets"""
    i = index
    train_pairs_2 = train_2_pairs_set[i]  
    eval_pairs = eval_pairs_set[i]

    return train_pairs_2, eval_pairs



def expand_replicates(train_pairs,test_pairs):
    """Takes two lists of drug-cl pairs (train-test or train-eval) and returns twos list with cell line names of replicates added"""
    # lists for triplicate pairs
    train_pairs_triplicate = []
    test_pairs_triplicate = []
    # add replicates for train pairs
    for pair in train_pairs:
        cl,drug = pair.split('::')
        replicate_0 = pair
        replicate_1 = '::'.join([f'{cl}-1',drug])
        replicate_2 = '::'.join([f'{cl}-2',drug])
        train_pairs_triplicate.extend([replicate_0,replicate_1,replicate_2])
    # add replicates for test/eval pairs    
    for pair in test_pairs:
        cl,drug = pair.split('::')
        replicate_0 = pair
        replicate_1 = '::'.join([f'{cl}-1',drug])
        replicate_2 = '::'.join([f'{cl}-2',drug])
        test_pairs_triplicate.extend([replicate_0,replicate_1,replicate_2])
        
    return train_pairs_triplicate,test_pairs_triplicate