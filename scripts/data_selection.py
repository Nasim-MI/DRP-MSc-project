"""Functions for indexing data using train/test pairs and preparing k-fold cross validation train/validation pairs using for splitting"""

import pandas as pd
import scipy
from sklearn.model_selection import RepeatedKFold


def mixed_kfold(train_pairs, k = 3, n = 1, rand_seed = 42):
    """Mixed-set splitting. Takes list drug-cell line pairs list and splits pairs by k-fold splitting. 
    Outputs separate train and validation pairs.
    Cell lines and drugs can overlap across splits
    """

    train_set = []
    test_set = []
    rkf = RepeatedKFold(n_splits=k, n_repeats=n, random_state=rand_seed)
    for train, test in rkf.split(train_pairs):
        train_set.append(train)
        test_set.append(test)

    train_2_pairs_set = []
    eval_pairs_set = []
    
    print(f'cblind K-fold CV: k = {k}, n = {n}, seed = {rand_seed} ')
    
    return train_2_pairs_set, eval_pairs_set



def mixed_split(train_pairs,_all_cls,seed_list, k=3, n=1):
    """Mixed-set splitting. Takes list drug-cell line pairs list and splits pairs by k-fold splitting for each seed in seed_list. 
    Outputs separate train and validation pairs.
    Cell lines and drugs can overlap across splits.
    """

    # create lists to store training and eval split indices
    train_2_pairs_set = [] # training pairs for 1 seed
    eval_pairs_set = [] # eval pairs for 1 seed

    train_CV_sets = [] # training pairs for all seeds
    eval_CV_sets = [] # eval pairs for all seeds

    # create cblind kfold sets for cross validation
    for i in range(len((seed_list))):
        cv_seed = seed_list[i]
        train_2_pairs_set, eval_pairs_set = mixed_kfold(train_pairs, k, n, rand_seed=cv_seed)
        train_CV_sets.append(train_2_pairs_set)
        eval_CV_sets.append(eval_pairs_set)
    
    return train_CV_sets, eval_CV_sets



# cancer blind splitting, cell lines cannot overlap, drugs can overlap 
def cblind_kfold(train_pairs, _all_cls, k = 3, n = 1, rand_seed = 42, verbose=0):
    """Cancer-blind splitting. Takes list drug-cell line pairs list and splits pairs by k-fold splitting. 
    Outputs separate train and validation pairs.
    Cell lines cannot overlap across splits, drugs can overlap.
    """

    # create empty lists for cls and drugs in training data
    train_cls = []
    # add training cls to new list
    for pair in train_pairs:
        for cl in _all_cls:
            if cl in pair:
                train_cls.append(cl)
    # remove duplicate cls from list      
    train_cls = list(dict.fromkeys(train_cls)) 

    # splits cell lines into 6 sets of 0.8 train - 0.2 test split
    train_set = []
    test_set = []
    rkf = RepeatedKFold(n_splits=k, n_repeats=n, random_state=rand_seed)
    for train, test in rkf.split(train_cls):
        train_set.append(train)
        test_set.append(test)
    # loop train_2-eval pairs creation for all train-cl indicies
    train_2_pairs_set = []
    eval_pairs_set = []
    
    for i in range(k*n):
        # use k-fold indicies to find cl names
        train_2_cls = pd.Series(train_cls).iloc[train_set[i]] # k-fold split cls for train
        eval_cls = pd.Series(train_cls).iloc[test_set[i]] # k-fold split cls for test

        # create lists for training and eval pairs
        train_pairs_2 = []
        eval_pairs = []
        # add train pairs with train_2 cls to list
        for pair in train_pairs:
            for cl in train_2_cls:
                if cl in pair:
                    train_pairs_2.append(pair)
        # add train pairs with eval cls to list
        for pair in train_pairs:
            for cl in eval_cls:
                if cl in pair:
                    eval_pairs.append(pair)
        # remove duplicate pairs from lists
        train_pairs_2 = list(dict.fromkeys(train_pairs_2)) 
        eval_pairs = list(dict.fromkeys(eval_pairs)) 
        
        # validate that there is no cl overlap 
        eval_pairs_cls = [pair.split('::', 1)[0] for pair in eval_pairs]
        train_pairs_2_cls = [pair.split('::', 1)[0] for pair in train_pairs_2]
        len(set(train_pairs_2_cls).intersection(eval_pairs_cls)) 
        # identify offending cls
        common_train_cls = list(set(train_pairs_2_cls).intersection(eval_pairs_cls))
        
        # find pairs in training set with overlapping cls
        # non-cblind in training pairs
        non_cblind_train_pairs = [] 
        for pair in train_pairs_2:
            for cl in common_train_cls:
                if cl in pair:
                    non_cblind_train_pairs.append(pair)
        # non-cblind in eval pairs
        non_cblind_eval_pairs = []
        for pair in eval_pairs:
            for cl in common_train_cls:
                if cl in pair:
                    non_cblind_eval_pairs.append(pair)
                    
        # identify which list had the least overlapping pairs for removal
        if len(non_cblind_train_pairs) < len(non_cblind_eval_pairs):
            #print(f'Overlapping cls removed: {len(non_cblind_train_pairs)}') 
            for pair in non_cblind_train_pairs:
                if pair in train_pairs_2:
                    train_pairs_2.remove(pair)
        else:
            #print(f'Overlapping cls removed: {len(non_cblind_eval_pairs)}') 
            for pair in non_cblind_eval_pairs:
                if pair in eval_pairs:
                    eval_pairs.remove(pair)
                
        # add train and eval pairs to list
        train_2_pairs_set.append(train_pairs_2)
        eval_pairs_set.append(eval_pairs)

    if verbose==1:
        print(f'cblind K-fold CV: k = {k}, n = {n}, seed = {rand_seed} ')
    
    return train_2_pairs_set, eval_pairs_set




def cblind_split(train_pairs,_all_cls,seed_list, k=3,verbose=0):
    """Cancer-blind splitting. Takes list drug-cell line pairs list and splits pairs by k-fold splitting for each seed in seed_list. 
    Outputs separate train and validation pairs.
    Cell lines cannot overlap across splits, drugs can overlap.
    """
    
    # create lists to store training and eval split indices
    train_2_pairs_set = [] # training pairs for 1 seed
    eval_pairs_set = [] # eval pairs for 1 seed

    train_CV_sets = [] # training pairs for all seeds
    eval_CV_sets = [] # eval pairs for all seeds

    # create cblind kfold sets for cross validation
    for i in range(len((seed_list))):
        cv_seed = seed_list[i]
        train_2_pairs_set, eval_pairs_set = cblind_kfold(train_pairs, _all_cls, k, n=1, rand_seed=cv_seed, verbose=verbose)
        train_CV_sets.append(train_2_pairs_set)
        eval_CV_sets.append(eval_pairs_set)
    
    return train_CV_sets, eval_CV_sets




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