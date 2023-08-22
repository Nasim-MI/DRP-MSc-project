import os, sys
import random
import pickle
import numpy as np
from scripts.data_wrangling import import_phos
from scripts.data_preparation import data_prep
from scripts.splitting import split

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
# function for creating train-test splits and storing
def create_train_test_pairs(n_pairs=10):
    
    # set random seeds for python and numpy
    random.seed(42)
    np.random.seed(seed=42)
    # import phosphoproteomics data
    phos_df, drug_df, drug_matrix, _all_cls, _all_drugs, common_ind = import_phos()
    # Create dataframes for omics, drugs and target values
    x_drug, x_all, y_series = data_prep(drug_df,phos_df,common_ind)
    

    # initial test train split indexes
    rand_seed_list = [871, 539, 25, 34, 114, 409, 573, 854, 220, 925]
    print('Random seeds for train-test splitting: ',rand_seed_list)
    
    train_test_list = []
    
    _train_size = 0.75 # set train-test ratio
    pairs_with_truth_vals =  y_series.index

    for rand_seed in rand_seed_list:
        with HiddenPrints():
            train_pairs, test_pairs = split(rand_seed, _all_cls, _all_drugs, pairs_with_truth_vals,
                                                    train_size=_train_size, split_type='cblind')

            train_test_list.append([train_pairs, test_pairs])
    
    # save training pairs
    train_pairs_path = f'train_test_pairs/_train_test_pairs_20.ob'
    with open(train_pairs_path, 'wb') as f:
        pickle.dump(train_test_list, f)


# create train and test pairs list
create_train_test_pairs(n_pairs=10)
     