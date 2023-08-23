import os, sys
import random
import pickle
import numpy as np
from scripts.data_wrangling import import_phos, import_SMILES
from scripts.data_preparation import data_prep_SMILES, encode_SMILES
from scripts.splitting import split

# function for creating a list of train-test pairs which can be imported for testing models. Used for SMILES and molecular fingerprint models  models (312 drugs)

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
# function for creating train-test splits and storing
def create_train_test_pairs_alt(n_pairs=10):
    
    # set random seeds for python and numpy
    random.seed(42)
    np.random.seed(seed=42)
    phos_df, drug_df, drug_matrix, _all_cls, _all_drugs, common_ind  = import_phos()
    SMILES_df = import_SMILES(drug_matrix)
    # one-hot encode SMILES and prepare create dataframes 
    one_hot_SMILES = encode_SMILES(SMILES_df) # one-hot encode SMILES strings
    x_drug, x_all, y_series = data_prep_SMILES(drug_df,one_hot_SMILES,phos_df,common_ind)
    

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
    train_pairs_path = f'train_test_pairs/_train_test_pairs_alt_2.ob'
    with open(train_pairs_path, 'wb') as f:
        pickle.dump(train_test_list, f)


# create train and test pairs list
create_train_test_pairs_alt(n_pairs=10)
     