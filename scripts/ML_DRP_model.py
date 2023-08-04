"""Functions for training and evaluating DRP Tensorflow and XGBoost models"""

import numpy as np
import pandas as pd
import time
from sklearn.metrics import r2_score, mean_squared_error
from scripts.data_selection import cblind_split, data_indexing, crossval_pairs


def train_model(model, lr_scheduler, x_train, x_test, y_train, y_test, epochs):

    # unpackage x_train and x_test 
    xo_train, xd_train = x_train
    xo_test, xd_test = x_test

    # set learning rate scheduler
    callbacks_list =  [lr_scheduler]

    # train
    start = time.time() # start timer
    
    history = model.fit([xo_train, xd_train], y_train,
                        epochs=epochs, 
                        batch_size=None, 
                        verbose=1,
                        callbacks=callbacks_list)
    
    y_pred = model.predict([xo_test, xd_test])  

    end = time.time() # end timer
    result = end - start
    print('%.3f seconds' % result)

    # compare prediction to test data to get r2 and mse scores
    prediction_metrics(y_test, y_pred)



def train_model_multi(model, lr_scheduler, x_train, x_test, y_train, y_test, epochs):

    # unpackage x_train and x_test 
    xo_train_phos, xo_train_prot, xd_train = x_train
    xo_test_phos, xo_test_prot, xd_test = x_test

    # set learning rate scheduler
    callbacks_list =  [lr_scheduler]

    # train
    start = time.time() # start timer
    
    history = model.fit([xo_train_phos, xo_train_prot, xd_train], y_train,
                        epochs=epochs, 
                        batch_size=None, 
                        verbose=1,
                        callbacks=callbacks_list)

    y_pred = model.predict([xo_test_phos, xo_test_prot, xd_test], verbose=1)   

    end = time.time() # end timer
    result = end - start
    print('%.3f seconds' % result)

    # compare prediction to test data to get r2 and mse scores
    prediction_metrics(y_test, y_pred)



# Cross Validation function
def train_model_cv(model,lr_scheduler,train_pairs,x_all,x_drug,y_series,epochs,rand_seed_list):
    


    # take list of all cell lines in train pairs
    train_cls = [pair.split('::')[0] for pair in train_pairs] # split drug-cl pairs
    train_cls = [*set(train_cls)] # remove duplicates

    # create CV sets from train pairs
    train_CV_sets, eval_CV_sets = cblind_split(train_pairs, train_cls, rand_seed_list, k=3, verbose = 0)

    ## Model Testing

    num_seeds = len(rand_seed_list)
    k = len(train_CV_sets[0])
    all_r2 = []
    final_scores = [] # stores scores for all seeds tested on one set of params

    # loop for each seed
    for s in range(num_seeds):
        print(f'Seed {rand_seed_list[s]}:\n')
        train_2_pairs_set = train_CV_sets[s] # set the set of training k-fold splits for a seed
        eval_pairs_set = eval_CV_sets[s]     # set the set of eval k-fold splits for a seed
        seed_r2 = [] # stores all r2 scores for a seed

        # loop for each k-fold
        for i in range(k):
            print('K-fold', i+1)
            # set learning rate scheduler
            callbacks_list =  [lr_scheduler]
            # set training and eval pairs and split data for training
            train_pairs_2, eval_pairs = crossval_pairs(train_2_pairs_set,eval_pairs_set,i)
            xo_train, xd_train, y_train, xo_test, xd_test, y_test = data_indexing(train_pairs_2,eval_pairs,x_all,x_drug,y_series)
            
            # Start timer
            start = time.time() 
            # Train model
            history = model.fit([xo_train, xd_train], y_train,
                                validation_data=([xo_test, xd_test], y_test),
                                epochs=epochs, 
                                batch_size=None, 
                                verbose=1,
                                callbacks=callbacks_list)
            
            # End timer
            end = time.time()
            result = end - start
            print('%.3f seconds' % result)
            
            # model predictions
            y_pred = model.predict([xo_test, xd_test],verbose=0)
            seed_r2.append(r2_score(y_test, y_pred))
            prediction_metrics(y_test, y_pred)

        # print r2 scores for seed
        print(f'r2 Scores for seed {rand_seed_list[s]}:')
        print(*seed_r2, sep = '\n')
        # calculate mean r2 for seed
        mean_r2 = round(np.mean(seed_r2), 4)
        final_scores.append(mean_r2)
        print(f'\nmean r2: {mean_r2}')
        print('-----\n')
        # add seed r2 list to full list
        all_r2.extend(seed_r2)

    # print results for all seeds (maybe add mse later)
    print('Final results:')
    for seed,score in zip(rand_seed_list,final_scores):
        print(f'{seed}: {score}')



# Multi-omics Cross Validation function
def train_model_multi_cv(model,lr_scheduler,train_pairs,x_all,x_drug,y_series,epochs,rand_seed_list):
    
    # unpackage x_all
    x_all_phos, x_all_prot = x_all 

    # take list of all cell lines in train pairs
    train_cls = [pair.split('::')[0] for pair in train_pairs] # split drug-cl pairs
    train_cls = [*set(train_cls)] # remove duplicates

    # create CV sets from train pairs
    train_CV_sets, eval_CV_sets = cblind_split(train_pairs, train_cls, rand_seed_list, k=3, verbose = 0)

    ## Model Testing

    num_seeds = len(rand_seed_list)
    k = len(train_CV_sets[0])
    all_r2 = []
    final_scores = [] # stores scores for all seeds tested on one set of params

    # loop for each seed
    for s in range(num_seeds):
        print(f'Seed {rand_seed_list[s]}:\n')
        train_2_pairs_set = train_CV_sets[s] # set the set of training k-fold splits for a seed
        eval_pairs_set = eval_CV_sets[s]     # set the set of eval k-fold splits for a seed
        seed_r2 = [] # stores all r2 scores for a seed

        # loop for each k-fold
        for i in range(k):
            print('K-fold', i+1)
            # set learning rate scheduler
            callbacks_list =  [lr_scheduler]

            # set training and eval pairs and split data for training
            train_pairs_2, eval_pairs = crossval_pairs(train_2_pairs_set,eval_pairs_set,i)
            xo_train_phos, xd_train, y_train, xo_test_phos, xd_test, y_test = data_indexing(train_pairs,eval_pairs,x_all_phos,x_drug,y_series) 
            xo_train_prot, xd_train, y_train, xo_test_prot, xd_test, y_test = data_indexing(train_pairs,eval_pairs,x_all_prot,x_drug,y_series)
            
            # Start timer
            start = time.time() 
            # Train model
            history = model.fit([xo_train_phos, xo_train_prot, xd_train], y_train,
                                validation_data=([xo_test_phos, xo_test_prot, xd_test], y_test),
                                epochs=epochs, 
                                batch_size=None, 
                                verbose=1,
                                callbacks=callbacks_list)
            
            # End timer
            end = time.time()
            result = end - start
            print('%.3f seconds' % result)
            
            # model predictions
            y_pred = model.predict([xo_test_phos, xo_test_prot, xd_test], verbose=0)  
            seed_r2.append(r2_score(y_test, y_pred))
            prediction_metrics(y_test, y_pred)

        # print r2 scores for seed
        print(f'r2 Scores for seed {rand_seed_list[s]}:')
        print(*seed_r2, sep = '\n')
        # calculate mean r2 for seed
        mean_r2 = round(np.mean(seed_r2), 4)
        final_scores.append(mean_r2)
        print(f'\nmean r2: {mean_r2}')
        print('-----\n')
        # add seed r2 list to full list
        all_r2.extend(seed_r2)

    # print results for all seeds (add mse later)
    print('Final results:')
    for seed,score in zip(rand_seed_list,final_scores):
        print(f'{seed}: {score}')



# Preparing SMILES data for model
def prep_xd(xd_train):
    """Takes xd_train or xd_test dataframes and returns a 3D array of values"""
    # prepare SMILES data arrays
    samples = len(xd_train.iloc[:,0].values) # number of samples for zeroed array
    # convert xd values to a proper 3d array
    max_len = len(xd_train.iloc[:,0][0])
    max_char = len(xd_train.iloc[:,0][0][1])
    xd_vals = np.zeros(shape=(samples, max_len, max_char))
    for ind,array in enumerate(xd_train.iloc[:,0].values):
        xd_vals[ind] = array
        
    return xd_vals



def train_model_SMILES(model, lr_scheduler, x_train, x_test, y_train, y_test, epochs):

    # unpackage x_train and x_test 
    xo_train, xd_train = x_train
    xo_test, xd_test = x_test
        
    # convert drug dataframes to 3D arrays
    xd_train_vals = prep_xd(xd_train)
    xd_test_vals  = prep_xd(xd_test)

    # set learning rate scheduler
    callbacks_list =  [lr_scheduler]

    # Start timer
    start = time.time() 
    # Train model
    history = model.fit([xo_train, xd_train_vals], y_train,
                        epochs=epochs, 
                        batch_size=None, 
                        verbose=1,
                        callbacks=callbacks_list)
    
    y_pred = model.predict([xo_test, xd_test_vals])  

    end = time.time() # end timer
    result = end - start
    print('%.3f seconds' % result)

    # compare prediction to test data to get r2 and mse scores
    prediction_metrics(y_test, y_pred)



# Cross Validation function for SMILES model
def train_model_SMILES_cv(model,lr_scheduler,train_pairs,x_all,x_drug,y_series,epochs,rand_seed_list):
    
    # take list of all cell lines in train pairs
    train_cls = [pair.split('::')[0] for pair in train_pairs] # split drug-cl pairs
    train_cls = [*set(train_cls)] # remove duplicates

    # create CV sets from train pairs
    train_CV_sets, eval_CV_sets = cblind_split(train_pairs, train_cls, rand_seed_list, k=3, verbose = 0)

    ## Model Testing

    num_seeds = len(rand_seed_list)
    k = len(train_CV_sets[0])
    all_r2 = []
    final_scores = [] # stores scores for all seeds tested on one set of params

    # loop for each seed
    for s in range(num_seeds):
        print(f'Seed {rand_seed_list[s]}:\n')
        train_2_pairs_set = train_CV_sets[s] # set the set of training k-fold splits for a seed
        eval_pairs_set = eval_CV_sets[s]     # set the set of eval k-fold splits for a seed
        seed_r2 = [] # stores all r2 scores for a seed

        # loop for each k-fold
        for i in range(k):
            print('K-fold', i+1)
            # set learning rate scheduler
            callbacks_list =  [lr_scheduler]

            # set training and eval pairs and split data for training
            train_pairs_2, eval_pairs = crossval_pairs(train_2_pairs_set,eval_pairs_set,i)
            xo_train, xd_train, y_train, xo_test, xd_test, y_test = data_indexing(train_pairs_2,eval_pairs,x_all,x_drug,y_series)
            
            # convert drug dataframes to 3D arrays
            xd_train_vals = prep_xd(xd_train)
            xd_test_vals  = prep_xd(xd_test)

            # Start timer
            start = time.time() 
            # Train model
            history = model.fit([xo_train, xd_train_vals], y_train,
                                validation_data=([xo_test, xd_test_vals], y_test),
                                epochs=epochs, 
                                batch_size=None, 
                                verbose=1,
                                callbacks=callbacks_list)
            
            # End timer
            end = time.time()
            result = end - start
            print('%.3f seconds' % result)
            
            # model predictions
            y_pred = model.predict([xo_test, xd_test_vals],verbose=0)
            seed_r2.append(r2_score(y_test, y_pred))
            prediction_metrics(y_test, y_pred)

        # print r2 scores for seed
        print(f'r2 Scores for seed {rand_seed_list[s]}:')
        print(*seed_r2, sep = '\n')
        # calculate mean r2 for seed
        mean_r2 = round(np.mean(seed_r2), 4)
        final_scores.append(mean_r2)
        print(f'\nmean r2: {mean_r2}')
        print('-----\n')
        # add seed r2 list to full list
        all_r2.extend(seed_r2)

    # print results for all seeds (maybe add mse later)
    print('Final results:')
    for seed,score in zip(rand_seed_list,final_scores):
        print(f'{seed}: {score}')



def train_model_XGBr(model, x_train, x_test, y_train, y_test):

    # unpackage x_train and x_test 
    xo_train, xd_train = x_train
    xo_test, xd_test = x_test

    # concatenate one hot encoded drugs to phosphoproteomics dataframe for RF
    X_train = pd.concat([xo_train, xd_train], axis=1)
    X_test = pd.concat([xo_test, xd_test], axis=1)
    # convert all column names from x_drug to str
    X_train.columns = X_train.columns.map(str)
    X_test.columns = X_test.columns.map(str)

    # train
    start = time.time() # start timer
    
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    end = time.time() # end timer
    result = end - start
    print('%.3f seconds' % result)

    # compare prediction to test data to get r2 and mse scores
    prediction_metrics(y_test, y_pred)



def prediction_metrics(y_test, y_pred):
    """Takes y_pred predictions, y_test target values and calculates r2 and mse metrics to be printed by function"""
    print('r2  score: ', r2_score(y_test, y_pred))
    print('mse score: ', mean_squared_error(y_test, y_pred))
    print('-----')
