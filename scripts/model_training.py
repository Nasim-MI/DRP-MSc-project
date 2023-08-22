"""Functions for training and evaluating DRP Tensorflow and XGBoost models"""

import numpy as np
import pandas as pd
import time
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
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
                        verbose=0,
                        callbacks=callbacks_list)
    
    y_pred = model.predict([xo_test, xd_test])  

    end = time.time() # end timer
    result = end - start
    print('%.3f seconds' % result)

    # compare prediction to test data to get r2 and mse scores
    prediction_metrics(y_test, y_pred)
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    pearson = pearsonr(y_test, y_pred)
    
    return([r2,mse,pearson])



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
                        verbose=0,
                        callbacks=callbacks_list)

    y_pred = model.predict([xo_test_phos, xo_test_prot, xd_test], verbose=1)   

    end = time.time() # end timer
    result = end - start
    print('%.3f seconds' % result)

    # compare prediction to test data to get r2 and mse scores
    prediction_metrics(y_test, y_pred)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    pearson = pearsonr(y_test, y_pred)
    
    return([r2,mse,pearson])



# Preparing SMILES data for model
def prep_xd(xd_train):
    """Takes xd_train or xd_test dataframes and returns a 3D array of values"""
    # prepare SMILES data arrays
    samples = len(xd_train[0].values) # number of samples for zeroed array
    # convert xd values to a proper 3d array
    max_len = len(xd_train[0][0])
    max_char = len(xd_train[0][0][1])
    xd_vals = np.zeros(shape=(samples, max_len, max_char))
    for ind,array in enumerate(xd_train[0].values):
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
                        verbose=0,
                        callbacks=callbacks_list)
    
    y_pred = model.predict([xo_test, xd_test_vals])  

    end = time.time() # end timer
    result = end - start
    print('%.3f seconds' % result)

    # compare prediction to test data to get r2 and mse scores
    prediction_metrics(y_test, y_pred)

    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    pearson = pearsonr(y_test, y_pred)
    
    return([r2,mse,pearson])



# For training XGBoost and Random Forest models
def train_model_ML(model, x_train, x_test, y_train, y_test):

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
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    pearson = pearsonr(y_test, y_pred)
    
    return([r2,mse,pearson])



def prediction_metrics(y_test, y_pred):
    """Takes y_pred predictions, y_test target values and calculates r2 and mse metrics to be printed by function"""
    print('r2  score: ', r2_score(y_test, y_pred))
    print('mse score: ', mean_squared_error(y_test, y_pred))
    print('-----')

    

def split_all_scores(all_scores):
    
    # take pearson score from tuple
    for score_list in all_scores:
        score_list[2] = score_list[2][0][0]
    
    r2_scores = []
    mse_scores = []
    pearson_scores = []
    
    for score_list in all_scores:
        r2_scores.append(score_list[0])
        mse_scores.append(score_list[1])
        pearson_scores.append(score_list[2])
        
    return r2_scores, mse_scores, pearson_scores



# scheduler function for learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)