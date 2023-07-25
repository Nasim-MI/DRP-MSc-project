"""Functions for preparing omics and drug data for model input. 
Omics models with no drug specific data are duplicated for every drug in drug data
Functions output omics dataframe, one-hot encoded drug dataframe and target value series with indices for each drug-cell line combination"""

import pandas as pd
import numpy as np
from rdkit import Chem # required for SMILES model

def y_series_replicates(y_series):
    '''Takes y_series of IC50 values with drug-cl index and returns new series with entries for replicates (cl,cl-1,cl-2)'''
    
    y_series_vals = []
    y_series_index = []
    for i in range(len(y_series)):
        # extract value and drug-cl pair name
        val =  y_series[i]
        pair = y_series.index[i]

        # create replicate names for new index
        cl,drug = pair.split('::')
        replicate_0 = pair
        replicate_1 = '::'.join([f'{cl}-1',drug])
        replicate_2 = '::'.join([f'{cl}-2',drug])
        y_series_index.extend([replicate_0,replicate_1,replicate_2])

        # add value to list of val for each replicate
        y_series_vals.extend([val] * 3)

    # create new y_series
    y_series_triplicate = pd.Series(y_series_vals,index=y_series_index)
    
    return y_series_triplicate



def data_prep(drug_df,omics_df,common_ind):
    """For omics dataframe with cell line data which is not drug-specific, duplicates sample data for each drug in drug dataframe.
    Indices are consistent across output dataframes and series
    
    Inputs

    drug_df - pd dataframe.
    Contains target values, index is drug names
     
    omics_df - pd dataframe.
    Omics data, cell lines represented by index, features are cols 
     
    common_ind - list.
    List of common cell lines between omics and drug dataframes

    Outputs

    x_drug - pd dataframe.
    One-hot encoded representation of all drugs for every cell line

    x_all - pd dataframe.
    Omics data, cell line data repeated for every drug in drug_df
    
    y_series - pd series.
    Target values for all drugs and cell lines
    """


    ## Drug one-hot encoding (x_drug)

    # convert list of drugs into one-hot encoded dataframe
    drug_list = drug_df['DRUG_NAME'].drop_duplicates().to_list()
    one_hot_drugs = pd.get_dummies(drug_list).T



    ## Create Drug-Cell Line paired indexes 

    # get list of cl repeats
    cl_name_repeats = []
    for drug in one_hot_drugs:
        cl_name_repeats.extend(omics_df.index.values)
    # get list of drug repeats
    drug_name_repeats = np.repeat(one_hot_drugs.index,len(common_ind))
    # combine with repeated list of drugs
    drug_cl_index = []
    for cl,drug in zip(cl_name_repeats,drug_name_repeats):
        drug_cl_index.append(cl +'::'+ drug)
    # duplicate rows and add paired index
    x_drug = pd.DataFrame(np.repeat(one_hot_drugs.T.values, len(common_ind), axis=0), index=drug_cl_index)



    ## Drug-Cell Line pairing for Omics data (x_all)

    # reformat Omics data for drug-cell line pairing index
    # add repeats
    Omics_paired = pd.concat([omics_df]*len(one_hot_drugs)) # change to len(drugs) or something
    # set dataframe with repeated rows to have modified index
    Omics_paired.index = drug_cl_index
    x_all = Omics_paired



    ## Target IC50 series for Drug-Cell Line pairs (y_series)

    # filter to common cls
    drug_series_df = drug_df.loc[drug_df['CELL_LINE_NAME'].isin(common_ind)]
    # set index to cl column
    drug_series_df = drug_series_df.set_index('CELL_LINE_NAME')
    # build new index
    drug_cl_index = drug_series_df.index + '::' + drug_series_df['DRUG_NAME'].to_list() 
    # add new index, drop drug name column and convert to series
    drug_series_df = drug_series_df.set_index(drug_cl_index).drop('DRUG_NAME',axis=1) # add new index to dataframe
    drug_cl_series = drug_series_df['LN_IC50'] # create a series with new index and IC50 values
    drug_cl_series.index.name = None # remove index name
    y_series = drug_cl_series



    ## Final adjustments for consistent indexes across dataframes

    # filter down and reorder both one-hot and omics dataframes based on drug series index
    x_drug = x_drug.filter(drug_cl_series.index,axis=0)
    x_all = x_all.filter(drug_cl_series.index,axis=0)

    return x_drug, x_all, y_series



def data_prep_mixed(drug_df,omics_df,common_ind):
    """For omics dataframe with cell line data which is not drug-specific, duplicates sample data for each drug in drug dataframe. 
    Also outputs one-hot encoded dataframe representation of cell lines for mixed-set benchmark models
    Indices are consistent across output dataframes and series
    
    Inputs

    drug_df - pd dataframe.
    Contains target values, index is drug names
     
    omics_df - pd dataframe.
    Omics data, cell lines represented by index, features are cols 
     
    common_ind - list.
    List of common cell lines between omics and drug dataframes

    Outputs

    x_drug - pd dataframe.
    One-hot encoded representation of all drugs for every cell line

    x_cls - pd dataframe.
    One-hot encoded representation of all cell lines for every drug

    x_all - pd dataframe.
    Omics data, cell line data repeated for every drug in drug_df
    
    y_series - pd series.
    Target values for all drugs and cell lines
    """


    ## Drug one-hot encoding (x_drug)

    # convert list of drugs into one-hot encoded dataframe
    drug_list = drug_df['DRUG_NAME'].drop_duplicates().to_list()
    one_hot_drugs = pd.get_dummies(drug_list).T



    ## Create Drug-Cell Line paired indexes 

    # get list of cl repeats
    cl_name_repeats = []
    for drug in one_hot_drugs:
        cl_name_repeats.extend(omics_df.index.values)
    # get list of drug repeats
    drug_name_repeats = np.repeat(one_hot_drugs.index,len(common_ind))
    # combine with repeated list of drugs
    drug_cl_index = []
    for cl,drug in zip(cl_name_repeats,drug_name_repeats):
        drug_cl_index.append(cl +'::'+ drug)
    # duplicate rows and add paired index
    x_drug = pd.DataFrame(np.repeat(one_hot_drugs.T.values, len(common_ind), axis=0), index=drug_cl_index)
    
    
    
    ## One-hot encoded cell lines (x_cls) - used for mixed set benchmark
    
    # one-hot encode cell lines
    one_hot_cls = pd.get_dummies(common_ind).T
    
    # get list of drug repeats
    drug_name_repeats = []
    for cl in one_hot_cls:
        drug_name_repeats.extend(drug_list)
    # get list of cl repeats
    cl_name_repeats = np.repeat(one_hot_cls.index,len(drug_list))
    # combine with repeated list of drugs
    drug_cl_index = []
    for cl,drug in zip(cl_name_repeats,drug_name_repeats):
        drug_cl_index.append(cl +'::'+ drug)
    # duplicate rows and add paired index
    x_cls = pd.DataFrame(np.repeat(one_hot_cls.T.values, len(drug_list), axis=0), index=drug_cl_index)



    ## Drug-Cell Line pairing for omics data (x_all)

    # reformat omics data for drug-cell line pairing index
    # add repeats
    omics_paired = pd.concat([omics_df]*len(one_hot_drugs)) # change to len(drugs) or something
    # set dataframe with repeated rows to have modified index
    omics_paired.index = drug_cl_index
    x_all = omics_paired



    ## Target IC50 series for Drug-Cell Line pairs (y_series)

    # filter to common cls
    drug_series_df = drug_df.loc[drug_df['CELL_LINE_NAME'].isin(common_ind)]
    # set index to cl column
    drug_series_df = drug_series_df.set_index('CELL_LINE_NAME')
    # build new index
    drug_cl_index = drug_series_df.index + '::' + drug_series_df['DRUG_NAME'].to_list() 
    # add new index, drop drug name column and convert to series
    drug_series_df = drug_series_df.set_index(drug_cl_index).drop('DRUG_NAME',axis=1) # add new index to dataframe
    drug_cl_series = drug_series_df['LN_IC50'] # create a series with new index and IC50 values
    drug_cl_series.index.name = None # remove index name
    y_series = drug_cl_series



    ## Final adjustments for consistent indexes across dataframes

    # filter down and reorder both one-hot and omics dataframes based on drug series index
    x_drug = x_drug.filter(drug_cl_series.index,axis=0)
    x_cls = x_cls.filter(drug_cl_series.index,axis=0)
    x_all = x_all.filter(drug_cl_series.index,axis=0)

    return x_drug, x_cls, x_all, y_series



def data_prep_replicates(drug_df,omics_df,common_ind):
    """For omics dataframe with cell line data which is not drug-specific, duplicates sample data for each drug in drug dataframe.
    Used if all replicates are in triplicate omics dataset, target values are duplicated for each replicate
    Indices are consistent across output dataframes and series
    
    Inputs

    drug_df - pd dataframe.
    Contains target values, index is drug names
     
    omics_df - pd dataframe.
    Omics data, cell lines represented by index, features are cols 
     
    common_ind - list.
    List of common cell lines between omics and drug dataframes

    Outputs

    x_drug - pd dataframe.
    One-hot encoded representation of all drugs for every cell line

    x_all - pd dataframe.
    Omics data, cell line data repeated for every drug in drug_df
    
    y_series - pd series.
    Target values for all drugs and cell lines
    """



    ## Drug one-hot encoding (x_drug)

    # convert list of drugs into one-hot encoded dataframe
    drug_list = drug_df['DRUG_NAME'].drop_duplicates().to_list()
    one_hot_drugs = pd.get_dummies(drug_list).T

    ## Create Drug-Cell Line paired indexes 

    # get list of cl repeats
    cl_name_repeats = []
    for drug in one_hot_drugs:
        cl_name_repeats.extend(omics_df.index.values)
    cl_name_repeats
    # get list of drug repeats
    drug_name_repeats = np.repeat(one_hot_drugs.index,len(common_ind))
    # combine with repeated list of drugs
    drug_cl_index = []
    for cl,drug in zip(cl_name_repeats,drug_name_repeats):
        drug_cl_index.append(cl +'::'+ drug)
    # duplicate rows and add paired index
    x_drug = pd.DataFrame(np.repeat(one_hot_drugs.T.values, len(common_ind), axis=0), index=drug_cl_index)

    ## Drug-Cell Line pairing for phosphoproteomics data (x_all)

    # reformat phospho data for drug-cell line pairing index
    # add repeats
    phospho_paired = pd.concat([omics_df]*len(one_hot_drugs.columns)) 
    # set dataframe with repeated rows to have modified index
    phospho_paired.index = drug_cl_index
    x_all = phospho_paired

    ## Target IC50 series for Drug-Cell Line pairs (y_series)

    # filter to common cls
    drug_series_df = drug_df.loc[drug_df['CELL_LINE_NAME'].isin(common_ind)]

    # set index to cl column
    drug_series_df = drug_series_df.set_index('CELL_LINE_NAME')

    # build new index
    drug_cl_index = drug_series_df.index + '::' + drug_series_df['DRUG_NAME'].to_list() 

    # add new index, drop drug name column and convert to series
    drug_series_df = drug_series_df.set_index(drug_cl_index).drop('DRUG_NAME',axis=1) # add new index to dataframe
    drug_cl_series = drug_series_df['LN_IC50'] # create a series with new index and IC50 values
    drug_cl_series.index.name = None # remove index name
    y_series = drug_cl_series

    # extra step for replicates - create duplicate series entries for replicates
    y_series = y_series_replicates(y_series)

    ## Final adjustments for consistent indexes across dataframes

    # filter down and reorder both one-hot and phospho dataframes based on drug series index
    x_drug = x_drug.filter(y_series.index,axis=0)
    x_all = x_all.filter(y_series.index,axis=0)

    return x_drug, x_all, y_series





def encode_SMILES(SMILES_df):
    """Takes dataframe containing column with SMILES strings from import_SMILES function output and creates a one-hot encoded SMILES dataframe
    """
    SMILES_CHARS = [' ',
                    '#', '%', '(', ')', '+', '-', '.', '/',
                    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                    '=', '@',
                    'A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P',
                    'R', 'S', 'T', 'V', 'X', 'Z',
                    '[', '\\', ']',
                    'a', 'b', 'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's',
                    't', 'u']
    smi2index = dict( (c,i) for i,c in enumerate( SMILES_CHARS ) )
    index2smi = dict( (i,c) for i,c in enumerate( SMILES_CHARS ) )

    def smiles_encoder( smiles, maxlen=300 ):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles( smiles ))
        X = np.zeros( ( maxlen, len( SMILES_CHARS ) ) )
        for i, c in enumerate( smiles ):
            X[i, smi2index[c] ] = 1
        return X
    

    # find longest SMILES string length
    max_len = SMILES_df.SMILES.str.len().max()

    # create list of encoded SMILES
    SMILE_list_encoded = []
    for SMILE in SMILES_df['SMILES']:
        encoded_SMILE = smiles_encoder(SMILE,max_len)
        SMILE_list_encoded.append(encoded_SMILE)

    # add encoded SMILES to list
    SMILES_df['SMILE_encoded'] = SMILE_list_encoded

    # create separate one-hot encoded SMILES dataframe
    one_hot_SMILES = SMILES_df.drop('SMILES',axis=1).set_index('name')
    one_hot_SMILES = one_hot_SMILES[~one_hot_SMILES.index.duplicated(keep='first')] # remove rows with duplicate indicies

    return one_hot_SMILES



# Wrangle function for SMILES drug representation

def data_prep_SMILES(drug_df,one_hot_drugs,phospho_df,common_ind):
    """For omics dataframe with cell line data which is not drug-specific, duplicates sample data for each drug in drug dataframe.
    Used if drug representation is SMILES strings
    Indices are consistent across output dataframes and series

    Inputs

    drug_df - pd dataframe.
    Contains target values, index is drug names

    one_hot_drugs - pd dataframe.
    Drug representation of GDSC drugs, one-hot encoded dataframe of SMILES strings
     
    omics_df - pd dataframe.
    Omics data, cell lines represented by index, features are cols 
     
    common_ind - list.
    List of common cell lines between omics and drug dataframes

    Outputs

    x_drug - pd dataframe.
    One-hot encoded representation of all drugs for every cell line

    x_all - pd dataframe.
    Omics data, cell line data repeated for every drug in drug_df
    
    y_series - pd series.
    Target values for all drugs and cell lines
    """

    ## Create Drug-Cell Line paired indexes 

    # get list of cl repeats
    cl_name_repeats = []
    for i in range(len(one_hot_drugs)):
        cl_name_repeats.extend(phospho_df.index.values)

    # get list of drug repeats
    drug_name_repeats = np.repeat(one_hot_drugs.index,len(common_ind))

    # combine with repeated list of drugs
    drug_cl_index = []
    for cl,drug in zip(cl_name_repeats,drug_name_repeats):
        drug_cl_index.append(cl +'::'+ drug)
        
    # duplicate rows and add paired index
    x_drug = pd.DataFrame(np.repeat(one_hot_drugs.values, len(common_ind), axis=0), index=drug_cl_index)

    ## Drug-Cell Line pairing for phosphoproteomics data (x_all)

    # reformat phospho data for drug-cell line pairing index
    # add repeats
    phospho_paired = pd.concat([phospho_df]*len(one_hot_drugs)) 
    # set dataframe with repeated rows to have modified index
    phospho_paired.index = drug_cl_index
    x_all = phospho_paired

    ## Target IC50 series for Drug-Cell Line pairs (y_series)

    # filter to common cls
    drug_series_df = drug_df.loc[drug_df['CELL_LINE_NAME'].isin(common_ind)]
    # set index to cl column
    drug_series_df = drug_series_df.set_index('CELL_LINE_NAME')
    # build new index
    drug_cl_index = drug_series_df.index + '::' + drug_series_df['DRUG_NAME'].to_list() 
    # add new index, drop drug name column and convert to series
    drug_series_df = drug_series_df.set_index(drug_cl_index).drop('DRUG_NAME',axis=1) # add new index to dataframe
    drug_cl_series = drug_series_df['LN_IC50'] # create a series with new index and IC50 values
    drug_cl_series.index.name = None # remove index name
    drug_cl_series = drug_cl_series.loc[~drug_cl_series.index.duplicated()] # remove duplicate targe value entries
    y_series = drug_cl_series

    ## Final adjustments for consistent indexes across dataframes

    # filter down and reorder both one-hot and phospho dataframes based on drug series index 
    x_drug = x_drug.filter(drug_cl_series.index,axis=0)
    x_all = x_all.filter(drug_cl_series.index,axis=0)
    y_series = y_series.filter(x_all.index)

    return x_drug, x_all, y_series