"""Functions for importing and wrangling phosphoproteomics, proteomics and GDSC1 drug datasets used in MSc project. 
Default output is the mean of replicate values for each cell line in omics datasets.
Import functions have parameter options for removing outliers outside of an upper and lower limit set by wrangling function.
Import functions also have option for outputing data from a single replicate or all replicates.
"""

import numpy as np
import pandas as pd

def import_phos(outlier_handling=False,multiplier=1,replicate=None):
    """Imports and wrangles phosphoproteomics and GDSC1 drug datasets, outputs mean of all replicates for phosphoproteomics data by default.
    Setting outlier_handling -> True: Outliers for each set of cell line triplicates outside of an upper or lower limit (set by multiplier) are ignored when calculating mean.
    Multiplier: if outlier_handling == True, determines upper and lower limit, upper_limit = Q3 + multiplier * IQR, lower_limit = Q1 + multiplier * IQR.
    Replicate: if replicate == 1, 2 or 3, function only outputs data from specific replicate, if replicate == "All", function outputs all replicates.
    """

    drug_data_path = "datasets/drug_data.tsv"
    phos_data_path = "datasets/phospho_data.tsv"

    drug_df = pd.read_csv(drug_data_path,sep='\t')
    phos_df = pd.read_csv(phos_data_path,sep='\t',index_col='col.name').T

    # wrangle dataframes and create lists of all common cell lines and drugs
    if outlier_handling == True:
        drug_matrix,phos_df,_all_cls,_all_drugs,common_ind = wrangle_outliers(drug_df,phos_df,multiplier)
    elif replicate != None:
        if replicate == 1 or replicate == 2 or replicate == 3:
            drug_matrix,phos_df,_all_cls,_all_drugs,common_ind = wrangle_single(drug_df,phos_df,replicate)
        elif replicate == "All":
            drug_matrix,phos_df,_all_cls,_all_drugs,common_ind = wrangle_replicates(drug_df,phos_df)
        else:
            print("Replicate must be an integer value of 1,2 or 3")
    else:
        drug_matrix,phos_df,_all_cls,_all_drugs,common_ind = wrangle(drug_df,phos_df)
    
    
    print('Data Imports and Wrangling: Done')

    return phos_df,drug_df,drug_matrix,_all_cls,_all_drugs,common_ind 



def import_prot(outlier_handling=False,multiplier=1,replicate=None):
    """Imports and wrangles proteomics and GDSC1 drug datasets, outputs mean of all replicates for proteomics data by default.
    Setting outlier_handling -> True: Outliers for each set of cell line triplicates outside of an upper or lower limit (set by multiplier) are ignored when calculating mean.
    Multiplier: if outlier_handling == True, determines upper and lower limit, upper_limit = Q3 + multiplier * IQR, lower_limit = Q1 + multiplier * IQR.
    Replicate: if replicate == 1, 2 or 3, function only outputs data from specific replicate, if replicate == "All", function outputs all replicates.
    """
    drug_data_path = "datasets/drug_data.tsv"
    prot_data_path = "datasets/prot_data.tsv"

    drug_df = pd.read_csv(drug_data_path,sep='\t')
    prot_df = pd.read_csv(prot_data_path,sep='\t',index_col=0).T

    # wrangle dataframes and create lists of all common cell lines and drugs
    if outlier_handling == True:
        drug_matrix,prot_df,_all_cls,_all_drugs,common_ind = wrangle_outliers(drug_df,prot_df,multiplier)
    elif replicate != None:
        if replicate == 1 or replicate == 2 or replicate == 3:
            drug_matrix,prot_df,_all_cls,_all_drugs,common_ind = wrangle_single(drug_df,prot_df,replicate)
        elif replicate == "All":
            drug_matrix,phos_df,_all_cls,_all_drugs,common_ind = wrangle_replicates(drug_df,prot_df)
        else:
            print("Replicate must be an integer value of 1,2 or 3")
    else:
        drug_matrix,prot_df,_all_cls,_all_drugs,common_ind = wrangle(drug_df,prot_df)

    print('Data Imports and Wrangling: Done')

    return prot_df,drug_df,drug_matrix,_all_cls,_all_drugs,common_ind 



def import_all(outlier_handling=False,multiplier=1,replicate=None):
    """Imports and wrangles phosphoproteomics, proteomics and GDSC1 drug datasets, outputs mean of all replicates for phosphoproteomics and proteomics data by default.
    Setting outlier_handling -> True: Outliers for each set of cell line triplicates outside of an upper or lower limit (set by multiplier) are ignored when calculating mean.
    Multiplier: if outlier_handling == True, determines upper and lower limit, upper_limit = Q3 + multiplier * IQR, lower_limit = Q1 + multiplier * IQR.
    Replicate: if replicate == 1, 2 or 3, function only outputs data from specific replicate, if replicate == "All", function outputs all replicates.
    """
    drug_data_path = "datasets/drug_data.tsv"
    prot_data_path = "datasets/prot_data.tsv"
    phos_data_path = "datasets/phospho_data.tsv"

    drug_df = pd.read_csv(drug_data_path,sep='\t')
    phos_df = pd.read_csv(phos_data_path,sep='\t',index_col='col.name').T
    prot_df = pd.read_csv(prot_data_path,sep='\t',index_col=0).T

    # wrangle dataframes and create lists of all common cell lines and drugs for proteomics dataset
    if outlier_handling == True:
        drug_matrix,phos_df,_all_cls,_all_drugs,common_ind = wrangle_outliers(drug_df,phos_df,multiplier)
    elif replicate != None:
        if replicate == 1 or replicate == 2 or replicate == 3:
            drug_matrix,phos_df,_all_cls,_all_drugs,common_ind = wrangle_single(drug_df,phos_df,replicate)
        elif replicate == "All":
            drug_matrix,phos_df,_all_cls,_all_drugs,common_ind = wrangle_replicates(drug_df,phos_df)    
        else:
            print("Replicate must be an integer value of 1,2 or 3")
    else:
        drug_matrix,phos_df,_all_cls,_all_drugs,common_ind = wrangle(drug_df,phos_df)

    # wrangle dataframes and create lists of all common cell lines and drugs for phosphoproteomics dataset
    if outlier_handling == True:
        drug_matrix,prot_df,_all_cls,_all_drugs,common_ind = wrangle_outliers(drug_df,prot_df,multiplier)
    elif replicate != None:
        if replicate == 1 or replicate == 2 or replicate == 3:
            drug_matrix,prot_df,_all_cls,_all_drugs,common_ind = wrangle_single(drug_df,prot_df,replicate)
        elif replicate == "All":
            drug_matrix,phos_df,_all_cls,_all_drugs,common_ind = wrangle_replicates(drug_df,phos_df)  
        else:
            print("Replicate must be an integer value of 1,2 or 3")
    else:
        drug_matrix,prot_df,_all_cls,_all_drugs,common_ind = wrangle(drug_df,prot_df)

    print('Data Imports and Wrangling: Done')

    return phos_df,prot_df,drug_df,drug_matrix,_all_cls,_all_drugs,common_ind 



# new wrangle function, triplicate values merged into single row for each cell line using mean of triplicate
def wrangle(drug_df,omics_df):

    '''Wrangles drug and phosphoproteomics dataframes. Phospho dataframe also has triplicate rows merged into a row with mean values for each cell line.'''
    
    # filter down drug dataframe to relevant columns
    drug_df = drug_df[['CELL_LINE_NAME','DRUG_NAME','LN_IC50']]
    # convert table to matrix data structure
    drug_matrix = drug_df.groupby(['DRUG_NAME','CELL_LINE_NAME']).sum()['LN_IC50'].unstack().reset_index()
    drug_matrix = drug_matrix.rename_axis(None, axis=1).set_index('DRUG_NAME').T

    # take cell line name from first index name for every trio of index names
    n = 3
    unique_cls = []
    for cl in omics_df.index:
        if n % 3 == 0:
            unique_cls.append(cl)
        n+=1

    # create new dataframe for triplicate means
    omics_df_means = pd.DataFrame(columns=omics_df.columns)

    # fill dataframe with triplicate means for all cell lines
    for cl in unique_cls:
        # find rows with cell line name
        cl_rows = omics_df.filter(regex=cl,axis=0)
        # create row from mean of cell line rows
        omics_df_means.loc[cl] = cl_rows.mean()

    # reformat phospho dataframe for consistent cell line indices with drug dataframe
    omics_df = omics_df_means.set_index(omics_df_means.index.str.replace('.','-', regex=True))

    # create a list of all x cell lines overlapping both datasets and filter
    common_ind = list(set(omics_df.index).intersection(set(drug_matrix.index)))
    drug_matrix = drug_matrix.filter(common_ind,axis=0)
    omics_df = omics_df.filter(common_ind,axis=0)
    # extract list of cell lines and drugs
    _all_cls = omics_df.index
    _all_drugs = drug_matrix.columns 

    return (drug_matrix,omics_df,_all_cls,_all_drugs,common_ind)



def wrangle_outliers(drug_df,omics_df,multiplier):

    '''Wrangles drug and phosphoproteomics dataframes. Phospho dataframe also has triplicate rows merged into a row with mean values for each cell line with outliers removed based on IQR.'''
    
    # filter down drug dataframe to relevant columns
    drug_df = drug_df[['CELL_LINE_NAME','DRUG_NAME','LN_IC50']]
    # convert table to matrix data structure
    drug_matrix = drug_df.groupby(['DRUG_NAME','CELL_LINE_NAME']).sum()['LN_IC50'].unstack().reset_index()
    drug_matrix = drug_matrix.rename_axis(None, axis=1).set_index('DRUG_NAME').T

    # take cell line name from first index name for every trio of index names
    n = 3
    unique_cls = []
    for cl in omics_df.index:
        if n % 3 == 0:
            unique_cls.append(cl)
        n+=1

    # create new dataframe for triplicate means
    omics_df_means = pd.DataFrame(columns=omics_df.columns)

    # fill dataframe with triplicate means for all cell lines
    for cl in unique_cls:
        # find rows with cell line name
        cl_rows = omics_df.filter(regex=cl,axis=0)
        # find 25% and 75% quartiles
        quartile1 = cl_rows.quantile(0.25)
        quartile3 = cl_rows.quantile(0.75)
        # calculate interquartile range
        iqr = quartile3 - quartile1
        # calculate upper and lower limits
        upper_limit = quartile3 + multiplier * iqr
        lower_limit = quartile1 - multiplier * iqr
        # mask values outside of bounds with np.nan
        cl_rows_masked = cl_rows.mask((cl_rows > upper_limit) | (cl_rows < lower_limit))
        # create row from mean of cell line rows
        omics_df_means.loc[cl] = cl_rows_masked.mean()

    # reformat phospho dataframe for consistent cell line indices with drug dataframe
    omics_df = omics_df_means.set_index(omics_df_means.index.str.replace('.','-', regex=True))

    # create a list of all x cell lines overlapping both datasets and filter
    common_ind = list(set(omics_df.index).intersection(set(drug_matrix.index)))
    drug_matrix = drug_matrix.filter(common_ind,axis=0)
    omics_df = omics_df.filter(common_ind,axis=0)
    # extract list of cell lines and drugs
    _all_cls = omics_df.index
    _all_drugs = drug_matrix.columns 

    return (drug_matrix,omics_df,_all_cls,_all_drugs,common_ind)



def wrangle_replicates(drug_df,omics_df):

    '''Wrangles drug and phosphoproteomics dataframes. Includes all phospho cell line replicates'''
    
    # filter down drug dataframe to relevant columns
    drug_df = drug_df[['CELL_LINE_NAME','DRUG_NAME','LN_IC50']]
    # convert table to matrix data structure
    drug_matrix = drug_df.groupby(['DRUG_NAME','CELL_LINE_NAME']).sum()['LN_IC50'].unstack().reset_index()
    drug_matrix = drug_matrix.rename_axis(None, axis=1).set_index('DRUG_NAME').T

    # take cell line name from first index name for every trio of index names
    n = 3
    unique_cls = []
    for cl in omics_df.index:
        if n % 3 == 0:
            unique_cls.append(cl)
        n+=1

    # create new dataframe for triplicate means
    omics_df_means = pd.DataFrame(columns=omics_df.columns)

    # fill dataframe with triplicate means for all cell lines
    for cl in unique_cls:
        # find rows with cell line name
        cl_rows = omics_df.filter(regex=cl,axis=0)
        # create row from mean of cell line rows
        omics_df_means.loc[cl] = cl_rows.mean()

    # reformat phospho dataframe for consistent cell line indices with drug dataframe
    omics_df = omics_df.set_index(omics_df.index.str.replace('.','-', regex=True))

    # duplicate each row to 3x to match phospho data replicates
    series_list = []
    for ind in drug_matrix.index:
        # index row of specific cell line, create duplicates and rename 
        row_series = drug_matrix.loc[ind]
        row_series_1 = row_series.rename(f'{ind}-1')
        row_series_2 = row_series.rename(f'{ind}-2')
        # add row and duplicates to list of series
        series_list.extend([row_series,row_series_1,row_series_2])

    # convert list of series into dataframe
    drug_matrix_corrected = pd.DataFrame(series_list)

    # create a list of all x cell lines overlapping both datasets and filter
    common_ind = list(set(omics_df.index).intersection(set(drug_matrix_corrected.index)))
    drug_matrix = drug_matrix_corrected.filter(common_ind,axis=0)
    omics_df = omics_df.filter(common_ind,axis=0)
    # extract list of cell lines and drugs
    _all_cls = omics_df.index
    _all_drugs = drug_matrix.columns 

    return (drug_matrix,omics_df,_all_cls,_all_drugs,common_ind)



def wrangle_single(drug_df,omics_df,replicate=1):
    '''Wrangles drug and phosphoproteomics dataframes. Outputs only one replicate from each set of samples'''
    
    # set replicate samples to be extracted from dataset
    if replicate == 1:
        replicate_num = ''
    elif replicate == 2:
        replicate_num = '-1'
    elif replicate == 3:
        replicate_num = '-2'
        
    # filter down drug dataframe to relevant columns
    drug_df = drug_df[['CELL_LINE_NAME','DRUG_NAME','LN_IC50']]
    # convert table to matrix data structure
    drug_matrix = drug_df.groupby(['DRUG_NAME','CELL_LINE_NAME']).sum()['LN_IC50'].unstack().reset_index()
    drug_matrix = drug_matrix.rename_axis(None, axis=1).set_index('DRUG_NAME').T

    # reformat phospho dataframe for consistent cell line indices with drug dataframe
    omics_df = omics_df.set_index(omics_df.index.str.replace('.','-', regex=True))

    # create a list of all x cell lines overlapping both datasets and filter
    common_ind = list(set(omics_df.index).intersection(set(drug_matrix.index)))
    drug_matrix = drug_matrix.filter(common_ind,axis=0)
    # create additional index to target specific replicates
    common_ind_replicate = [ind + replicate_num for ind in common_ind] # append ending tag to all cls
    omics_df = omics_df.filter(common_ind_replicate,axis=0) # filter for all samples of specific replicate
    omics_df.index = common_ind # remove replicate indicator to only give cell line name for samples
    # extract list of cell lines and drugs
    _all_cls = omics_df.index
    _all_drugs = drug_matrix.columns 

    return (drug_matrix,omics_df,_all_cls,_all_drugs,common_ind)



def import_SMILES(drug_matrix):
    """Imports and wrangles drug names and SMILES strings for GDSC drugs in phosphoproteomics and proteomics datasets
    """
    # import GDSC drug SMILES strings
    SMILES_df = pd.read_table('datasets/GDSC_canonical_SMILES.tsv',index_col=0,header=0)
    # filter SMILES dataframe to only drugs with GDSC data from cell lines in omics dataset
    SMILES_df = SMILES_df.set_index('DRUG_NAME')
    SMILES_df.index.name = None
    # remove duplicates
    SMILES_df = SMILES_df[~SMILES_df.index.duplicated(keep='first')]
    # filter SMILES dataframe to only drugs with GDSC data from cell lines in omics dataset
    SMILES_df = SMILES_df.filter(items = drug_matrix.columns , axis=0)

    return SMILES_df