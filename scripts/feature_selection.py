"""Functions for reducing omics dataset features to improve DRP performance"""

import numpy as np
import pandas as pd

def fs_landmark(omics_df):
    """Feature Selection method for omics dataframes where feature names contain gene names. 
    Filters omics dataframe features to only keep features with column names containing landmark genes 
    """

    # create list of landmark genes
    landmark_genes_df = pd.read_csv("datasets/landmark_genes_LINCS.txt",sep='\t')
    landmark_genes = landmark_genes_df['Symbol']
    # find all phosphosites in landmark genes
    landmark_features = []
    for feature in omics_df.columns:
        for gene in landmark_genes:
            if gene in feature:
                landmark_features.append(feature)
    # remove duplicates
    landmark_features = list(dict.fromkeys(landmark_features))

    return landmark_features



def fs_landmark_targets(phos_df):
    """Feature Selection method for phosphoproteomics dataframes.
    Filters omics dataframe features to only keep phosphorylation sites which are targets of Landmark genes
    """

    # load ptm relationships dataset and filter for phosphorylation ptms only
    ptm_df = pd.read_csv('datasets/ptm_relationships.csv',index_col=0)
    ptm_df_phos = ptm_df[ptm_df['modification']=='phosphorylation']
    # load landmark genes list
    landmark_genes_df = pd.read_csv("datasets/landmark_genes_LINCS.txt",sep='\t')
    landmark_genes = landmark_genes_df['Symbol']
    # filter ptms for landmark gene enzymes only
    ptm_df_phos_lm = ptm_df_phos[ptm_df_phos['enzyme_genesymbol'].isin(landmark_genes)]
    # create list of all psites from filtered dataframe
    lm_psites = []
    for ptm in ptm_df_phos_lm.values:
        substrate = ptm[3] 
        residue = ptm[4]
        offset = str(ptm[5])
        psite = f'{substrate}({residue+offset});'
        lm_psites.append(psite)

    # find all phosphosites in landmark genes
    phosphosites = []
    for phosphosite in phos_df.columns:
        for psite in lm_psites:
            if psite in phosphosite:
                phosphosites.append(phosphosite)
    # remove duplicates
    phosphosites = list(dict.fromkeys(phosphosites))

    return phosphosites



def fs_functional_score(phos_df,cutoff=90):
    """Feature Selection method for phosphoproteomics dataframes.
    Filters omics dataframe features to only keep phosphorylation sites above functional score cutoff from phosphosite functional score dataset
    """

    # import phosphosites dataset
    psite_df = pd.read_table('datasets/functional_score_psites.tsv')

    # filter dataframe by specified cutoff
    perc = cutoff
    print('cutoff percentile: ', perc)
    # finding cutoffs based on percentile of all values in col
    ranking_score = psite_df['functional_score'].to_numpy()
    # set cutoff
    cutoff = np.percentile(ranking_score, perc)
    print('functional score cutoff: ', cutoff)
    # filter dataframe for rows with with cutoff
    psite_df_cutoff = psite_df[psite_df['functional_score'] > cutoff] 

    # create list of tuples containg gene and ptm position
    gene_pos_tuples = list(zip(psite_df_cutoff['gene'],psite_df_cutoff['position']))

    ranked_phosphosites = []
    # create list of all functional ranked ptms from dataset in phospho dataframe columns
    for tuple in gene_pos_tuples:
        gene =  tuple[0]
        pos = str(tuple[1])
        # check for any phosphosites that match the conditions
        psites = list(filter(lambda x: gene in x and pos in x,phos_df.columns)) 
        if len(psites) > 0:
            ranked_phosphosites.extend(psites)

    # remove duplicates
    ranked_phosphosites = list(dict.fromkeys(ranked_phosphosites))

    return ranked_phosphosites



def fs_atlas_landmark(phos_df,cutoff=90):
    """Feature Selection method for phosphoproteomics dataframes.
    Filters omics dataframe features to only keep phosphorylation sites above median percentile 
    or promiscuity index cutoff from phosphosite substrate specificities dataset and are targets of landmark genes
    """

    # create list of landmark genes
    landmark_genes_df = pd.read_csv("datasets/landmark_genes_LINCS.txt",sep='\t')
    landmark_genes = landmark_genes_df['Symbol']

    # load substrate specificity dataset
    atlas_df = pd.read_csv('datasets/atlas_ptms.csv')
    # remove empty columns
    for col in atlas_df.columns:
        if 'Unnamed:' in col:
            atlas_df.drop(col, axis=1, inplace=True)

    phos_df_genes = []
    for col in phos_df.columns:
        phos_df_genes.append(col.split('(')[0])
    # remove duplicates
    phos_df_genes = list(dict.fromkeys(phos_df_genes))

    ## uniprot genes mapped to ids for all genes not in substrate specificity dataset

    uniprot_mapping_df = pd.read_csv("datasets/uniprot.tsv",sep='\t')
    swissprot_mapping_df = pd.read_csv("datasets/uniprot_swissprot.tsv",sep='\t')
    uniprot_all_mapping = pd.concat([uniprot_mapping_df, swissprot_mapping_df], axis=0).reset_index()

    # create dictionary
    gene_id_dict = {}
    for ind in uniprot_all_mapping.index:
        gene = uniprot_all_mapping['From'][ind]
        id = uniprot_all_mapping['Entry'][ind]
        gene_id_dict[id] = gene

    # find which genes are in substrate specificity dataset  using uniprot id
    extra_ids = []
    for id in gene_id_dict:
        if id in atlas_df['Database Uniprot Accession'].to_list():
            extra_ids.append(id)

    # filter dataframe for rows using genes appearing either in 'Gene', 'Alternative Gene Names' 
    # or 'Protein' column of substrate specificity dataset
    atlas_df_filtered_gene = atlas_df[(atlas_df['Gene'].isin(phos_df_genes)) | 
                                                (atlas_df['Protein'].isin(phos_df_genes)) | 
                                                (atlas_df['Alternative Gene Names'].isin(phos_df_genes))]
    # filter dataframe for rows with additional genes using uniprot ids
    atlas_df_filtered_id = atlas_df[atlas_df['Database Uniprot Accession'].isin(extra_ids)]
    # combine into single dataframe
    atlas_df_filtered = pd.concat([atlas_df_filtered_gene, atlas_df_filtered_id], axis=0).reset_index()

    # create separate lists for enzyme columns and other columns
    enzyme_colnames = []
    other_colnames = []
    for col in atlas_df_filtered.columns:
        if 'rank' in col or 'percentile' in col and col != 'median_percentile':
            enzyme_colnames.append(col)
        else:
            other_colnames.append(col)
    # extract enzyme names from colnames and remove duplicates
    enzyme_list = []
    for col in enzyme_colnames:
        enz = col.split('_')[0]
        enzyme_list.append(enz)
    enzyme_list = list(dict.fromkeys(enzyme_list))
    # find enzymes which are also landmark genes
    enzyme_list_lm = list(set(enzyme_list).intersection(landmark_genes.to_list()))
    # find colnames with landmark gene enzymes
    enzyme_colnames_LM = []
    for col in enzyme_colnames:
        for enz in enzyme_list_lm:
            if enz in col:
                enzyme_colnames_LM.append(col)

    # combine lists
    other_colnames.extend(enzyme_colnames_LM)
    LM_colnames = other_colnames

    # filter dataframe to get data for LM enzymes only
    atlas_LM = atlas_df_filtered[LM_colnames]

    # select for percentile columns except for median percentile column
    percentile_vals = atlas_LM.filter(regex='percentile').iloc[: , 1:].values 
    from statistics import median
    # iterate through list of arrays, calculate new median and add to list
    median_percentile_list = []
    for percentiles in percentile_vals:
        median_percentile_list.append(median(percentiles))
    # iterate through list of arrays, calculate number of kinases scoring above 90th percentile and add to list
    promiscuity_index_list = []
    for percentiles in percentile_vals:
        promiscuity_index_list.append((percentiles > 90).sum())
    # replace median percentile and promiscuity index columns with new values
    atlas_LM = atlas_LM.assign(median_percentile=pd.Series(median_percentile_list), promiscuity_index=pd.Series(promiscuity_index_list))

    # set cutoff percentile 0-100
    cutoff_perc = cutoff
    # set cutoff type to promiscuity index or median percentile
    cutoff_type = 'median_perc' # set to 'prom_index' or 'median_perc'

    print('cutoff percentile: ', cutoff_perc)

    if cutoff_type == 'prom_index': # using promiscuity index cutoff
        # calculate cutoff
        ranking_prom_index = atlas_LM['promiscuity_index'].to_numpy()
        print('promiscuity index cutoff: ', np.percentile(ranking_prom_index, cutoff_perc))
        # filter dataframe
        cutoff = np.percentile(ranking_prom_index, cutoff_perc) # change number to cutoff percentile
        atlas_LM_cutoff = atlas_LM[atlas_LM['promiscuity_index'] > cutoff]

    elif cutoff_type == 'median_perc': # using median percentile cutoff
        # calculate cutoff
        ranking_median_perc = atlas_LM['median_percentile'].to_numpy()
        print('median percentile cutoff: ', np.percentile(ranking_median_perc, cutoff_perc))
        # filter dataframe
        cutoff = np.percentile(ranking_median_perc, cutoff_perc) # change number to cutoff percentile
        atlas_LM_cutoff = atlas_LM[atlas_LM['median_percentile'] > cutoff] 

    # create list of gene-phosphosite pairs within substrate specificity dataset filtered by median cutoff
    formatted_phosphosites = []
    for ind in atlas_LM_cutoff.index:
        gene = atlas_LM_cutoff.loc[ind]['Gene']
        phosphosite = atlas_LM_cutoff.loc[ind]['Phosphosite']
        formatted_phosphosites.append(f'{gene}({phosphosite});')

    # find all phos_df features containing cutoff phosphosites
    phosphosites = []
    for psite in phos_df.columns:
        for cutoff_psite in formatted_phosphosites:
            if cutoff_psite in psite:
                phosphosites.append(psite)
    # remove duplicates
    phosphosites = list(dict.fromkeys(phosphosites))

    return phosphosites




def true_phosphosite_filter(phosphosites):
    """Takes a list of phosphorylation sites and removes false positive phosphosites 
    based on dataset filtered to only contain true positive phosphosites
    """


    # import dataset for phosphosite quality control
    data_path = "datasets/filtered_psites.csv"
    df = pd.read_csv(data_path,header=2,low_memory=False)
    true_psites_vals = df[['GENE','Site','MOD_RD']].values

    # create list of true phosphosites formatted for feature selection
    true_psites = []
    for psite in true_psites_vals:
        gene = psite[0]
        residue = psite[1]
        offset = psite[2]
        # add to list
        true_psites.append(f'{gene}({residue}{offset});')
    
    # check if phosphosites from feature select contains true phosphosites which will be added to new list
    filtered_phosphosites = []
    for psite in phosphosites:
        res = any(ele in psite for ele in true_psites)
        if res == True:
            filtered_phosphosites.append(psite)
    
    return filtered_phosphosites