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

    # load substrate specificity dataset filtered for substrate specificities of landmark genes only
    atlas_LM_df = pd.read_csv('datasets/atlas_LM_ptms.csv')
    # remove empty columns
    for col in atlas_LM_df.columns:
        if 'Unnamed:' in col:
            atlas_LM_df.drop(col, axis=1, inplace=True)

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
        if id in atlas_LM_df['Database Uniprot Accession'].to_list():
            extra_ids.append(id)

    # filter dataframe for rows using genes appearing either in 'Gene', 'Alternative Gene Names' 
    # or 'Protein' column of substrate specificity dataset
    atlas_LM_filtered_gene = atlas_LM_df[(atlas_LM_df['Gene'].isin(phos_df_genes)) | 
                                                (atlas_LM_df['Protein'].isin(phos_df_genes)) | 
                                                (atlas_LM_df['Alternative Gene Names'].isin(phos_df_genes))]
    # filter dataframe for rows with additional genes using uniprot ids
    atlas_LM_filtered_id = atlas_LM_df[atlas_LM_df['Database Uniprot Accession'].isin(extra_ids)]
    # combine into single dataframe
    atlas_LM_filtered = pd.concat([atlas_LM_filtered_gene, atlas_LM_filtered_id], axis=0).reset_index()



    # set cutoff percentile 0-100
    cutoff_perc = cutoff
    # set cutoff type to promiscuity index or median percentile
    cutoff_type = 'median_perc' # set to 'prom_index' or 'median_perc'

    print('cutoff percentile: ', cutoff_perc)

    if cutoff_type == 'prom_index': # using promiscuity index cutoff
        # calculate cutoff
        ranking_prom_index = atlas_LM_filtered['promiscuity_index'].to_numpy()
        print('promiscuity index cutoff: ', np.percentile(ranking_prom_index, cutoff_perc))
        # filter dataframe
        cutoff = np.percentile(ranking_prom_index, cutoff_perc) # change number to cutoff percentile
        atlas_LM_cutoff = atlas_LM_filtered[atlas_LM_filtered['promiscuity_index'] > cutoff]

    elif cutoff_type == 'median_perc': # using median percentile cutoff
        # calculate cutoff
        ranking_median_perc = atlas_LM_filtered['median_percentile'].to_numpy()
        print('median percentile cutoff: ', np.percentile(ranking_median_perc, cutoff_perc))
        # filter dataframe
        cutoff = np.percentile(ranking_median_perc, cutoff_perc) # change number to cutoff percentile
        atlas_LM_cutoff = atlas_LM_filtered[atlas_LM_filtered['median_percentile'] > cutoff] 

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