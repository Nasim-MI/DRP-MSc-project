import pandas as pd
import pubchempy as pcp

def request_SMILES(CIDs,drug_names):
    """Takes a list of Drug PubChem IDs and names and requests canonical SMILES strings via pubchempy. 
    Tries CID first, then name. If both CID and name fails, prints drug name and index of list
    Returns list of canonical SMILES and drug names"""


    canonical_smiles_list = []
    canonical_smiles_name = []
    error_inds = []
    
    for ind, (entry,name) in enumerate(zip(CIDs,drug_names)):
        # try cid, if request fails, attempt name
        try:
            c = pcp.Compound.from_cid(entry)
            canonical_smiles_list.append(c.canonical_smiles)
            canonical_smiles_name.append(name)
        except:
            # try name, if request fails, print offending name and index
            try:
                c = pcp.get_compounds(name, 'name')
                canonical_smiles_list.append(c[0].canonical_smiles)
                canonical_smiles_name.append(name)
            except:
                if len(error_inds) == 0:
                    print("Failed to retrieved canonical SMILES for:")
                    print(ind, name)
                    error_inds.append(ind)
                else:
                    print(ind, name)
                    error_inds.append(ind)
    


    return canonical_smiles_list, canonical_smiles_name

input_data_path = 'datasets/GDSC_drug.tsv'
output_data_path = 'datasets/GDSC_canonical_SMILES.tsv'

# import and extract GDSC drug names
gdsc_drugs = pd.read_table(input_data_path)
# if a PubChem CID is in the name column, copy to CIDS column
col_index = gdsc_drugs.columns.get_loc(' PubCHEM') # column index of PubChem CIDs
for index, row in gdsc_drugs.iterrows():
    if row[' Name'].isnumeric() == True:
        gdsc_drugs.iloc[index,col_index] = row[' Name']
        
drug_name_id = gdsc_drugs.iloc[:,[1,5]].drop_duplicates()

# convert dataframe into list of drug names and CIDs
CIDs = drug_name_id[' PubCHEM'].to_list()
drug_names = drug_name_id[' Name'].to_list()
print(f'Attempting to request canonical SMILES strings for {len(drug_name_id)} drug compounds')
# request canonical SMILES strings from PubChem via pubchempy
canonical_smiles_list, canonical_smiles_name = request_SMILES(CIDs,drug_names)

# create dataframe with SMILES as data and drug names as index
data = {'DRUG_NAME': canonical_smiles_name, 
        'SMILES': canonical_smiles_list} 

canonical_SMILES_df = pd.DataFrame.from_dict(data)
canonical_SMILES_df = canonical_SMILES_df[~canonical_SMILES_df.index.duplicated(keep='first')]
canonical_SMILES_df.to_csv(output_data_path,sep='\t')

print(f'Successfully retrieved canonical SMILES for {len(canonical_smiles_name)}')
print(f'Canonical SMILES saved to {output_data_path}')
