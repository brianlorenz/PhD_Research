# Counds the number of AGN in each cluster 

import numpy as np
import pandas as pd
import initialize_mosdef_dirs as imd
from read_data import mosdef_df
from cluster_data_funcs import get_cluster_fields_ids



def check_for_all_agn(n_clusters):
    """Counts the number of agn in every group, and outputs this to a text file

    Parameters: 
    n_clusters (int): Number of clusters
    """
    groupIDs = range(n_clusters)
    num_agn_in_group = []
    n_gals_in_group = []
    percent_agn_in_group = []
    # Loop over all groups
    for groupID in groupIDs:
        # For each group, loop over all galaxies in the group
        num_agn = 0
        cluster_names, fields_ids = get_cluster_fields_ids(groupID)
        n_gals = len(fields_ids)
        for zobj in fields_ids:
            # Search the AGNFLAG variable, and add to the count if it is a 1
            field = zobj[0]
            v4id = int(zobj[1])
            if mosdef_df[np.logical_and(mosdef_df['FIELD_STR']==field, mosdef_df['V4ID']==v4id)].iloc[0]['AGNFLAG'] == 1:
                num_agn += 1
        # Compute what percentage of the group is agn
        percent_agn = num_agn / n_gals
        # Save the result to this dataframe
        num_agn_in_group.append(num_agn)
        n_gals_in_group.append(n_gals)
        percent_agn_in_group.append(np.round(percent_agn, 3))
    
    # Save the dataframe into the cluster_dir
    agn_count_df = pd.DataFrame(zip(groupIDs, num_agn_in_group, n_gals_in_group, percent_agn_in_group), columns=['groupID', 'n_agn', 'n_gals', 'percent_agn'])
    agn_count_df.to_csv(imd.cluster_dir + '/number_agn.csv', index=False)
    tot_agn = np.sum(agn_count_df['n_agn'])
    print(f'Total number of AGN: {tot_agn}')


check_for_all_agn(29)