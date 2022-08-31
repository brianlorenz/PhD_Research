# Makes a datafrmae with summarized properties from all the clusters
import initialize_mosdef_dirs as imd
import numpy as np
from astropy.io import ascii
import pandas as pd

def make_clusters_summary_df(n_clusters):
    """Makes a datafrmae with summarized properties from all the clusters
    
    Parameters:
    n_clusters (int): Number of clusters
    """
    
    groupIDs = []
    n_galss = []

    median_zs = []

    median_uvs = []
    median_vjs = []

    median_masses = []
    median_sfrs = []
    median_ssfrs = []

    
    for groupID in range(n_clusters):

        # Read in the df
        group_df = ascii.read(imd.cluster_indiv_dfs_dir + f'/{groupID}_cluster_df.csv')
        group_df['log_use_sfr'] = np.log10(group_df['use_sfr'])
        n_gals = len(group_df)


        # Compute properties
        median_z = np.median(group_df['Z_MOSFIRE'])
    
        median_vj = np.median(group_df[group_df['V_J']>0]['V_J'])
        median_uv = np.median(group_df[group_df['U_V']>0]['U_V'])

        median_mass = np.median(group_df[group_df['log_mass']>0]['log_mass'])
        median_sfr = np.median(group_df[group_df['log_use_sfr']>0]['log_use_sfr'])
        median_ssfr = np.log10((10**median_sfr)/(10**median_mass))

        # Save properties
        groupIDs.append(groupID)
        n_galss.append(n_gals)

        median_zs.append(median_z)

        median_vjs.append(median_vj)
        median_uvs.append(median_uv)

        median_masses.append(median_mass)
        median_sfrs.append(median_sfr)
        median_ssfrs.append(median_ssfr)

    # Build into DataFrame
    clusters_summary_df = pd.DataFrame(zip(groupIDs, n_galss, median_zs, median_masses, median_sfrs, median_ssfrs, median_vjs, median_uvs), columns=['groupID', 'n_gals', 'redshift', 'log_mass', 'log_sfr', 'log_ssfr', 'median_V_J', 'median_U_V'])
    clusters_summary_df.to_csv(imd.loc_cluster_summary_df, index=False)

make_clusters_summary_df(23)