# Code to generate clusters from scratch

import initialize_mosdef_dirs as imd
from query_funcs import get_zobjs
from axis_ratio_funcs import read_interp_axis_ratio
from emission_measurements import read_emission_df
from interpolate import gen_all_seds
from cross_correlate import correlate_all_seds
from clustering import cluster_seds
from astropy.io import ascii
from spectra_funcs import check_quick_coverage
from mosdef_obj_data_funcs import get_mosdef_obj
import numpy as np
import matplotlib.pyplot as plt
import time

def generate_clusters(n_clusters, stop_to_eval=True, skip_slow_steps=False):
    """Main method that will generate all the clusters from scratch
    
    Parameters: 
    stop_to_eval(boolean): Set to True to pause and make a plot of eigenvalues
    skip_slow_steps(boolean): Set to True to skip over making mock seds and cross_correlating
    
    """
    print('Are you sure you want to clear the cluster directory? c for yes, exit() for no')
    breakpoint()
    # Prepare the directories
    imd.reset_cluster_dirs(imd.cluster_dir)
    imd.reset_sed_dirs(imd.mosdef_dir)

    # Prepare the galaxies dataframes
    filter_gal_df()
    gal_df = read_filtered_gal_df()
    zobjs = [(gal_df['field'].iloc[i], gal_df['v4id'].iloc[i])
             for i in range(len(gal_df))]
    
    # Generate the mock sed csvs
    if skip_slow_steps==False:
        gen_all_seds(zobjs)

    # Cross-correlate the mock seds
    if skip_slow_steps==False:
        pass
        correlate_all_seds(zobjs)

    affinity_matrix = ascii.read(imd.cluster_dir + '/similarity_matrix.csv').to_pandas().to_numpy()

    # Evaluate how many clusters to make
    if stop_to_eval==True:
        eigenvals, eignvectors = np.linalg.eig(affinity_matrix)
        x_axis = np.arange(1, len(eigenvals)+1, 1)
        plt.plot(x_axis, eigenvals, ls='-', marker='o', color='black')
        plt.show()
    
    # Make the clusters
    if skip_slow_steps==False:
        pass
        cluster_seds(n_clusters)

    # Make dataframes for each cluster with galaxy properties
    make_cluster_dfs(n_clusters, gal_df)



    
def plot_eigenvalues():
    fig, ax = plt.subplots(figsize=(8,8))
    affinity_matrix = ascii.read(imd.cluster_dir + '/similarity_matrix.csv').to_pandas().to_numpy()
    eigenvals, eignvectors = np.linalg.eig(affinity_matrix)
    x_axis = np.arange(1, len(eigenvals)+1, 1)
    ax.plot(x_axis, eigenvals, ls='-', marker='o', color='black')
    ax.set_yscale('log')
    ax.set_xlim(0, 50)
    ax.set_ylim(0.1, 1000)
    fig.savefig(imd.cluster_dir+'/paper_figures/eigenvalues.pdf')
    

def make_cluster_dfs(n_clusters, gal_df):
    """Make dataframes that contain properties for the individual galaxies in each cluster
    
    Parameters:
    n_clusters(int): Number of clusters
    gal_df (pd.DataFrame): Dataframe of galaxies that were used for clustering

    """
    imd.check_and_make_dir(imd.cluster_indiv_dfs_dir)
    zobjs_clusters = ascii.read(imd.cluster_dir + '/zobjs_clustered.csv').to_pandas()

    for group_num in range(n_clusters):
        group_members_df = zobjs_clusters[zobjs_clusters['cluster_num']==group_num]
        group_members_df = group_members_df.drop(['original_zobjs_index'], axis=1)
        group_gal_df = group_members_df.merge(gal_df, on=['field', 'v4id'])
        group_gal_df['group_gal_id'] = np.arange(len(group_gal_df))
        group_gal_df.to_csv(imd.cluster_indiv_dfs_dir + f'/{group_num}_cluster_df.csv', index=False)


def filter_gal_df():
    """Brings together all data sources into a single dataframe for easy access"""
    gal_df = read_interp_axis_ratio()

    full_df = read_interp_axis_ratio()

    len_before_agn = len(gal_df)
    #Filter out objects that are flagged as AGN in MOSDEF (see readme)
    agn_zero = gal_df['agn_flag'] == 0
    agn_six = gal_df['agn_flag'] == 6
    agn_good = np.logical_or(agn_zero, agn_six)
    gal_df = gal_df[agn_good] # Used to be < 4
    len_after_agn = len(gal_df)
    print(f'removed {len_before_agn-len_after_agn} galaxies for AGN flag')

    len_before_zfilt = len(gal_df)
    #Filter out objects that don't have a spectroscopic redshift
    gal_df = gal_df[gal_df['z_qual_flag'] == 7]
    len_after_zfilt = len(gal_df)
    print(f'removed {len_before_zfilt-len_after_zfilt} galaxies for bad redshift')

    len_before_serendip = len(gal_df)
    #Filter out serendipds
    gal_df = gal_df[gal_df['serendip_flag'] == 1]
    len_after_serendip = len(gal_df)
    print(f'removed {len_before_serendip-len_after_serendip} galaxies for serendips')

    len_before_id_dup = len(gal_df)
    #Filter out serendipds
    good_vals = gal_df['v4id'].drop_duplicates().index
    gal_df = gal_df.filter(items = good_vals, axis=0)
    len_after_id_dup= len(gal_df)
    print(f'removed {len_before_id_dup-len_after_id_dup} galaxies for duplicates')

    coverage_list = [
            ('Halpha', 6564.61),
            ('Hbeta', 4862.68),
        ]
    # 
    lines_covereds = []
    for i in range(len(gal_df)):
        mosdef_obj = get_mosdef_obj(gal_df.iloc[i]['field'], gal_df.iloc[i]['v4id'])
        lines_covered = check_quick_coverage(mosdef_obj, coverage_list, verb=False)
        lines_covereds.append(lines_covered)
    gal_df['ha_hb_covered'] = lines_covereds
    len_before_hahb_coverage = len(gal_df)
    #Filter out serendipds
    gal_df = gal_df[gal_df['ha_hb_covered'] == 1]
    len_after_hahb_coverage = len(gal_df)
    print(f'removed {len_before_hahb_coverage-len_after_hahb_coverage} galaxies for halpha or hbeta not covered')

    print(f'{len(gal_df)} galaxies remain')

    print('Save updated filtered gals and removed gals? c to continue')
    breakpoint()

    # Save removed galaxies 
    removed_gals = full_df.drop(gal_df.index)
    removed_gals.to_csv(imd.loc_removed_gal_df, index=False)

    gal_df.to_csv(imd.loc_filtered_gal_df, index=False)

def read_filtered_gal_df():
    gal_df = ascii.read(imd.loc_filtered_gal_df).to_pandas()
    return gal_df

def read_removed_gal_df():
    gal_df = ascii.read(imd.loc_removed_gal_df).to_pandas()
    return gal_df

# filter_gal_df()
# plot_eigenvalues()
# gal_df = read_filtered_gal_df()
# print(len(gal_df))
# generate_clusters(20, stop_to_eval=False, skip_slow_steps=True)