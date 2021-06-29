'''Runs all of the plotting codes for clusters'''

import initialize_mosdef_dirs as imd
from astropy.io import ascii
from bpt_clusters import plot_bpt_cluster
from emission_measurements import read_emission_df
from plot_mass_sfr import plot_mass_sfr_cluster, read_sfr_df, get_all_sfrs_masses
from uvj_clusters import plot_full_uvj, plot_uvj_cluster
from cluster_stats import plot_similarity_cluster

def generate_cluster_plots(groupID, emission_df, all_sfrs_res, zobjs, similarity_matrix):
    '''
    Generates all plots for the cluster with the given groupID
    
    Parameters:
    groupID (int): the groupID of the cluster
    emission_df (pd.DataFrame): Use read_emission_df
    all_sfrs_res (pd.DataFrame): Use read_sfr_df and get_all_ssfrs_masses
    zobjs (list of tuples): see get_zobjs()
    similarity_matrix (array): Matrix containing the similarities between all clusters

    Returns:
    '''
    plot_similarity_cluster(groupID, zobjs, similarity_matrix)
    plot_bpt_cluster(emission_df, groupID)
    plot_mass_sfr_cluster(groupID, all_sfrs_res)
    plot_uvj_cluster(groupID)
    
    


def generate_all_cluster_plots(n_clusters):
    '''
    Generates all plots for all the clusters
    
    Parameters:
    n_clusters (int): number of clusters

    Returns:
    '''
    emission_df = read_emission_df()

    sfr_df = read_sfr_df()
    all_sfrs_res = get_all_sfrs_masses(sfr_df)

    similarity_matrix = ascii.read(
        imd.cluster_dir + '/similarity_matrix.csv').to_pandas().to_numpy()
    zobjs = ascii.read(
        imd.cluster_dir + '/zobjs_clustered.csv', data_start=1).to_pandas()
    zobjs['new_index'] = zobjs.index

    for groupID in range(n_clusters):
        generate_cluster_plots(groupID, emission_df, all_sfrs_res, zobjs, similarity_matrix)
        
    plot_full_uvj(n_clusters)


generate_all_cluster_plots(1)