'''Runs all of the plotting codes for clusters'''

import matplotlib.pyplot as plt
import initialize_mosdef_dirs as imd
from astropy.io import ascii
from bpt_clusters import plot_bpt_cluster
from emission_measurements import read_emission_df
from plot_mass_sfr import plot_mass_sfr_cluster, read_sfr_df, get_all_sfrs_masses
from uvj_clusters import plot_full_uvj, plot_uvj_cluster
from cluster_stats import plot_similarity_cluster
from stack_spectra import plot_spec

def generate_cluster_plots(groupID, emission_df, all_sfrs_res, zobjs, similarity_matrix, overview=False):
    '''
    Generates all plots for the cluster with the given groupID
    
    Parameters:
    groupID (int): the groupID of the cluster
    emission_df (pd.DataFrame): Use read_emission_df
    all_sfrs_res (pd.DataFrame): Use read_sfr_df and get_all_ssfrs_masses
    zobjs (list of tuples): see get_zobjs()
    similarity_matrix (array): Matrix containing the similarities between all clusters
    overview (boolean): Set to true to generate an overview rather than individual plots

    Returns:
    '''
    if overview == True:
        # Figure setup
        fig = plt.figure(figsize=(14, 8))
        ax_sed = fig.add_axes([0.04, 0.40, 0.44, 0.50])
        ax_spectrum = fig.add_axes([0.04, 0.04, 0.44, 0.30])
        ax_similarity = fig.add_axes([0.52, 0.63, 0.20, 0.30])
        ax_bpt = fig.add_axes([0.76, 0.63, 0.20, 0.30])
        ax_mass_sfr = fig.add_axes([0.52, 0.09, 0.20, 0.30])
        ax_uvj = fig.add_axes([0.76, 0.09, 0.20, 0.30])
        plot_spec(groupID, 'cluster_norm', thresh=0.1, axis_obj = ax_spectrum)
        
    
    else:
        ax_similarity = 'False'
        ax_bpt = 'False'
        ax_mass_sfr = 'False'
        ax_uvj = 'False'
        
    plot_similarity_cluster(groupID, zobjs, similarity_matrix, axis_obj=ax_similarity)
    plot_bpt_cluster(emission_df, groupID, axis_obj=ax_bpt)
    plot_mass_sfr_cluster(groupID, all_sfrs_res, axis_obj=ax_mass_sfr)
    plot_uvj_cluster(groupID, axis_obj=ax_uvj)
    
    fig.savefig(imd.cluster_overview_dir + f'/{groupID}_overview.pdf')
    
    


def generate_all_cluster_plots(n_clusters, overview=False):
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
        generate_cluster_plots(groupID, emission_df, all_sfrs_res, zobjs, similarity_matrix, overview=overview)
        
    plot_full_uvj(n_clusters)


generate_all_cluster_plots(1, overview=True)