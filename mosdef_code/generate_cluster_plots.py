'''Runs all of the plotting codes for clusters'''

import matplotlib.pyplot as plt
import initialize_mosdef_dirs as imd
from astropy.io import ascii
from bpt_clusters import plot_bpt_cluster
from emission_measurements import read_emission_df
from plot_mass_sfr import plot_mass_sfr_cluster, read_sfr_df, get_all_sfrs_masses
from uvj_clusters import plot_full_uvj, plot_uvj_cluster, plot_all_uvj_clusters, plot_all_uvj_clusters_paper
from cluster_stats import plot_similarity_cluster
from stack_spectra import plot_spec
from composite_sed import vis_composite_sed
from plot_cluster_a_vs_b import plot_cluster_summaries
from overview_plot_of_clusters import make_overview_plot_clusters
from bpt_clusters_singledf import plot_bpt_all_composites
from composite_and_spec_overview import composite_and_spec_overview
from plot_similarity_matrix import plot_sim_matrix

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
        vis_composite_sed(0, groupID=groupID, run_filters=False, axis_obj=ax_sed)
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
    
    imd.check_and_make_dir(imd.cluster_overview_dir)
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


def generate_newer_cluster_plots(n_clusters):
    ignore_groups = imd.ignore_groups

    imd.check_and_make_dir(imd.cluster_dir + f'/cluster_stats/sfrs')
    
    # # SFR comparison plots
    # plot_cluster_summaries('norm_median_halphas', 'ha_flux', 'sfrs/ha_flux_compare', color_var='balmer_dec', plot_lims=[6e-18, 8e-16, 6e-18, 8e-16], one_to_one=True, ignore_groups=ignore_groups, log=True)
    # plot_cluster_summaries('median_log_sfr', 'computed_log_sfr', 'sfrs/sfr_compare', color_var='balmer_dec', plot_lims=[0.3, 3.5, 0.3, 3.5], one_to_one=True, ignore_groups=ignore_groups)
    # plot_cluster_summaries('median_log_ssfr', 'computed_log_ssfr', 'sfrs/ssfr_compare', color_var='balmer_dec', plot_lims=[-10.7, -6.5, -10.7, -6.5], one_to_one=True, ignore_groups=ignore_groups)

    # # SFMS
    # plot_cluster_summaries('median_log_mass', 'median_log_ssfr', 'sfrs/sfms', color_var='O3N2_metallicity', ignore_groups=ignore_groups)
    # plot_cluster_summaries('median_log_mass', 'computed_log_ssfr', 'sfrs/sfms_computed', color_var='O3N2_metallicity', ignore_groups=ignore_groups)
    # plot_cluster_summaries('median_log_mass', 'computed_log_ssfr', 'sfrs/sfms_computed_balmercolor', color_var='balmer_dec', ignore_groups=ignore_groups)

    # #AV comparison
    # plot_cluster_summaries('AV', 'balmer_av', 'sfrs/av_compare', color_var='norm_median_log_mass', ignore_groups=ignore_groups, one_to_one=True, plot_lims=[0, 4.5, 0, 4.5])
    

    # # BPT diagrams
    # color_codes = ['None', 'log_mass', 'log_sfr', 'balmer_dec', 'metallicity', 'log_ssfr']
    # for color_code in color_codes:
    #     plot_bpt_all_composites(color_code=color_code)

    # UVJ Diagrams
    plot_all_uvj_clusters_paper(n_clusters)
    plot_all_uvj_clusters(n_clusters)
    plot_full_uvj(n_clusters, include_unused_gals='No')
    plot_full_uvj(n_clusters, include_unused_gals='Only')
    plot_full_uvj(n_clusters, color_type='balmer')
    plot_full_uvj(n_clusters, color_type='ssfr')
    plot_full_uvj(n_clusters, color_type='metallicity')

    # Similarity Matrix
    plot_sim_matrix(n_clusters, ssfr_order=True)
    plot_sim_matrix(n_clusters)

    composite_and_spec_overview(n_clusters, ignore_groups)
    make_overview_plot_clusters(n_clusters, bpt_color=True, paper_overview=False, prospector_spec=False)
    


# generate_all_cluster_plots(19, overview=True)
# generate_newer_cluster_plots(19)