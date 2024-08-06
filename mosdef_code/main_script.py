'''Runs all methods after clustering the SEDs'''
import sys
import os
import string
import numpy as np
import initialize_mosdef_dirs as imd


from composite_sed import get_all_composite_seds
from stack_spectra import stack_all_spectra
from fit_emission import fit_all_emission, compute_bootstrap_uncertainties
from generate_cluster_plots import generate_all_cluster_plots, generate_newer_cluster_plots
from interpolate import gen_all_mock_composites
from uvj_clusters import observe_all_uvj
from convert_filter_to_sedpy import convert_all_folders_to_sedpy, find_median_redshifts
from convert_flux_to_maggies import convert_folder_to_maggies
from plot_scaled_comps import plot_scaled_composites
from scale_spectra import scale_all_spectra, scale_all_spec_to_median_halpha
from fit_prospector_emission import multiply_fit_by_lumdist, setup_all_prospector_fit_csvs, fit_all_prospector_emission
from check_for_agn import check_for_all_agn
from filter_groups import remove_groups_by_similiary
from make_clusters_summary_df import make_clusters_summary_df
from add_norm_factors_to_group_dfs import add_norm_factors
from compute_cluster_sfrs import compute_cluster_sfrs, compute_new_sfrs_compositepaper
from balmer_dec_histogram import compute_balmer_lower_limits, plot_balmer_hist
from compute_indiv_sfrs_from_halpha import compute_indiv_sfrs
from plot_group_hists import plot_group_hists
from prospector_output_props import add_props_to_cluster_summary_df, save_props
from cluster_stats import plot_all_similarity, remove_dissimilar_gals, remove_flagged_seds
from generate_clusters import generate_clusters, read_filtered_gal_df
from filter_groups import find_bad_seds


'''Starting point: One folder ('cluster_folder') that contains: 
-folders labeled '0', '1', ..., 'N' where N is the number of clusters-1. These will be the cluster "groups"
-each of these folders contains images of each of the seds in a given cluster, named as '{field}_{v4id}_mock.pdf'

Specify the directories in initialize_mosdef_dirsl;k
'''

# Make sure to run generate_clusters.py first to set up the clusters


# Set the total number of clusters
n_clusters = 20
# Set the name of the prospector run
run_name = 'removed_kewley_agn'
norm_method = 'luminosity'
# Set which group numbers to ignore since their data is not good
ignore_groups = []
# ignore_groups = [0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19]
# ignore_groups = [0,1,2,3,4,5,6,7,8,9,10,11,19]
# ignore_groups = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
# Set hwo many times to bootstrap
big_bootstrap_num = 1000
bootstrap = big_bootstrap_num
# bootstrap = -1
halpha_scaled=False

imd.check_and_make_dir(imd.cluster_dir + '/composite_filters')
imd.check_and_make_dir(imd.composite_filter_csvs_dir)
imd.check_and_make_dir(imd.composite_filter_images_dir)
imd.check_and_make_dir(imd.composite_filter_sedpy_dir)

# # generate_clusters(20, stop_to_eval=False, skip_slow_steps=True)

# get_all_composite_seds(n_clusters, run_filters=False)
# gen_all_mock_composites(n_clusters)
# plot_all_similarity(n_clusters)
# remove_dissimilar_gals(n_clusters) ### Run this only once

# # Begin running all the functions
# print('Generating composite seds...')
# get_all_composite_seds(n_clusters, run_filters=True)
# print('Generating composite spectra...')
# stack_all_spectra(n_clusters, norm_method, bootstrap=bootstrap, ignore_groups=ignore_groups)


# # Re-fit the emission of the composites and now fit the boostrapped ones
# print('Fitting emission lines...')
# fit_all_emission(n_clusters, norm_method, ignore_groups, bootstrap=bootstrap)
# bootstrap = -1
# fit_all_emission(n_clusters, norm_method, ignore_groups, bootstrap=bootstrap)

# # Add the normalizations to the group dfs
# add_norm_factors(n_clusters)

# make_clusters_summary_df(n_clusters, ignore_groups)

# # Need to do a few things to composites (measure uvj, generate mocks sed, etc. before we can plot)
# print('Generating plots')
# gen_all_mock_composites(n_clusters)
# observe_all_uvj(n_clusters, individual_gals=False, composite_uvjs=True)
# print('Done - composite SEDs are ready')

# # Compute the ssfr for the groups
# bootstrap = big_bootstrap_num
# plot_balmer_hist(n_clusters, bootstrap)
# bootstrap = -1
# compute_balmer_lower_limits(sig_noise_thresh=3, hb_sig_noise_thresh=3)
# compute_cluster_sfrs(luminosity=True, monte_carlo=True)
# compute_new_sfrs_compositepaper(n_clusters, imf='subsolar')
# compute_indiv_sfrs(n_clusters, lower_limit=True)

# # Have to run this twice, since ignore_groups won't be loaded properly the first time
# generate_newer_cluster_plots(n_clusters, norm_method)
# plot_group_hists(n_clusters)

# generate_all_cluster_plots(n_clusters)
# breakpoint()


# # Prepare for prospector:
# print('Preparing data for Prospector')
# find_median_redshifts(n_clusters)
# convert_all_folders_to_sedpy(n_clusters, ignore_groups=ignore_groups)
# convert_folder_to_maggies(imd.composite_sed_csvs_dir)

# Plot of all of the scaled composites, must be run after convert_folder_to_maggies
# plot_scaled_composites(n_clusters)


# Re-fit the prospector spectra in the same way that we fit the mosdef ones:

# setup_all_prospector_fit_csvs(n_clusters, run_name)
# fit_all_prospector_emission(n_clusters, run_name)
# multiply_fit_by_lumdist(n_clusters, run_name)
# save_props(n_clusters, run_name) 
# add_props_to_cluster_summary_df(n_clusters, run_name) # Adds masses and computes sfr/ssfr



# Shortcut for recalculating sfrs and avs
# make_clusters_summary_df(n_clusters, ignore_groups)
# plot_balmer_hist(n_clusters, bootstrap)
# compute_balmer_lower_limits(sig_noise_thresh=3, hb_sig_noise_thresh=3)
# compute_cluster_sfrs(luminosity=True, monte_carlo=True)
# compute_new_sfrs_compositepaper(n_clusters, imf='subsolar')
# compute_indiv_sfrs(n_clusters, lower_limit=True)
# save_props(n_clusters, run_name) 
# add_props_to_cluster_summary_df(n_clusters, run_name) # Adds masses and computes sfr/ssfr

### OLD

# Scale and re-fit the spectra using the scale that was used for the composites
# scale_all_spectra(n_clusters)

# # # Add the bootstrapped uncertainties - shouldn't do this currently
# compute_bootstrap_uncertainties(n_clusters, 'None', bootstrap=bootstrap, clustering=True, ignore_groups=ignore_groups)
