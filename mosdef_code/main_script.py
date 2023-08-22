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
from scale_spectra import scale_all_spectra
from fit_prospector_emission import setup_all_prospector_fit_csvs, fit_all_prospector_emission
from check_for_agn import check_for_all_agn
from filter_groups import generate_skip_file, remove_groups_by_similiary
from make_clusters_summary_df import make_clusters_summary_df
from add_norm_factors_to_group_dfs import add_norm_factors
from compute_cluster_sfrs import compute_cluster_sfrs
from balmer_dec_histogram import compute_balmer_lower_limits
from scale_spectra import scale_all_spec_to_median_halpha


'''Starting point: One folder ('cluster_folder') that contains: 
-folders labeled '0', '1', ..., 'N' where N is the number of clusters-1. These will be the cluster "groups"
-each of these folders contains images of each of the seds in a given cluster, named as '{field}_{v4id}_mock.pdf'

Specify the directories in initialize_mosdef_dirsl;k
'''

# Make sure to go to initialize_mosdef_dirs to set all the directories properly

# Set the total number of clusters
n_clusters = 19
# Set the name of the prospector run
run_name = 'test'
# Set which group numbers to ignore since their data is not good
ignore_groups = []
# ignore_groups = [0,1,2,3,4,5,6,7,8,9,10,11,19]
# ignore_groups = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
# Set hwo many times to bootstrap
bootstrap = 1000
halpha_scaled=True

# imd.check_and_make_dir(imd.composite_filter_csvs_dir)
# imd.check_and_make_dir(imd.composite_filter_images_dir)
# imd.check_and_make_dir(imd.composite_filter_sedpy_dir)

# # Begin running all the functions
# print('Generating composite seds...')
# get_all_composite_seds(n_clusters, run_filters=True)
# print('Generating composite spectra...')
# stack_all_spectra(n_clusters, 'cluster_norm', bootstrap=bootstrap, ignore_groups=ignore_groups)
# print('Fitting emission lines...')

# # # Check for agn and list which groups do not have enough galaxies - cuts down to 20
# check_for_all_agn(n_clusters)


# # Don't fit the bootstrapped spectra yet - need to scale them first
# fit_all_emission(n_clusters, 'cluster_norm', ignore_groups, bootstrap=-1)
# # Scales the composite spectra and the boostrapped spectra
# scale_all_spec_to_median_halpha(n_clusters, bootstrap=bootstrap)
# # Re-fit the emission of the composites and now fit the boostrapped ones
# fit_all_emission(n_clusters, 'cluster_norm', ignore_groups, bootstrap=bootstrap, halpha_scaled=halpha_scaled)

# # # Add the bootstrapped uncertainties
# compute_bootstrap_uncertainties(n_clusters, 'None', bootstrap=bootstrap, clustering=True, ignore_groups=ignore_groups, halpha_scaled=halpha_scaled)

# # # Add the normalizations to the group dfs
# add_norm_factors(n_clusters)

# make_clusters_summary_df(n_clusters, ignore_groups, halpha_scaled=halpha_scaled)

# # Need to do a few things to composites (measure uvj, generate mocks sed, etc. before we can plot)
# print('Generating plots')
# gen_all_mock_composites(n_clusters)
# observe_all_uvj(n_clusters, individual_gals=False, composite_uvjs=True)
# print('Done - composite SEDs are ready')

# # Figure out which groups to exclude from plots - CHECK THRESHOLD in /cluster_stats/similarities/composite_similarities.csv
# remove_groups_by_similiary(n_clusters, sim_thresh=0.8)

# # Compute the ssfr for the groups
# compute_balmer_lower_limits()
# compute_cluster_sfrs()

# # Have to run this twice, since ignore_groups won't be loaded properly the first time
# generate_newer_cluster_plots(n_clusters)
# generate_all_cluster_plots(n_clusters)
breakpoint()


# Prepare for prospector:
print('Preparing data for Prospector')
find_median_redshifts(n_clusters)
convert_all_folders_to_sedpy(n_clusters)
convert_folder_to_maggies(imd.composite_sed_csvs_dir)

# Plot of all of the scaled composites, must be run after convert_folder_to_maggies
plot_scaled_composites(n_clusters)
# Scale and re-fit the spectra using the scale that was used for the composites
scale_all_spectra(n_clusters)

# Re-fit the prospector spectra in the same way that we fit the mosdef ones:

# setup_all_prospector_fit_csvs(29, run_name)
# fit_all_prospector_emission(29, run_name)


