'''Runs all methods after clustering the SEDs'''

import sys
import os
import string
import numpy as np
import initialize_mosdef_dirs as imd

from composite_sed import get_all_composite_seds
from stack_spectra import stack_all_spectra
from fit_emission import fit_all_emission
from generate_cluster_plots import generate_all_cluster_plots
from interpolate import gen_all_mock_composites
from uvj_clusters import observe_all_uvj
from convert_filter_to_sedpy import convert_all_folders_to_sedpy, find_median_redshifts
from convert_flux_to_maggies import convert_folder_to_maggies
from plot_scaled_comps import plot_scaled_composites
from scale_spectra import scale_all_spectra
from fit_prospector_emission import setup_all_prospector_fit_csvs, fit_all_prospector_emission
from check_for_agn import check_for_all_agn
from filter_groups import generate_skip_file

'''Starting point: One folder ('cluster_folder') that contains: 
-folders labeled '0', '1', ..., 'N' where N is the number of clusters-1. These will be the cluster "groups"
-each of these folders contains images of each of the seds in a given cluster, named as '{field}_{v4id}_mock.pdf'

Specify the directories in initialize_mosdef_dirs
'''

# Make sure to go to initialize_mosdef_dirs to set all the directories properly

# Set the total number of clusters
n_clusters = 29
# Set the name of the prospector run
run_name = 'redshift_maggies'

# Begin running all the functions
print('Generating composite seds...')
get_all_composite_seds(n_clusters, run_filters=True)
print('Generating composite spectra...')
stack_all_spectra(n_clusters, 'cluster_norm')
print('Fitting emission lines...')

#Check for agn and list which groups do not have enough galaxies
check_for_all_agn(n_clusters)
generate_skip_file()


# Will break here if one of the spectra is so bad that it can't fit
fit_all_emission(n_clusters, 'cluster_norm')

# Need to do a few things to composites (measure uvj, generate mocks sed, etc. before we can plot)
print('Generating plots')
gen_all_mock_composites(n_clusters)
observe_all_uvj(n_clusters, individual_gals=False, composite_uvjs=True)

generate_all_cluster_plots(n_clusters)

# Prepare for prospector:
print('Preparing data for Prospector')
convert_all_folders_to_sedpy(n_clusters)
find_median_redshifts(n_clusters)
convert_folder_to_maggies(imd.composite_sed_csvs_dir)

# Plot of all of the scaled composites, must be run after convert_folder_to_maggies
plot_scaled_composites(n_clusters)
# Scale and re-fit the spectra using the scale that was used for the composites
scale_all_spectra(n_clusters)
# Re-fit the prospector spectra in the same way that we fit the mosdef ones:


setup_all_prospector_fit_csvs(29, run_name)
fit_all_prospector_emission(29, run_name)

