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
from convert_filter_to_sedpy import convert_all_folders_to_sedpy
from convert_flux_to_maggies import convert_folder_to_maggies

'''Starting point: One folder ('cluster_folder') that contains: 
-folders labeled '0', '1', ..., 'N' where N is the number of clusters-1. These will be the cluster "groups"
-each of these folders contains images of each of the seds in a given cluster, named as '{field}_{v4id}_mock.pdf'
'''

# Make sure to go to initialize_mosdef_dirs to set all the directories properly

# Set the total number of clusters
n_clusters = 29

# # Begin running all the functions
# print('Generating composite seds...')
# get_all_composite_seds(n_clusters, run_filters=True)
# print('Generating composite spectra...')
# stack_all_spectra(n_clusters, 'cluster_norm')
# print('Fitting emission lines...')

# # Will break here if one of the spectra is so bad that it can't fit
# fit_all_emission(n_clusters, 'cluster_norm')

# # Need to do a few things to composites (measure uvj, generate mocks sed, etc. before we can plot)
# gen_all_mock_composites(n_clusters)
# observe_all_uvj(n_clusters, individual_gals=False, composite_uvjs=True)
 
# generate_all_cluster_plots(n_clusters)

# Prepare for prospector:
# convert_all_folders_to_sedpy(n_clusters)
convert_folder_to_maggies(imd.composite_sed_csvs_dir)



