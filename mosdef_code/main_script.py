'''Runs all methods after clustering the SEDs'''

import sys
import os
import string
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
from read_data import mosdef_df
from cross_correlate import get_cross_cor
import matplotlib.pyplot as plt
from filter_response import lines, overview, get_index, get_filter_response
from scipy import interpolate
import initialize_mosdef_dirs as imd
import time
from composite_sed import get_all_composite_seds
from stack_spectra import stack_all_spectra

'''Starting point: One folder ('cluster_folder') that contains: 
-folders labeled '0', '1', ..., 'N' where N is the number of clusters-1. These will be the cluster "groups"
-each of these folders contains images of each of the seds in a given cluster, named as '{field}_{v4id}_mock.pdf'
'''

# Make sure to go to initialize_mosdef_dirs to set all the directories properly

# Set the total number of clusters
n_clusters = 1
print('Generating composite seds...')
# get_all_composite_seds(n_clusters, run_filters=True)
print('Generating composite spectra...')
stack_all_spectra(n_clusters, 'cluster_norm')
