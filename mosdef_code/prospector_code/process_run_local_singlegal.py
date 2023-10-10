# Performs the parts of process_run that can quickly be done locally

from cosmology_calcs import luminosity_to_flux
from convert_flux_to_maggies import prospector_maggies_to_flux, prospector_maggies_to_flux_spec
import prospect.io.read_results as reader
import sys
import os
import numpy as np
import pandas as pd
import pickle
import initialize_mosdef_dirs as imd
from prospector_plot import make_all_prospector_plots_2groups, make_all_prospector_plots

# 2 groups sectoin
# run_name = 'par_ly_mask'
# groupID1 = 0
# groupID2 = 11
# groupID3 = 14
# groupID4 = 18

# All groups
run_name = 'first_test_19groups'
n_clusters = 19

prospector_csvs_dir = '/Users/brianlorenz/mosdef/prospector_singlegal_tests/csvs'
prospector_plots_dir = '/Users/brianlorenz/mosdef/prospector_singlegal_tests/'
        
def make_tfig_cfig(run_name):
    # all_files = os.listdir(imd.mosdef_dir + f'/prospector_singlegal_tests/')
    all_files = os.listdir(imd.mosdef_dir + f'/prospector_singlegal_tests/')
    all_files = [file for file in all_files if '.h5' in file]

    imd.check_and_make_dir(prospector_plots_dir + f'/{run_name}_plots')
    for file in all_files:
        file_shortname = file[:-19]

        res, obs, mod = reader.results_from(imd.mosdef_dir + f'/prospector_singlegal_tests/' + file)
        breakpoint()

        tfig = reader.traceplot(res)
        tfig.savefig(prospector_plots_dir + f'/{run_name}_plots' + f'/{file_shortname}_tfig.pdf')
        cfig = reader.subcorner(res)
        cfig.savefig(prospector_plots_dir + f'/{run_name}_plots' + f'/{file_shortname}_cfig.pdf')










make_tfig_cfig(run_name)
make_all_prospector_plots(n_clusters, run_name)
# make_all_prospector_plots_2groups(groupID1, groupID2, groupID3, groupID4, run_name)