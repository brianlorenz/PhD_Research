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
from prospector_plot import make_all_prospector_plots_2groups

run_name = 'fourgroups_2'
groupID1 = 0
groupID2 = 1
groupID3 = 3
groupID4 = 16


def make_tfig_cfig(run_name):
    all_files = os.listdir(imd.prospector_h5_dir + f'/{run_name}_h5s')
    all_files = [file for file in all_files if '.h5' in file]

    imd.check_and_make_dir(imd.prospector_plot_dir + f'/{run_name}_plots')
    for file in all_files:
        file_shortname = file[:-19]

        res, obs, mod = reader.results_from(imd.prospector_h5_dir + f'/{run_name}_h5s/' + file)

        tfig = reader.traceplot(res)
        tfig.savefig(imd.prospector_plot_dir + f'/{run_name}_plots' + f'/{file_shortname}_tfig.pdf')
        cfig = reader.subcorner(res)
        cfig.savefig(imd.prospector_plot_dir + f'/{run_name}_plots' + f'/{file_shortname}_cfig.pdf')







# make_tfig_cfig(run_name)
make_all_prospector_plots_2groups(groupID1, groupID2, groupID3, groupID4, run_name)