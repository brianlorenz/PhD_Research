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
import matplotlib.pyplot as plt
from prospector_plot import make_all_singleplots_2groups, make_all_prospector_plots_2groups, make_all_prospector_plots, make_all_singleplots
from fit_prospector_emission import multiply_fit_by_lumdist, setup_all_prospector_fit_csvs, fit_all_prospector_emission
from prospector_output_props import add_props_to_cluster_summary_df, save_props


# 2 groups sectoin
# run_name = 'par_ly_mask'
# groupID1 = 0
# groupID2 = 11
# groupID3 = 14
# groupID4 = 18

# All groups
run_name = 'removed_kewley_agn'
n_clusters = 20




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
        plt.close('all')




# make_tfig_cfig(run_name)
# make_all_prospector_plots(n_clusters, run_name)
# make_all_singleplots(n_clusters, run_name)
# setup_all_prospector_fit_csvs(n_clusters, run_name)
# fit_all_prospector_emission(n_clusters, run_name)
# multiply_fit_by_lumdist(n_clusters, run_name)
# save_props(n_clusters, run_name) 
# add_props_to_cluster_summary_df(n_clusters, run_name) # Adds masses and computes sfr/ssfr


# make_all_prospector_plots_2groups(0, 7, 8, 11, run_name)
# make_all_singleplots_2groups(0, 7, 8, 11, run_name)