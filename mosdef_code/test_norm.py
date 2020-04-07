# Tests and plots the normalization process

import sys
import os
import string
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
from read_data import mosdef_df
from mosdef_obj_data_funcs import read_sed
from clustering import cluster_dir
import matplotlib.pyplot as plt
from filter_response import lines, overview, get_index, get_filter_response
from scipy import interpolate
from composite_sed import get_normalized_sed, get_good_idx


def vis_normalized_sed(target_field, target_v4id, field, v4id):
    target_sed = read_sed(target_field, target_v4id)
    sed = get_normalized_sed(target_field, target_v4id, field, v4id)

    axisfont = 14
    ticksize = 12
    ticks = 8
    titlefont = 24
    legendfont = 14
    textfont = 16

    target_good_idx = get_good_idx(target_sed)
    sed_good_idx = get_good_idx(sed)

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.errorbar(target_sed[target_good_idx]['peak_wavelength'], target_sed[target_good_idx]['f_lambda'], yerr=target_sed[target_good_idx]['err_f_lambda'],
                ls='', marker='o', markersize=4, color='black', label='target')
    ax.errorbar(sed[sed_good_idx]['peak_wavelength'], sed[sed_good_idx]['f_lambda'], yerr=sed[sed_good_idx]['err_f_lambda'], ls='',
                marker='o', markersize=4, color='red', label='without norm')
    ax.errorbar(sed[sed_good_idx]['peak_wavelength'], sed[sed_good_idx]['f_lambda_norm'], yerr=sed[sed_good_idx]['err_f_lambda_norm'], ls='',
                marker='o', markersize=4, color='blue', label='normalized')

    ax.set_xlabel('Rest Wavelength ($\AA$)', fontsize=axisfont)
    ax.set_ylabel('Flux', fontsize=axisfont)
    ax.set_xscale('log')
    #ax_sed.set_ylim(-0.2, 5)
    #ax_filt.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.show()
    ax.legend()
    # fig.savefig(f'/Users/galaxies-air/mosdef/Clustering/composite_seds/{groupID}_sed.pdf')
    # plt.close()


vis_normalized_sed('COSMOS', 6202, 'GOODS-N', 19654)
