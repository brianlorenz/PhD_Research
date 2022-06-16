# Run KS tests on various pieces of the data

from pygments import highlight
from scipy import stats
import numpy as np
from astropy.io import ascii
import pandas as pd
import initialize_mosdef_dirs as imd
from axis_ratio_funcs import filter_ar_df, read_filtered_ar_df
import matplotlib.pyplot as plt
from plot_vals import *


def run_ks_tests(cut_middle=True):
    '''
    
    Parameters:
    cut_middle (boolean): Set to True to remove the middle set of axis ratios, from 0.4 to 0.7
    '''
    ar_df = read_filtered_ar_df()
    print(len(ar_df))
    save_str = ''

    if cut_middle == True:
        middle_idxs = np.logical_and(ar_df['use_ratio']>0.4, ar_df['use_ratio']<0.7)
        ar_df = ar_df[middle_idxs]
        print(len(ar_df))
        save_str = '_mid_cut'

    
    low_axis = ar_df['use_ratio']<0.55
    high_axis = ~low_axis

    # Are the mass distributions the same between axis ratios? 
    low_mass = ar_df[low_axis]['log_mass']
    high_mass = ar_df[high_axis]['log_mass']
    ks_stat, pvalue = run_one_ks_test(low_mass, high_mass, 'low axis mass', 'high axis mass')
    bins = np.arange(9, 11, 0.05)
    histogram_of_samples(low_mass, high_mass, 'low axis mass', 'high axis mass', bins, stellar_mass_label, ks_stat, pvalue, fig_savename=f'axis_ratio_mass_kstest{save_str}')

    # Are the sfr distributions the same between axis ratios? 
    low_sfr = np.log10(ar_df[low_axis]['use_sfr'])
    high_sfr = np.log10(ar_df[high_axis]['use_sfr'])
    ks_stat, pvalue = run_one_ks_test(low_sfr, high_sfr, 'low axis sfr', 'high axis sfr')
    bins = np.arange(0, 3, 0.1)
    histogram_of_samples(low_sfr, high_sfr, 'low axis sfr', 'high axis sfr', bins, sfr_label, ks_stat, pvalue, fig_savename=f'axis_ratio_logsfr_kstest{save_str}')

def run_one_ks_test(data1, data2, name1, name2):
    """Runs a ks_test between the two columns:

    Parameters:
    data1 and data2 (array-like): data points to run the test on
    name1 and name2 (str): descriptive names to use when displaying results
    """
    print('------------------------------------------')
    ks_stat, pvalue = stats.ks_2samp(data1, data2)
    print(f'{name1} mean: {round(np.mean(data1), 3)}, std: {round(np.std(data1), 3)}')
    print(f'{name2} mean: {round(np.mean(data2), 3)}, std: {round(np.std(data2), 3)}')
    print(f'ks_statistic: {round(ks_stat, 3)}, pvalue: {round(pvalue, 3)}')
    print('------------------------------------------')
    return ks_stat, pvalue

def histogram_of_samples(data1, data2, name1, name2, bins, xlabel, ks_stat, pvalue, fig_savename='ks_test'):
    """Plots the two samples used in the ks test

    Parameters:
    data1 and data2 (array-like): data points to run the test on
    name1 and name2 (str): descriptive names to use when displaying results
    bins (array-like): histogram bins
    """
    fig, ax = plt.subplots(figsize=(8,8))
    
    ax.hist(data1, color=light_color, bins=bins, label=name1, zorder=2, alpha=0.5)
    ax.hist(data2, color=dark_color, bins=bins, label=name2, zorder=1)
    ax.set_xlabel(xlabel, fontsize = 14)
    ax.set_ylabel('Counts', fontsize = 14)
    ax.text(0.02, 0.95, f'ks_statistic: {round(ks_stat, 3)}, pvalue: {round(pvalue, 3)}', transform=ax.transAxes, fontsize = 12)
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=12)
    fig.savefig(imd.axis_output_dir + f'/ks_tests/{fig_savename}.pdf')


