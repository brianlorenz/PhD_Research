# Makes a flag for galaxies that are 3 sigma off of the F160/F125 axis ratio relation

import numpy as np
import pandas as pd
from astropy.io import ascii
import matplotlib.pyplot as plt
import matplotlib as mpl
import initialize_mosdef_dirs as imd
import matplotlib.patheffects as path_effects


def flag_axis_ratios(plot=False):
    # Read in the axis ratio data
    ar_df = ascii.read(imd.loc_axis_ratio_cat).to_pandas()

    # add a new column for the ar flag
    ar_df['axis_ratio_flag'] = np.zeros(len(ar_df))

    # Remove where either column is -999
    bad_gals = np.logical_or(ar_df['F160_axis_ratio']<-99, ar_df['F125_axis_ratio']<-99)
    flag_idxs = ar_df[bad_gals].index
    ar_df.loc[flag_idxs, 'axis_ratio_flag'] = -999
    good_gals = ar_df[~bad_gals]

    # Compute the difference and standard devaition
    ar_diff = good_gals['F160_axis_ratio'] - good_gals['F125_axis_ratio']
    err_ar_diff = good_gals['F160_err_axis_ratio'] + good_gals['F125_err_axis_ratio']
    std_ar_diff = np.std(ar_diff)

    # Flag the galaixes greater than 2 sigma
    above_sigma = ar_diff > 2*std_ar_diff
    below_sigma = ar_diff < -2*std_ar_diff
    flagged_gals = np.logical_or(above_sigma, below_sigma)
    idxs = ar_diff[flagged_gals].index
    ar_df.loc[idxs, 'axis_ratio_flag'] = 1
    flagged_gals_ar = ar_df['axis_ratio_flag'] == 1
    flagged_gals_diff = ar_df[flagged_gals_ar]['F160_axis_ratio'] - ar_df[flagged_gals_ar]['F125_axis_ratio']
    err_flagged_gals_diff = ar_df[flagged_gals_ar]['F160_err_axis_ratio'] + ar_df[flagged_gals_ar]['F125_err_axis_ratio']

    if plot==True:
        fig, ax = plt.subplots(figsize=(8,8))
        ax.errorbar(good_gals['F160_axis_ratio'], ar_diff, yerr = err_ar_diff, marker='o', ls='None', color='black', zorder=1)
        ax.errorbar(ar_df[flagged_gals_ar]['F160_axis_ratio'], flagged_gals_diff, yerr=err_flagged_gals_diff, marker='o', ls='None', color='red', zorder=2)
        ax.axhline(std_ar_diff, color='skyblue', ls='--', label='1 sigma', zorder=3)
        ax.axhline(-std_ar_diff, color='skyblue', ls='--', zorder=3)
        ax.axhline(2*std_ar_diff, color='blue', ls='--', label='2 sigma', zorder=3)
        ax.axhline(2*(-std_ar_diff), color='blue', ls='--', zorder=3)
        ax.set_xlabel('F160 Axis Ratio')
        ax.set_ylabel('F160 ar - F125 ar')
        ax.set_ylim(-1, 1)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=14)
        fig.savefig(imd.axis_output_dir + f'/axis_ratio_filtering.pdf')

        ar_df.to_csv(imd.loc_axis_ratio_cat, index=False)

flag_axis_ratios()