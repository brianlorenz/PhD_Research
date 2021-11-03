# Plot a histogram of the axis ratios of the galaxies being used
import numpy as np
import pandas as pd
from astropy.io import ascii
import matplotlib.pyplot as plt
import matplotlib as mpl
import initialize_mosdef_dirs as imd
import matplotlib.patheffects as path_effects


def plot_ar_hist(use_column = 'use_ratio'):
    '''Makes an axis ratio hisogram

    Parameters: 
    use_column (str): Column to retrieve the axis ratios from
    '''

    ar_df = ascii.read(imd.loc_axis_ratio_cat).to_pandas()
    
    fig, ax = plt.subplots(figsize=(8,8))

    ar_df = ar_df[ar_df[use_column]>-90]

    ax.hist(ar_df[use_column], 20, color='black')

    ax.axvline(0.4, ls='--', color='red')
    ax.axvline(0.7, ls='--', color='red')

    ax.set_xlabel('Axis Ratio', fontsize=14) 
    ax.set_ylabel('Number of Galaxies', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.text(-0.07, -10, 'Edge-on', fontsize=14, zorder=100)
    ax.text(0.95, -10, 'Face-on', fontsize=14, zorder=100)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 110)
    fig.savefig(imd.axis_output_dir + f'/ar_histogram_{use_column}.pdf')


plot_ar_hist()
plot_ar_hist(use_column='F125_axis_ratio')
plot_ar_hist(use_column='F140_axis_ratio')
plot_ar_hist(use_column='F160_axis_ratio')



def compare_ar_measurements(col1, err_col1, col2, err_col2):
    '''Makes an axis ratio plot comparing two methods of determining ar

    Parameters: 
    col1 (str): Column to retrieve the axis ratios from
    col1 (str): Column to retrieve the axis ratios from
    '''

    ar_df = ascii.read(imd.loc_axis_ratio_cat).to_pandas()


    cmap = mpl.cm.viridis 
    norm = mpl.colors.Normalize(vmin=1.3, vmax=2.8) 
    
    fig, ax = plt.subplots(figsize=(8,8))

    ar_df = ar_df[ar_df[col1]>-90]
    ar_df = ar_df[ar_df[col2]>-90]


    for i in range(len(ar_df)):
        row = ar_df.iloc[i]
        rgba = cmap(norm(row['z']))

        ax.errorbar(row[col1], row[col2], xerr=row[err_col1], yerr=row[ err_col2], marker='o', ls='None', color=rgba)
        

    ax.plot((-1,2), (-1, 2), color='red', ls='-')

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label('Redshift', fontsize=14)

    ax.set_xlabel(f'{col1} Axis Ratio', fontsize=14) 
    ax.set_ylabel(f'{col2} Axis Ratio', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.text(-0.07, -10, 'Edge-on', fontsize=14, zorder=100)
    ax.text(0.95, -10, 'Face-on', fontsize=14, zorder=100)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    fig.savefig(imd.axis_output_dir + f'/ar_compare_{col1}_{col2}.pdf')

compare_ar_measurements('F125_axis_ratio', 'F125_err_axis_ratio', 'F160_axis_ratio', 'F160_err_axis_ratio')
compare_ar_measurements('use_ratio', 'err_use_ratio', 'F160_axis_ratio', 'F160_err_axis_ratio')
compare_ar_measurements('use_ratio', 'err_use_ratio', 'F125_axis_ratio', 'F125_err_axis_ratio')