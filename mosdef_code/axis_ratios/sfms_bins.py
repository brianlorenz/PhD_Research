
import os
from scipy import stats
from astropy.io import ascii
import numpy as np
import matplotlib.pyplot as plt
import initialize_mosdef_dirs as imd
from axis_ratio_funcs import read_interp_axis_ratio, filter_ar_df, read_filtered_ar_df

sfms_slope = 0.598 
sfms_yint = -4.475


def find_sfms():
    '''Find the slope and intercept of the sfms in our sample'''
    
    # ar_df = read_interp_axis_ratio()
    # ar_df = filter_ar_df(ar_df)

    ar_df = read_filtered_ar_df()

    # Add a column for ssfr
    ar_df['log_ssfr'] = np.log10((ar_df['sfr'])/(10**ar_df['log_mass']))
    ar_df['log_halpha_ssfr'] = np.log10((ar_df['halpha_sfrs'])/(10**ar_df['log_mass']))
    ar_df['log_use_ssfr'] = np.log10((ar_df['use_sfr'])/(10**ar_df['log_mass']))
    ar_df['log_use_sfr'] = np.log10(ar_df['use_sfr'])
    
    colors = ['red', 'orange', 'blue', 'black', 'brown', 'green', 'pink', 'grey', 'purple', 'cyan', 'navy', 'magenta']

    fig, ax = plt.subplots(figsize=(8,8))

    ax.plot(ar_df['log_mass'], ar_df['log_use_sfr'], color='black', ls='None', marker='o')

    ax.set_xlim(8.95, 11.05)
    ax.set_ylim(-0.1, 2.6)
    
    ax.set_xlabel('log(Stellar Mass)')
    ax.set_ylabel('log(SFR)')

    fit = stats.linregress(ar_df['log_mass'], ar_df['log_use_sfr'])
    y1 = fit.slope*x+fit.intercept
    plt.plot(x, y1, color='red', ls='--')
    ax.text(9, 2.3, f'slope: {round(fit.slope, 3)}, yint: {round(fit.intercept, 3)}')

    fig.savefig(imd.axis_output_dir + f'/sfms.pdf')

def plot_sfms_bins(save_name, nbins, split_by):
    '''Divide the galaxies in sfr_mass space along the sfms and plot it'''
    
    group_dfs = [ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_group_dfs/{i}_df.csv').to_pandas() for i in range(nbins)]
    colors = ['red', 'orange', 'blue', 'black', 'brown', 'green', 'pink', 'grey', 'purple', 'cyan', 'navy', 'magenta']

    fig, ax = plt.subplots(figsize=(8,8))

    for i in range(len(group_dfs)):
        df = group_dfs[i]
        ax.plot(df['log_mass'], df[split_by], color=colors[i], ls='None', marker='o')
    x = np.linspace(8.8, 11.2, 100)
    y1 = sfms_slope*x+sfms_yint
    # y2 = 1.07*x-9.15
    plt.plot(x, y1, color='black', ls='--')
    plt.axvline(10, color='black', ls='--')
    # plt.plot(x, y2, color='black', ls='--')

    ax.set_xlim(8.95, 11.05)
    ax.set_ylim(-0.1, 2.6)
    
    ax.set_xlabel('log(Stellar Mass)')
    ax.set_ylabel('log(SFR)')
    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/sfr_mass.pdf')

# find_sfms()
# plot_sfms_bins('both_sfms_6bin_median_2axis', 12, 'log_use_sfr')
#low cut - (9.5, 0.3), (11.0, 1.9)  y = 1.07x-9.83
#high cut - (9.0, 0.5), (10.5, 2.0) y = 1.07x-8.6


