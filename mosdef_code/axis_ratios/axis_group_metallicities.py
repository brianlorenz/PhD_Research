# Plot metallicity vs axis ratio for each group


import numpy as np
import pandas as pd
from astropy.io import ascii
import matplotlib.pyplot as plt
import matplotlib as mpl
import initialize_mosdef_dirs as imd
import matplotlib.patheffects as path_effects
from matplotlib.gridspec import GridSpec
from brokenaxes import brokenaxes

def plot_overlaid_spectra(savename='halpha_norm'):
    """Make the plot
    
    """
    summary_df = ascii.read(imd.axis_cluster_data_dir + f'/{savename}/summary.csv').to_pandas()
    
    # Start the figure
    # fig, axarr = plt.subplots(3, 2, figsize=(14,10))
    

    fig, axarr = plt.subplots(3, 2, figsize=(14, 20))

    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1.0)
    
    # axarr = GridSpec(3,2, left=0.1, right=0.8, wspace=0.4, hspace=0.6)
    ax_lowm_highs = axarr[0,0]
    ax_highm_highs = axarr[0,1]
    ax_lowm_lows = axarr[1,0]
    ax_highm_lows = axarr[1,1]
    ax_lowest_lows = axarr[2,0]
    ax_lowest_highs = axarr[2,1]

    # plot_lims = ((4850, 4875), (6540, 6590))

        
    # bax_lowm_highs = brokenaxes(xlims=plot_lims, subplot_spec=axarr[0,0])
    # bax_highm_highs = brokenaxes(xlims=plot_lims, subplot_spec=axarr[0,1])
    # bax_lowm_lows = brokenaxes(xlims=plot_lims, subplot_spec=axarr[1,0])
    # bax_highm_lows = brokenaxes(xlims=plot_lims, subplot_spec=axarr[1,1])
    # bax_lowest_lows = brokenaxes(xlims=plot_lims, subplot_spec=axarr[2,0])
    # bax_lowest_highs = brokenaxes(xlims=plot_lims, subplot_spec=axarr[2,1])


    for i in range(len(summary_df)):
        row = summary_df.iloc[i]
        axis_group = row['axis_group']

        if row['key'] == 'lowm_highs':
            ax = ax_lowm_highs
            ax.set_title('Low mass, high sSFR', color = row['color'])
        if row['key'] == 'highm_highs':
            ax = ax_highm_highs
            ax.set_title('High mass, high sSFR', color = row['color'])
        if row['key'] == 'lowm_lows':
            ax = ax_lowm_lows
            ax.set_title('Low mass, mid sSFR', color = row['color'])
        if row['key'] == 'highm_lows':
            ax = ax_highm_lows
            ax.set_title('High mass, mid sSFR', color = row['color'])
        if row['key'] == 'lowest_lows':
            ax = ax_lowest_lows
            ax.set_title('Low mass, low sSFR', color = row['color'])
        if row['key'] == 'lowest_highs':
            ax = ax_lowest_highs
            ax.set_title('High mass, low sSFR', color = row['color'])
        
        ar_df = ascii.read(imd.axis_cluster_data_dir + f'/{savename}/{savename}_group_dfs/{axis_group}_df.csv').to_pandas()


        if row['shape'] == '+': 
            color = 'red'
            label = 'Axis Ratio < 0.4'
        if row['shape'] == 'd':
            color = 'mediumseagreen'
            label = '0.4 < Axis Ratio < 0.7'
        if row['shape'] == 'o':
            color = 'blue'
            label = '0.7 < Axis Ratio'

        # compute uncertainties
        ar_df['metal_err_high'] = ar_df['u68_logoh_pp_n2'] - ar_df['logoh_pp_n2']
        ar_df['metal_err_low'] = ar_df['logoh_pp_n2'] - ar_df['l68_logoh_pp_n2']  
        
       
        # Plot 
        upper_lim_idx = ar_df['n2flag_metals'] == 1

        # Errors on non-lower-lim points
        yerrs_low = [ar_df['metal_err_low'][~upper_lim_idx].iloc[k] for k in range(len(ar_df[~upper_lim_idx]))]
        yerrs_high = [ar_df['metal_err_high'][~upper_lim_idx].iloc[k] for k in range(len(ar_df[~upper_lim_idx]))]
        yerrs = (yerrs_low, yerrs_high)

        ax.errorbar(ar_df['use_ratio'][~upper_lim_idx], ar_df['logoh_pp_n2'][~upper_lim_idx], xerr=ar_df['err_use_ratio'][~upper_lim_idx], yerr = yerrs, color=color, label = label, marker='o', ls='None') 
        ax.errorbar(ar_df['use_ratio'][upper_lim_idx], ar_df['logoh_pp_n2'][upper_lim_idx], xerr=ar_df['err_use_ratio'][upper_lim_idx], color=color, label = label, marker='o', mfc='white', ls='None') 
        ax.set_ylim(8, 9)
        ax.set_ylabel('12 + log(O/H)')
        ax.set_xlabel('Axis Ratio')


        if i == len(summary_df)-1:
            ax.legend(bbox_to_anchor=(1.5, 4.5, 0.20, 0.15), loc='upper right')


    fig.savefig(imd.axis_cluster_data_dir + f'/{savename}/metallicty_ar.pdf')


plot_overlaid_spectra()