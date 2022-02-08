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

def plot_metals(savename):
    """Make the plot
    
    """
    summary_df = ascii.read(imd.axis_cluster_data_dir + f'/{savename}/summary.csv').to_pandas()
    

    n_rows = int(len(summary_df) / 6)
    if len(summary_df) == 8:
        n_rows = 2
    fig, axarr = plt.subplots(n_rows, 2, figsize=(12, 6+3*n_rows))
        

    if n_rows == 1:
        ax_0 = axarr[0,0]
        ax_1 = axarr[0,1]
    if n_rows > 1:
        ax_0 = axarr[1,0]
        ax_1 = axarr[0,0]
        ax_2 = axarr[1,1]
        ax_3 = axarr[0,1]
    if n_rows > 2:
        ax_0 = axarr[2,0]
        ax_1 = axarr[1,0]
        ax_2 = axarr[0,0]
        ax_3 = axarr[2,1]
        ax_4 = axarr[1,1]
        ax_5 = axarr[0,1]

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

        if row['key'] == 'sorted0':
            ax = ax_0
            ax.set_title('sorted0', color = row['color'], fontsize=14)
        if row['key'] == 'sorted1':
            ax = ax_1
            ax.set_title('sorted1', color = row['color'], fontsize=14)
        if row['key'] == 'sorted2':
            ax = ax_2
            ax.set_title('sorted2', color = row['color'], fontsize=14)
        if row['key'] == 'sorted3':
            ax = ax_3
            ax.set_title('sorted3', color = row['color'], fontsize=14)
        if row['key'] == 'sorted4':
            ax = ax_4
            ax.set_title('sorted4', color = row['color'], fontsize=14)
        if row['key'] == 'sorted5':
            ax = ax_5
            ax.set_title('sorted5', color = row['color'], fontsize=14)
        
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
        ax.set_ylabel('12 + log(O/H)', fontsize=14)
        ax.set_xlabel('Axis Ratio', fontsize=14)


        if i == len(summary_df)-1:
            ax.legend(bbox_to_anchor=(1.5, 4.5, 0.20, 0.15), loc='upper right')

        ax.tick_params(labelsize=12, size=8)

    fig.savefig(imd.axis_cluster_data_dir + f'/{savename}/metallicty_ar.pdf')
    plt.close('all')

# plot_metals(savename='halpha_ssfr_4bin_mean_shifted')