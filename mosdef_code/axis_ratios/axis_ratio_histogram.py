# Plot a histogram of the axis ratios of the galaxies being used
from axis_ratio_funcs import filter_ar_df
import numpy as np
import pandas as pd
from astropy.io import ascii
import matplotlib.pyplot as plt
import matplotlib as mpl
import initialize_mosdef_dirs as imd
import matplotlib.patheffects as path_effects


def plot_ar_hist(use_column = 'use_ratio',  mass_split='False'):
    '''Makes an axis ratio hisogram

    Parameters: 
    use_column (str): Column to retrieve the axis ratios from
    mass_split(str): Set to "True" to make 2 panels by mass
    '''

    ar_df = ascii.read(imd.loc_axis_ratio_cat).to_pandas()
    ar_df = filter_ar_df(ar_df)
    bins = np.arange(0, 1, 0.05)
    
    if mass_split=='True':
        fig, axarr = plt.subplots(1, 2, figsize=(16,8))
        ax_low = axarr[0]
        ax_high = axarr[1]
        low_mass = ar_df['log_mass'] < 10.1
        ax_low.hist(ar_df[low_mass][use_column], bins=bins, color='black')
        ax_high.hist(ar_df[~low_mass][use_column], bins=bins, color='black')
        savename = '_bymass'
        ax_low.set_ylim(0, 80)
        ax_high.set_ylim(0, 80)
    
    if mass_split=='AndSSFR':
        mass_cut = 10.1
        ssfr_cut = -8.85

        ar_df['log_ssfr'] = np.log10(ar_df['sfr']/(10**ar_df['log_mass']))
        ar_df['log_halpha_ssfr'] = np.log10(ar_df['halpha_sfrs']/(10**ar_df['log_mass']))
        ar_df = ar_df[ar_df['log_ssfr']>-15]
        ar_df = ar_df[ar_df['log_halpha_ssfr']>-15]
        print(f'Total gals = {len(ar_df)}')

        fig, axarr = plt.subplots(2, 2, figsize=(16,16))
        axarr = np.array([axarr[0,0], axarr[0,1], axarr[1,0], axarr[1,1]])
        ax_lowm_highs = axarr[0]
        ax_highm_highs = axarr[1]
        ax_lowm_lows = axarr[2]
        ax_highm_lows = axarr[3]
        low_mass = ar_df['log_mass'] < mass_cut
        high_mass = ~low_mass
        lowm_lows = ar_df[low_mass][ar_df[low_mass]['log_halpha_ssfr'] <= ssfr_cut]
        print(f'Lowm_low gals = {len(lowm_lows)}')
        lowm_highs = ar_df[low_mass][ar_df[low_mass]['log_halpha_ssfr'] > ssfr_cut]
        print(f'Lowm_high gals = {len(lowm_highs)}')
        highm_lows = ar_df[high_mass][ar_df[high_mass]['log_halpha_ssfr'] <= ssfr_cut]
        print(f'highm_low gals = {len(highm_lows)}')
        highm_highs = ar_df[high_mass][ar_df[high_mass]['log_halpha_ssfr'] > ssfr_cut]
        print(f'highm_high gals = {len(highm_highs)}')
        
        ax_lowm_highs.hist(lowm_highs[use_column], bins=bins, color='black')
        ax_lowm_lows.hist(lowm_lows[use_column], bins=bins, color='black')
        ax_highm_highs.hist(highm_highs[use_column], bins=bins, color='black')
        ax_highm_lows.hist(highm_lows[use_column], bins=bins, color='black')
        ax_lowm_highs.set_title(f'Mass <= {mass_cut}, log(ssfr) > {ssfr_cut}')
        ax_lowm_lows.set_title(f'Mass <= {mass_cut}, log(ssfr) <= {ssfr_cut}')
        ax_highm_highs.set_title(f'Mass > {mass_cut}, log(ssfr) > {ssfr_cut}')
        ax_highm_lows.set_title(f'Mass > {mass_cut}, log(ssfr) <= {ssfr_cut}')
        savename = '_bymass_halpha_ssfr'
        ax_lowm_highs.set_ylim(0, 30)
        ax_lowm_lows.set_ylim(0, 30)
        ax_highm_highs.set_ylim(0, 30)
        ax_highm_lows.set_ylim(0, 30)
        
    else:
        fig, ax = plt.subplots(figsize=(8,8))
        ax.hist(ar_df[use_column], bins=bins, color='black')
        axarr = [ax]
        savename = ''
        ax.set_ylim(0, 110)
    
    for ax in axarr:
        ax.axvline(0.4, ls='--', color='red')
        ax.axvline(0.7, ls='--', color='red')

        ax.set_xlabel('Axis Ratio', fontsize=14) 
        ax.set_ylabel('Number of Galaxies', fontsize=14)
        ax.tick_params(labelsize=12)
        ax.text(-0.07, -10, 'Edge-on', fontsize=14, zorder=100)
        ax.text(0.95, -10, 'Face-on', fontsize=14, zorder=100)
        ax.set_xlim(-0.05, 1.05)
        
    fig.savefig(imd.axis_output_dir + f'/ar_histogram_{use_column}{savename}.pdf')


plot_ar_hist()
plot_ar_hist(use_column='F125_axis_ratio')
plot_ar_hist(use_column='F140_axis_ratio')
plot_ar_hist(use_column='F160_axis_ratio')
plot_ar_hist(use_column='F160_axis_ratio', mass_split='True')
plot_ar_hist(use_column='F160_axis_ratio', mass_split='AndSSFR')



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
# compare_ar_measurements('use_ratio', 'err_use_ratio', 'F160_axis_ratio', 'F160_err_axis_ratio')
# compare_ar_measurements('use_ratio', 'err_use_ratio', 'F125_axis_ratio', 'F125_err_axis_ratio')
plot_ar_hist(use_column = 'use_ratio',  mass_split='AndSSFR')
