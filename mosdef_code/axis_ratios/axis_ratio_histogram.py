# Plot a histogram of the axis ratios of the galaxies being used
from axis_ratio_funcs import filter_ar_df, read_filtered_ar_df
import numpy as np
import pandas as pd
from astropy.io import ascii
import matplotlib.pyplot as plt
import matplotlib as mpl
import initialize_mosdef_dirs as imd
import matplotlib.patheffects as path_effects
from plot_vals import *


def read_rodriguez_data():
    data_loc = imd.mosdef_dir + '/axis_ratio_data/rodriguez2013_weighted_data.csv'
    rod_df = ascii.read(data_loc).to_pandas()
    xvals = np.arange(0, 1, 0.025)
    rod_df = rod_df.rename(columns={"col1": "a", "col2": "yvals"})
    rod_df['xvals'] = xvals
    rod_df['normalized_yvals'] = rod_df['yvals'] / np.max(rod_df['yvals'])
    return rod_df

def read_law_data():
    data_loc = imd.mosdef_dir + '/axis_ratio_data/law2011_data.csv'
    law_df = ascii.read(data_loc).to_pandas()
    xvals = np.arange(0, 1, 0.1)
    law_df = law_df.rename(columns={"col1": "a", "col2": "yvals"})
    law_df['xvals'] = xvals
    return law_df


def plot_ar_hist(use_column = 'use_ratio',  mass_split='False'):
    '''Makes an axis ratio hisogram

    Parameters: 
    use_column (str): Column to retrieve the axis ratios from
    mass_split(str): Set to "True" to make 2 panels by mass
    '''

    ar_df = read_filtered_ar_df()
    bins = np.arange(0, 1, 0.05)

    rod_df = read_rodriguez_data()
    law_df = read_law_data()

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
        def get_rodscale():
            n_gals = len(ar_df)
            n_rod = np.sum(rod_df['normalized_yvals'])
            rodscale = n_gals / n_rod
            return rodscale*2
        rodscale = get_rodscale()
        ax.plot(rod_df['xvals'], rod_df['normalized_yvals']*rodscale, color='white', ls='-', lw=6, marker='None')
        ax.plot(rod_df['xvals'], rod_df['normalized_yvals']*rodscale, color=dark_color, ls='-', lw=4, marker='None', label="Rodríguez+ 2013, n=92923, low redshift")
        bins_law = np.arange(0, 1, 0.1)
        # ax.plot(bins_law+0.05, law_df['yvals']*1.5, color='white', ls='-', lw=6, marker='None')
        # ax.plot(bins_law+0.05, law_df['yvals']*1.5, color='red', ls='-', lw=4, marker='None', label='Law+ 2011, n=306, 1.5<z<3.6')
        # ax.bar(bins_law+0.05, law_df['yvals']*1.5, color='red', alpha=0.5, width=0.1)
        # ax.hist(law_df['yvals'], bins=bins_law, color='red', alpha=0.5)
        ax.legend(fontsize=14)
        axarr = [ax]
        savename = ''
        ax.set_ylim(0, 60)
    
    for ax in axarr:
        # ax.axvline(0.4, ls='--', color='red')
        ax.axvline(0.55, ls='--', color='red')

        ax.set_xlabel('Axis Ratio', fontsize=single_column_axisfont) 
        ax.set_ylabel('Number of Galaxies', fontsize=single_column_axisfont)
        ax.tick_params(labelsize=single_column_ticksize)
        # ax.text(-0.10, -10, 'Edge-on', fontsize=single_column_axisfont, zorder=100)
        # ax.text(0.92, -10, 'Face-on', fontsize=single_column_axisfont, zorder=100)
        ax.set_xlim(-0.05, 1.05)
        
    ax.set_aspect(1/60)
    scale_aspect(ax)
    fig.savefig(imd.axis_output_dir + f'/ar_histogram_{use_column}{savename}.pdf',bbox_inches='tight')


# plot_ar_hist()
# plot_ar_hist(use_column='F125_axis_ratio')
# plot_ar_hist(use_column='F140_axis_ratio')
# plot_ar_hist(use_column='F160_axis_ratio')
# plot_ar_hist(use_column='F160_axis_ratio', mass_split='True')
# plot_ar_hist(use_column='F160_axis_ratio', mass_split='AndSSFR')



def compare_ar_measurements(col1, err_col1, col2, err_col2):
    '''Makes an axis ratio plot comparing two methods of determining ar

    Parameters: 
    col1 (str): Column to retrieve the axis ratios from
    col1 (str): Column to retrieve the axis ratios from
    '''

    # ar_df = ascii.read(imd.loc_axis_ratio_cat).to_pandas()
    ar_df = read_filtered_ar_df()

    cmap = mpl.cm.viridis 
    norm = mpl.colors.Normalize(vmin=1.3, vmax=2.8) 
    
    fig, ax = plt.subplots(figsize=(9,8))

    ar_df = ar_df[ar_df[col1]>-90]
    ar_df = ar_df[ar_df[col2]>-90]


    for i in range(len(ar_df)):
        row = ar_df.iloc[i]
        rgba = cmap(norm(row['z']))

        ax.errorbar(row[col1], row[col2], xerr=row[err_col1], yerr=row[ err_col2], marker='o', ls='None', color=rgba, zorder=2)
        # ax.plot(row[col1], row[col2], marker='o', ls='None', color=rgba, zorder=2)
        
    #1-to-1 line
    ax.plot((-1,2), (-1, 2), color='red', ls='-', zorder=1, label='1-1 line')
    # Lines that capture 16th and 84th percentiles
    differences = ar_df[col1]-ar_df[col2]
    offsets = np.percentile(differences, [16,84])
    x = np.linspace(-0.5, 1.5, 100)
    y0 = x+offsets[0]
    y1 = x+offsets[1]
    ax.plot(x, y0, color='red', ls='--', zorder=4, label='1$\\sigma$ region')
    ax.plot(x, y1, color='red', ls='--', zorder=4)
    ax.legend(loc=0, fontsize=single_column_axisfont-2)

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Redshift', fontsize=single_column_axisfont)
    cbar.ax.tick_params(labelsize=single_column_ticksize)

    ax.set_xlabel(f'{col1[0:4]}W Axis Ratio', fontsize=single_column_axisfont) 
    ax.set_ylabel(f'{col2[0:4]}W Axis Ratio', fontsize=single_column_axisfont)
    ax.tick_params(labelsize=single_column_ticksize)
    # ax.text(-0.07, -10, 'Edge-on', fontsize=single_column_axisfont, zorder=100)
    # ax.text(0.95, -10, 'Face-on', fontsize=single_column_axisfont, zorder=100)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect(1)
    scale_aspect(ax)
    fig.savefig(imd.axis_output_dir + f'/ar_compare_{col1}_{col2}.pdf',bbox_inches='tight')

# compare_ar_measurements('F125_axis_ratio', 'F125_err_axis_ratio', 'F160_axis_ratio', 'F160_err_axis_ratio')
plot_ar_hist(use_column = 'use_ratio')

# compare_ar_measurements('use_ratio', 'err_use_ratio', 'F160_axis_ratio', 'F160_err_axis_ratio')
# compare_ar_measurements('use_ratio', 'err_use_ratio', 'F125_axis_ratio', 'F125_err_axis_ratio')
