
import os
from scipy import stats
from astropy.io import ascii
import numpy as np
import matplotlib.pyplot as plt
import initialize_mosdef_dirs as imd
from axis_ratio_funcs import read_interp_axis_ratio, filter_ar_df, read_filtered_ar_df
from scipy.optimize import curve_fit

### OUTDATED
sfms_slope = 0.598 
sfms_yint = -4.745

# 06/03/22
sfms_lowz_slope = 0.604
sfms_lowz_yint = -4.98
sfms_highz_slope = 0.645
sfms_highz_yint = -5.115

# 06/17/22
# a_lowz_fit = -24.228
# a_highz_fit = -23.943

# 06/26/22 -- after mass correction
a_lowz_fit = -24.167
a_highz_fit = -23.874

def whitaker_sfms(mass, a):
    # a = -24.0415
    b = 4.1693
    c = -0.1638
    sfms = a + b*mass + c*mass**2
    return sfms


def find_sfms(divide_axis = False, divide_z = False, whitaker_z = False):
    '''Find the slope and intercept of the sfms in our sample
    
    Parameters:
    divide_axis: Split into axes groups and fit each sfms separately
    divide_z: Split into redshift groups and fit each sfms separately
    whitaker_z: Split into redshift groups and find the best fitting whitaker curve to each of them
    '''
    
    # ar_df = read_interp_axis_ratio()
    # ar_df = filter_ar_df(ar_df)

    ar_df = read_filtered_ar_df()

    # Add a column for ssfr
    ar_df['log_ssfr'] = np.log10((ar_df['sfr'])/(10**ar_df['log_mass']))
    ar_df['log_halpha_ssfr'] = np.log10((ar_df['halpha_sfrs'])/(10**ar_df['log_mass']))
    ar_df['log_use_ssfr'] = np.log10((ar_df['use_sfr'])/(10**ar_df['log_mass']))
    ar_df['log_use_sfr'] = np.log10(ar_df['use_sfr'])
    
    colors = ['red', 'orange', 'blue', 'black', 'brown', 'green', 'pink', 'grey', 'purple', 'cyan', 'navy', 'magenta']
    x = np.linspace(8.8, 11.2, 100)

    fig, ax = plt.subplots(figsize=(8,8))

    if divide_axis == True:
        # Divide into axis groups and fit inidividual sequences for those
        low_axis = ar_df['use_ratio'] < 0.55
        ax.plot(ar_df[low_axis]['log_mass'], ar_df[low_axis]['log_use_sfr'], color='orange', ls='None', marker='o', label='axis ratio < 0.55')
        ax.plot(ar_df[~low_axis]['log_mass'], ar_df[~low_axis]['log_use_sfr'], color='blue', ls='None', marker='o', label='axis ratio >= 0.55')
        
        fit_low = stats.linregress(ar_df[low_axis]['log_mass'], ar_df[low_axis]['log_use_sfr'])
        y1_low = fit_low.slope*x+fit_low.intercept
        plt.plot(x, y1_low, color='darkorange', ls='--', label='low fit')
        ax.text(9, 2.1, f'slope: {round(fit_low.slope, 3)}, yint: {round(fit_low.intercept, 3)}', color='darkorange')

        fit_high = stats.linregress(ar_df[~low_axis]['log_mass'], ar_df[~low_axis]['log_use_sfr'])
        y1_high = fit_high.slope*x+fit_high.intercept
        plt.plot(x, y1_high, color='darkblue', ls='--', label='high fit')
        ax.text(9, 1.9, f'slope: {round(fit_high.slope, 3)}, yint: {round(fit_high.intercept, 3)}', color='darkblue')
        fit_color = 'black'
        save_add = '_axis_split'
    elif divide_z == True:
        # Divide into axis groups and fit inidividual sequences for those
        low_z = ar_df['Z_MOSFIRE'] < 1.8
        ax.plot(ar_df[low_z]['log_mass'], ar_df[low_z]['log_use_sfr'], color='orange', ls='None', marker='o', label='redshift < 1.8')
        ax.plot(ar_df[~low_z]['log_mass'], ar_df[~low_z]['log_use_sfr'], color='blue', ls='None', marker='o', label='redshift > 1.8')
        
        fit_low = stats.linregress(ar_df[low_z]['log_mass'], ar_df[low_z]['log_use_sfr'])
        y1_low = fit_low.slope*x+fit_low.intercept
        plt.plot(x, y1_low, color='darkorange', ls='--', label='low fit')
        ax.text(9, 2.1, f'slope: {round(fit_low.slope, 3)}, yint: {round(fit_low.intercept, 3)}', color='darkorange')

        fit_high = stats.linregress(ar_df[~low_z]['log_mass'], ar_df[~low_z]['log_use_sfr'])
        y1_high = fit_high.slope*x+fit_high.intercept
        plt.plot(x, y1_high, color='darkblue', ls='--', label='high fit')
        ax.text(9, 1.9, f'slope: {round(fit_high.slope, 3)}, yint: {round(fit_high.intercept, 3)}', color='darkblue')
        fit_color = 'black'
        save_add = '_zsplit'
    elif whitaker_z == True:
        low_z = ar_df['Z_MOSFIRE'] < 1.8
        ax.plot(ar_df[low_z]['log_mass'], ar_df[low_z]['log_use_sfr'], color='orange', ls='None', marker='o', label='redshift < 1.8')
        ax.plot(ar_df[~low_z]['log_mass'], ar_df[~low_z]['log_use_sfr'], color='blue', ls='None', marker='o', label='redshift > 1.8')
    
        popt, pcov = curve_fit(whitaker_sfms, ar_df[low_z]['log_mass'], ar_df[low_z]['log_use_sfr'])
        a_lowz = popt[0]
        y1_low = whitaker_sfms(x, a_lowz)
        plt.plot(x, y1_low, color='darkorange', ls='--', label='low fit')
        ax.text(9, 2.1, f'a value: {round(a_lowz, 3)}', color='darkorange')

        popt, pcov = curve_fit(whitaker_sfms, ar_df[~low_z]['log_mass'], ar_df[~low_z]['log_use_sfr'])
        a_highz = popt[0]
        y1_high = whitaker_sfms(x, a_highz)
        plt.plot(x, y1_high, color='darkblue', ls='--', label='high fit')
        ax.text(9, 1.9, f'a value: {round(a_highz, 3)}', color='darkblue')
        fit_color = 'black'
        save_add = '_whitaker_zsplit'

    else:
        ax.plot(ar_df['log_mass'], ar_df['log_use_sfr'], color='black', ls='None', marker='o')
        fit_color = 'red'
        save_add = ''
    ax.set_xlim(8.95, 11.05)
    ax.set_ylim(-0.1, 2.6)
    
    ax.set_xlabel('log(Stellar Mass)')
    ax.set_ylabel('log(SFR)')

    #Overall sfms fit
    fit = stats.linregress(ar_df['log_mass'], ar_df['log_use_sfr'])
    y1 = fit.slope*x+fit.intercept
    plt.plot(x, y1, color=fit_color, ls='--', label='overall fit')
    ax.text(9, 2.3, f'slope: {round(fit.slope, 3)}, yint: {round(fit.intercept, 3)}', color=fit_color)

    ax.legend(loc=4)
    fig.savefig(imd.axis_output_dir + f'/sfms{save_add}.pdf')

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
# find_sfms(divide_axis=True)
# find_sfms(divide_z=True)
# find_sfms(whitaker_z=True)
# plot_sfms_bins('both_sfms_6bin_median_2axis', 12, 'log_use_sfr')
#low cut - (9.5, 0.3), (11.0, 1.9)  y = 1.07x-9.83
#high cut - (9.0, 0.5), (10.5, 2.0) y = 1.07x-8.6


