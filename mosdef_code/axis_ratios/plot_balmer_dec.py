import numpy as np
from astropy.io import ascii
import matplotlib as mpl
import initialize_mosdef_dirs as imd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from ellipses_for_plotting import get_ellipse_shapes
import matplotlib.gridspec as gridspec
from plot_vals import *


def plot_balmer_dec(save_name, n_groups, split_by, y_var = 'balmer_dec', color_var='log_ssfr', background_points=True, axarr=['None'], mass_ax='None', fig='None'):
    '''Makes the balmer decrement plots. Now can also do AV and Beta instead of balmer dec on the y-axis

    Parameters:
    save_name (str): Folder to pull data from and save to
    n_groups (int): Number of axis ratio groups
    split_by (str): Column name that was used for splitting into groups in y-axis, used for coloring
    y_var (str): What to plot on the y-axis - either "balmer_dec", "av", or "beta"
    color_var (str): Colorbar variable for the plots
    background_points (boolean): Set to true to plot all galaxies in the background in grey
    axarr (matplotlib axis): leave None to generate one, otherwise plot on the one provided. Len(2). For split mass plots
    ax (matplotlib axis): leave None to generate one, otherwise plot on the one provided. Len(1). For single mass plot

    '''

    # Fontsizes
    axis_fontsize = 14
    default_size = 7
    larger_size = 12


    # Axis limits
    if color_var == 'log_halpha_ssfr' or color_var == 'eq_width_ha':
        ylims = {
            'balmer_dec': (2, 10),
            'av': (0.25, 1.1),
            'beta': (-1.9, -0.95),
            'metallicity': (8.2, 8.9),
            'mips_flux': (0, 0.0063),
            'log_use_sfr': (-0.1, 2.6)
        }
    else:
        ylims = {
            'balmer_dec': (2.7, 5.5),
            'av': (0.18, 1.05),
            'beta': (-1.78, -1.1),
            'metallicity': (8.2, 8.9),
            'mips_flux': (0, 0.0063),
            'log_use_sfr': (-0.1, 2.6)
        }
    
    # Read in summary df
    summary_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/summary.csv').to_pandas()

    color_str = ''
    cmap = mpl.cm.inferno 
    if color_var=='eq_width_ha':
        # eq color map
        norm = mpl.colors.Normalize(vmin=100, vmax=500) 
    elif color_var=='log_halpha_ssfr' or color_var=='log_ssfr' or color_var=='log_use_ssfr':
        # ssfr color map
        norm = mpl.colors.Normalize(vmin=-9.3, vmax=-8.1) 
    elif color_var=='log_use_sfr':
        # ssfr color map
        norm = mpl.colors.Normalize(vmin=0, vmax=2.5) 
        color_str='_log_use_sfr_color'
    elif color_var=='metallicity':
        # metallicity color map
        norm = mpl.colors.Normalize(vmin=8.1, vmax=8.9) 
        color_str='_metal_color'
    elif color_var=='log_use_ssfr':
        # metallicity color map
        norm = mpl.colors.Normalize(vmin=8.1, vmax=8.9) 
        color_str='_log_use_ssfr_color'

 
    # Get the length of the y-axis
    y_axis_len = ylims[y_var][1] - ylims[y_var][0]

    # Figure 1 - all the balmer decs in axis ratio vs balmer dec space
    if len(axarr) == 1:
        fig, axarr = plt.subplots(1, 2, figsize=(20,8))
        made_new_axis = True
    else:
        made_new_axis = False
    ax_low_mass = axarr[0]
    ax_high_mass = axarr[1]

    for i in range(len(summary_df)):
        row = summary_df.iloc[i]

        # Set up the colormap on ssfr
        rgba = cmap(norm(row[color_var+'_median']))

        # Split into mass groups
        if row['log_mass_median'] < 10:
            ax = ax_low_mass
        elif row['log_mass_median'] >= 10:
            ax = ax_high_mass

        # Plot background points:
        if background_points==True:
            if y_var == 'balmer_dec':
                plot_var = 'balmer_dec'
            elif y_var == 'av':
                plot_var = 'AV'
            elif y_var == 'beta':
                plot_var = 'beta'
            elif y_var == 'metallicity':
                plot_var = 'None'
            elif y_var == 'log_use_sfr':
                plot_var = 'log_use_sfr'
            elif y_var == 'mips_flux':
                plot_var = 'None'
            # print(y_var)
            if plot_var!='None':
                for axis_group in range(len(summary_df)):
                    axis_group_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_group_dfs/{axis_group}_df.csv').to_pandas()
                    low_mass = axis_group_df['log_mass'] < 10
                    # ax_low_mass.plot(axis_group_df[low_mass]['use_ratio'], axis_group_df[low_mass][plot_var], color='grey', marker='o', ls='None', markersize=2, zorder=1)
                    # ax_high_mass.plot(axis_group_df[~low_mass]['use_ratio'], axis_group_df[~low_mass][plot_var], color='grey', marker='o', ls='None', markersize=2, zorder=1)

        if y_var == 'balmer_dec':
            x_cord = row['use_ratio_median']
            y_cord = row['balmer_dec']
            
            ellipse_width, ellipse_height = get_ellipse_shapes(1.1, y_axis_len, row['shape'])

            # Make the point obvoise if hbeta signal_noise is low
            if row['hbeta_snr'] < 3:
                rgba = 'skyblue'


            ax.errorbar(x_cord, y_cord, yerr=np.array([[row['err_balmer_dec_low'], row['err_balmer_dec_high']]]).T, marker='None', color=rgba)
            ax.add_artist(Ellipse((x_cord, y_cord), ellipse_width, ellipse_height, facecolor=rgba, edgecolor='black'))
            ax.set_ylabel(balmer_label, fontsize=axis_fontsize)
        elif y_var == 'av':
            x_cord = row['use_ratio_median']
            y_cord = row['av_median']

            ellipse_width, ellipse_height = get_ellipse_shapes(1.1, y_axis_len, row['shape'])

            ax.errorbar(x_cord, y_cord, yerr=row['err_av_median'], marker='None', color=rgba)
            ax.add_artist(Ellipse((x_cord, y_cord), ellipse_width, ellipse_height, facecolor=rgba, edgecolor='black', zorder=3))
            ax.set_ylabel('FAST AV', fontsize=axis_fontsize)
        elif y_var == 'beta':
            x_cord = row['use_ratio_median']
            y_cord = row['beta_median']

            ellipse_width, ellipse_height = get_ellipse_shapes(1.1, y_axis_len, row['shape'])

            ax.errorbar(x_cord, y_cord, yerr=np.array([[row['err_beta_median_low'], row['err_beta_median_high']]]).T, marker='None', color=rgba)
            ax.add_artist(Ellipse((x_cord, y_cord), ellipse_width, ellipse_height, facecolor=rgba, edgecolor='black', zorder=3))
            ax.set_ylabel('Betaphot', fontsize=axis_fontsize)
        elif y_var == 'metallicity':
            x_cord = row['use_ratio_median']
            y_cord = row['metallicity_median']

            ellipse_width, ellipse_height = get_ellipse_shapes(1.1, y_axis_len, row['shape'])

            ax.errorbar(x_cord, y_cord, yerr=np.array([[row['err_metallicity_median_low'], row['err_metallicity_median_high']]]).T, marker='None', color=rgba)
            ax.add_artist(Ellipse((x_cord, y_cord), ellipse_width, ellipse_height, facecolor=rgba, edgecolor='black', zorder=3))
            ax.set_ylabel('Metallicity', fontsize=axis_fontsize)
        elif y_var == 'mips_flux':
            x_cord = row['use_ratio_median']
            y_cord = row['mips_flux_median']

            ellipse_width, ellipse_height = get_ellipse_shapes(1.1, y_axis_len, row['shape'])

            ax.errorbar(x_cord, y_cord, yerr=row['err_mips_flux_median'], marker='None', color=rgba)
            ax.add_artist(Ellipse((x_cord, y_cord), ellipse_width, ellipse_height, facecolor=rgba, edgecolor='black', zorder=3))
            ax.set_ylabel('MIPS Flux', fontsize=axis_fontsize)
        elif y_var == 'log_use_sfr':
            x_cord = row['use_ratio_median']
            y_cord = row['log_use_sfr_median']

            ellipse_width, ellipse_height = get_ellipse_shapes(1.1, y_axis_len, row['shape'])

            ax.errorbar(x_cord, y_cord, marker='None', color=rgba)
            ax.add_artist(Ellipse((x_cord, y_cord), ellipse_width, ellipse_height, facecolor=rgba, edgecolor='black', zorder=3))
            ax.set_ylabel('log_use_sfr', fontsize=axis_fontsize)
    


    for ax in axarr:
        ax.set_xlabel('Axis Ratio', fontsize=axis_fontsize) 
        ax.tick_params(labelsize=12)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(ylims[y_var])
        if y_var == 'balmer_dec':
            if len(axarr)==1:
                ax.text(-0.07, 1.6, 'Edge-on', fontsize=14, zorder=100)
                ax.text(0.95, 1.6, 'Face-on', fontsize=14, zorder=100)
    
    ax_low_mass.set_title('log(Stellar Mass) < 10', fontsize=axis_fontsize)
    ax_high_mass.set_title('log(Stellar Mass) > 10', fontsize=axis_fontsize)
    
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axarr)
    cbar.set_label(color_var, fontsize=axis_fontsize)
    
    if made_new_axis==True:
        fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/{y_var}_vs_ar{color_str}.pdf', bbox_inches='tight')
    else:
        pass
    

    # Figure 2 - Decrement vs mass
    if mass_ax == 'None':
        fig, ax = plt.subplots(figsize=(9,8))
        made_new_axis == True
    else:
        ax = mass_ax
        made_new_axis == False


    for i in range(len(summary_df)):
        row = summary_df.iloc[i]

        # Set up the colormap on ssfr
        rgba = cmap(norm(row[color_var+'_median']))


        if y_var == 'balmer_dec':
            x_cord = row['log_mass_median']
            y_cord = row['balmer_dec']
            
            # Make the point obvoise if hbeta signal_noise is low
            if row['hbeta_snr'] < 3:
                rgba = 'skyblue'

            ellipse_width, ellipse_height = get_ellipse_shapes(1.5, y_axis_len, row['shape'])
            
            ax.errorbar(x_cord, y_cord, yerr=np.array([[row['err_balmer_dec_low'], row['err_balmer_dec_high']]]).T, marker='None', color=rgba)
            ax.add_artist(Ellipse((x_cord, y_cord), ellipse_width, ellipse_height, facecolor=rgba, edgecolor='black', zorder=3))
            ax.set_ylabel(balmer_label, fontsize=axis_fontsize)
        elif y_var == 'av':
            x_cord = row['log_mass_median']
            y_cord = row['av_median']

            ellipse_width, ellipse_height = get_ellipse_shapes(1.5, y_axis_len, row['shape'])

            ax.errorbar(x_cord, y_cord, yerr=row['err_av_median'], marker='None', color=rgba)
            ax.add_artist(Ellipse((x_cord, y_cord), ellipse_width, ellipse_height, facecolor=rgba, edgecolor='black', zorder=3))
            ax.set_ylabel('FAST AV', fontsize=axis_fontsize)
        elif y_var == 'beta':
            x_cord = row['log_mass_median']
            y_cord = row['beta_median']
            
            ellipse_width, ellipse_height = get_ellipse_shapes(1.5, y_axis_len, row['shape'])

            ax.errorbar(x_cord, y_cord, yerr=np.array([[row['err_beta_median_low'], row['err_beta_median_high']]]).T, marker='None', color=rgba)
            ax.add_artist(Ellipse((x_cord, y_cord), ellipse_width, ellipse_height, facecolor=rgba, edgecolor='black', zorder=3))
            ax.set_ylabel('Betaphot', fontsize=axis_fontsize)
        elif y_var == 'metallicity':
            x_cord = row['log_mass_median']
            y_cord = row['metallicity_median']
            
            ellipse_width, ellipse_height = get_ellipse_shapes(1.5, y_axis_len, row['shape'])

            ax.errorbar(x_cord, y_cord, yerr=np.array([[row['err_metallicity_median_low'], row['err_metallicity_median_high']]]).T, marker='None', color=rgba)
            ax.add_artist(Ellipse((x_cord, y_cord), ellipse_width, ellipse_height, facecolor=rgba, edgecolor='black', zorder=3))
            ax.set_ylabel('Metallcity', fontsize=axis_fontsize)
        elif y_var == 'mips_flux':
            x_cord = row['log_mass_median']
            y_cord = row['mips_flux_median']
            
            ellipse_width, ellipse_height = get_ellipse_shapes(1.5, y_axis_len, row['shape'])

            ax.errorbar(x_cord, y_cord, yerr=row['err_mips_flux_median'], marker='None', color=rgba)
            ax.add_artist(Ellipse((x_cord, y_cord), ellipse_width, ellipse_height, facecolor=rgba, edgecolor='black', zorder=3))
            ax.set_ylabel('MIPS Flux', fontsize=axis_fontsize)
        elif y_var == 'log_use_sfr':
            x_cord = row['use_ratio_median']
            y_cord = row['log_use_sfr_median']

            ellipse_width, ellipse_height = get_ellipse_shapes(1.1, y_axis_len, row['shape'])

            ax.errorbar(x_cord, y_cord, marker='None', color=rgba)
            ax.add_artist(Ellipse((x_cord, y_cord), ellipse_width, ellipse_height, facecolor=rgba, edgecolor='black', zorder=3))
            ax.set_ylabel('log_use_sfr', fontsize=axis_fontsize)

    if made_new_axis==True:
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(color_var, fontsize=axis_fontsize)
        if color_var == 'log_use_sfr':
            cbar.ax.tick_params(labelsize=18)
            cbar.set_label(sfr_label, fontsize=18)
    ax.set_xlabel(stellar_mass_label, fontsize=axis_fontsize) 
    
    ax.tick_params(labelsize=12)
    ax.set_xlim(9.25, 10.75)
    ax.set_ylim(ylims[y_var])
    if made_new_axis == True:
        fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/{y_var}_vs_mass{color_str}.pdf',bbox_inches='tight')




# plot_balmer_dec('whitaker_sfms_boot100', 8, 'log_use_sfr', y_var='balmer_dec', color_var='log_use_sfr')