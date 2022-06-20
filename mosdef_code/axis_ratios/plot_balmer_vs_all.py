# Plots the balmer decrement vs a vairiety of properies
from tkinter import font
import initialize_mosdef_dirs as imd
import matplotlib.pyplot as plt
from astropy.io import ascii
import numpy as np
from matplotlib.patches import Ellipse
from ellipses_for_plotting import get_ellipse_shapes
import matplotlib as mpl
from plot_vals import *

def plot_balmer_vs_all(save_name):
    """Plots balmer decrement vs a variety of measured galaxy propertties
    
    Parameters:
    save_name (str): Directory to save the images
    
    """
    
    # Make an output folder
    imd.check_and_make_dir(imd.axis_cluster_data_dir + f'/{save_name}/balmer_plots/')

    fontsize = 14
    
    summary_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/summary.csv').to_pandas()


    fig, axarr = plt.subplots(3,3,figsize = (20,20))

    colors = ['black', 'blue', 'orange', 'mediumseagreen', 'red', 'violet', 'grey', 'pink', 'cyan', 'darkblue', 'brown', 'darkgreen']

    def plot_balmer_on_axis(ax, x_points, err_x_points='None', err_x_points_high='None', color='None', colorbar=True, use_cbar_axis=False, cbar_axis = 'None'):
        """Makes one of the plots
        
        Parameters:
        ax (matplotlib.axes): Axis to plot on
        x_points (str): x_value column name to plot
        err_x_points (str): uncertainties on x_values to plot column name, can be 'None'
        """
        for i in range(len(summary_df)):
            row = summary_df.iloc[i]

            ax.set_ylim(2.7, 5.5)
            ax_y_len = 5.5-2.7
            if x_points == 'metallicity_median':
                ax.set_xlim(8.2, 8.8)
                ax_x_len = 0.6
                xlabel = 'Median Metallicity'
            elif x_points == 'log_mass_median':
                ax.set_xlim(9.5, 10.5)
                ax_x_len = 1.0
                xlabel = 'Median ' + stellar_mass_label
            elif x_points == 'log_use_sfr_median':
                ax.set_xlim(0.7, 1.9)
                ax_x_len = 1.2
                xlabel =  'Median ' + sfr_label
            elif x_points == 'av_median':
                ax.set_xlim(0.2, 0.9)
                ax_x_len = 0.7
                xlabel = 'Median AV'
            elif x_points == 'beta_median':
                ax.set_xlim(-1.8, -1.1)
                ax_x_len = 0.7
                xlabel = 'Median Beta'
            elif x_points == 'log_use_ssfr_median':
                ax.set_xlim(-9.4, -8.2)
                ax_x_len = 1.2
                xlabel = 'Median ' + ssfr_label
            elif x_points == 're_median':
                ax.set_xlim(0, 0.75)
                ax_x_len = 0.75
                xlabel = 'Median re'
            
            else:
                ax_x_len = 1
                xlabel = x_points

            if color=='mass':
                cmap = mpl.cm.inferno
                norm = mpl.colors.Normalize(vmin=9, vmax=11) 
                rgba = cmap(norm(row['log_mass_median']))
            elif color=='sfr':
                cmap = mpl.cm.viridis
                norm = mpl.colors.Normalize(vmin=0, vmax=2) 
                rgba = cmap(norm(row['log_use_sfr_median']))
            else:
                cmap = mpl.cm.inferno
                norm = mpl.colors.Normalize(vmin=-9.3, vmax=-8.1) 
                rgba = cmap(norm(row['log_use_ssfr_median']))

            ellipse_width, ellipse_height = get_ellipse_shapes(ax_x_len, ax_y_len, row['shape'])

            if err_x_points=='None':
                xerr=None
            else:
                if err_x_points_high=='None':
                    xerr=row[err_x_points]
                else:
                    xerr=np.array([[row[err_x_points], row[err_x_points_high]]]).T
            ax.errorbar(row[x_points], row['balmer_dec'], yerr=np.array([[row['err_balmer_dec_low'], row['err_balmer_dec_high']]]).T, xerr=xerr, color=rgba, marker='None', ls='None')
            ax.add_artist(Ellipse((row[x_points], row['balmer_dec']), ellipse_width, ellipse_height, facecolor=rgba))
            ax.set_xlabel(xlabel, fontsize=fontsize)
            ax.set_ylabel(balmer_label, fontsize=fontsize)
        
        if colorbar==True:
            if use_cbar_axis==False:
                cbar_axis = ax
                cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=cbar_axis, fraction=0.046, pad=0.04)
            else:
                cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_axis, fraction=0.046, pad=0.04)
            cbar.set_label('log_ssfr', fontsize=fontsize)
            if color=='mass':
                cbar.set_label(stellar_mass_label, fontsize=18)
            if color=='sfr':
                cbar.set_label(sfr_label, fontsize=18)
        ax.tick_params(labelsize=12)
        ax.set_aspect(ellipse_width/ellipse_height)
        
            
    

    plot_balmer_on_axis(axarr[0,0], 'log_mass_median')
    plot_balmer_on_axis(axarr[0,1], 'metallicity_median', 'err_metallicity_median_low', 'err_metallicity_median_high')
    plot_balmer_on_axis(axarr[0,2], 'log_use_sfr_median')
    plot_balmer_on_axis(axarr[1,2], 'log_use_ssfr_median')
    plot_balmer_on_axis(axarr[1,0], 'av_median', 'err_av_median')
    plot_balmer_on_axis(axarr[1,1], 'beta_median', 'err_beta_median')
    plot_balmer_on_axis(axarr[2,0], 're_median', 'err_re_median')

    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/balmer_plots/balmer_plots.pdf')

    ## PAPER FIGURE
    # fig, axarr = plt.subplots(1, 2, figsize=(15,8))
    # ax_balmer_mass = axarr[0]
    # ax_balmer_ssfr = axarr[1]
    # fig.subplots_adjust(right=0.85)
    # ax_cbar = fig.add_axes([0.90, 0.2, 0.02, 0.60])
    fig = plt.figure(figsize=(17, 8))
    ax_balmer_mass = fig.add_axes([0.01, 0.2, 0.45, 0.6])
    ax_balmer_ssfr = fig.add_axes([0.50, 0.2, 0.45, 0.6])
    ax_cbar_mass = fig.add_axes([0.40, 0.2, 0.02, 0.60])
    ax_cbar_ssfr = fig.add_axes([0.89, 0.2, 0.02, 0.60])
    plot_balmer_on_axis(ax_balmer_mass, 'log_mass_median', color='sfr', use_cbar_axis=True, cbar_axis=ax_cbar_mass)
    plot_balmer_on_axis(ax_balmer_ssfr, 'log_use_sfr_median', color='mass', use_cbar_axis=True, cbar_axis = ax_cbar_ssfr)
    ax_balmer_mass.set_xlabel(stellar_mass_label, fontsize=18)
    ax_balmer_ssfr.set_xlabel(sfr_label, fontsize=18)
    ax_balmer_mass.set_ylabel(balmer_label, fontsize=18)
    ax_balmer_ssfr.set_ylabel(balmer_label, fontsize=18)
    ax_balmer_mass.tick_params(labelsize=16)
    ax_balmer_ssfr.tick_params(labelsize=16)
    ax_cbar_mass.tick_params(labelsize=16)
    ax_cbar_ssfr.tick_params(labelsize=16)
    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/balmer_plots/balmer_ssfr_mass_color.pdf',bbox_inches='tight')




plot_balmer_vs_all('both_sfms_4bin_median_2axis_boot100')
# plot_balmer_vs_all('both_sfms_4bin_median_2axis_boot100_retest')