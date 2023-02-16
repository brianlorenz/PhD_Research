# Plots the balmer decrement vs a vairiety of properies
from tkinter import font
from axis_ratio_funcs import read_filtered_ar_df
import initialize_mosdef_dirs as imd
import matplotlib.pyplot as plt
from astropy.io import ascii
import numpy as np
from matplotlib.patches import Ellipse
from ellipses_for_plotting import get_ellipse_shapes
import matplotlib as mpl
from plot_vals import *
from a_balmer_to_balmer_dec import convert_attenuation_to_dec
from read_sdss import read_and_filter_sdss

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

    axis_fontsize = 24

    def plot_balmer_on_axis(ax, x_points, err_x_points='None', err_x_points_high='None', color='None', colorbar=True, use_cbar_axis=False, cbar_axis = 'None', use_balmer_av=False):
        """Makes one of the plots
        
        Parameters:
        ax (matplotlib.axes): Axis to plot on
        x_points (str): x_value column name to plot
        err_x_points (str): uncertainties on x_values to plot column name, can be 'None'
        use_balmer_av (boolean)= plot the balmer AV rather than balmer dec
        """
        for i in range(len(summary_df)):
            row = summary_df.iloc[i]

            if x_points == 'metallicity_median':
                ax.set_xlim(8.2, 8.8)
                ax_x_len = 0.6
                xlabel = 'Median Metallicity'
            elif x_points == 'log_mass_median':
                ax.set_xlim(9.5, 10.5)
                ax_x_len = 1.0
                xlabel = 'Median ' + stellar_mass_label
            elif x_points == 'log_use_sfr_median':
                ax.set_xlim(0.7, 2.0)
                ax_x_len = 1.3
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
                cmap = mpl.cm.inferno
                norm = mpl.colors.Normalize(vmin=0, vmax=2) 
                rgba = cmap(norm(row['log_use_sfr_median']))
            else:
                cmap = mpl.cm.inferno
                norm = mpl.colors.Normalize(vmin=-9.3, vmax=-8.1) 
                rgba = cmap(norm(row['log_use_ssfr_median']))

            if use_balmer_av == False:
                balmer_str = 'balmer_dec'
                ax.set_ylim(2.7, 6.0)
                ax_y_len = 6.0-2.7
            else:
                balmer_str = 'balmer_av'
                ax.set_ylim(0, 2.5)
                ax_y_len = 2.5

            ellipse_width, ellipse_height = get_ellipse_shapes(ax_x_len, ax_y_len, row['shape'])

            if err_x_points=='None':
                xerr=None
            else:
                if err_x_points_high=='None':
                    xerr=row[err_x_points]
                else:
                    xerr=np.array([[row[err_x_points], row[err_x_points_high]]]).T
            
            ax.errorbar(row[x_points], row[balmer_str], yerr=np.array([[row[f'err_{balmer_str}_low'], row[f'err_{balmer_str}_high']]]).T, xerr=xerr, color=rgba, marker='None', ls='None')
            
            zorder = 10-i
            ax.add_artist(Ellipse((row[x_points], row[balmer_str]), ellipse_width*1.1, ellipse_height*1.1, facecolor='white', edgecolor='white', zorder=zorder))
            ax.add_artist(Ellipse((row[x_points], row[balmer_str]), ellipse_width, ellipse_height, facecolor=rgba, edgecolor='black', zorder=zorder))
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
                cbar.set_label(stellar_mass_label, fontsize=axis_fontsize)
            if color=='sfr':
                cbar.set_label(sfr_label, fontsize=axis_fontsize)
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
    fig, axarr = plt.subplots(1, 2, figsize=(15,8))
    ax_balmer_mass = axarr[0]
    ax_balmer_ssfr = axarr[1]
    fig.subplots_adjust(right=0.85)
    ax_cbar = fig.add_axes([0.90, 0.2, 0.02, 0.60])
    fig = plt.figure(figsize=(17, 8))
    ax_balmer_mass = fig.add_axes([0.01, 0.2, 0.45, 0.6])
    ax_balmer_ssfr = fig.add_axes([0.50, 0.2, 0.45, 0.6])
    ax_cbar_mass = fig.add_axes([0.40, 0.2, 0.02, 0.60])
    ax_cbar_ssfr = fig.add_axes([0.89, 0.2, 0.02, 0.60])
    plot_balmer_on_axis(ax_balmer_mass, 'log_mass_median', color='sfr', use_cbar_axis=True, cbar_axis=ax_cbar_mass)
    plot_balmer_on_axis(ax_balmer_ssfr, 'log_use_sfr_median', color='mass', use_cbar_axis=True, cbar_axis = ax_cbar_ssfr)
    ax_balmer_mass.set_xlabel(stellar_mass_label, fontsize=axis_fontsize)
    ax_balmer_ssfr.set_xlabel(sfr_label, fontsize=axis_fontsize)
    ax_balmer_mass.set_ylabel(balmer_label, fontsize=axis_fontsize)
    ax_balmer_ssfr.set_ylabel(balmer_label, fontsize=axis_fontsize)
    ax_balmer_mass.tick_params(labelsize=axis_fontsize)
    ax_balmer_ssfr.tick_params(labelsize=axis_fontsize)
    ax_cbar_mass.tick_params(labelsize=axis_fontsize)
    ax_cbar_ssfr.tick_params(labelsize=axis_fontsize)
    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/balmer_plots/balmer_ssfr_mass_color.pdf',bbox_inches='tight')
    plt.close('all')


    def garn_best_curve(log_mass):
        mass = 10**log_mass
        x = np.log10(mass / 10**10)
        A_Balmer = 0.91 + 0.77*x + 0.11*x**2 + (-0.09*x**3)
        balmer_dec = convert_attenuation_to_dec(A_Balmer)
        return balmer_dec
    garn_masses = np.arange(9, 11, 0.02)
    garn_balmer_decs = garn_best_curve(garn_masses)
    


    def battisti_curve():
        battisti_df = ascii.read(imd.mosdef_dir + '/axis_ratio_data/Battisti_2021_data.csv').to_pandas()
        battisti_df.rename(columns = {'col1':'log_mass', 'col2':'tau_balmer'}, inplace = True)
        battisti_df['A_Balmer'] = 1.086*battisti_df['tau_balmer']
        battisti_df['balmer_dec'] = convert_attenuation_to_dec(battisti_df['A_Balmer'])
        return battisti_df

    battisti_df = battisti_curve()
    sdss_balmer_df = ascii.read(imd.mosdef_dir + '/axis_ratio_data/sdss_decs.csv').to_pandas()
    sdss_balmer_df = sdss_balmer_df.rename(columns={'col1': 'mass', 'col2': 'balmer_dec'})
    fig = plt.figure(figsize=(8, 8))
    ax_balmer_mass = fig.add_axes([0.01, 0.01, 0.9, 0.9])
    ax_cbar_mass = fig.add_axes([0.92, 0.01, 0.04, 0.9])
    # ax_balmer_mass.plot(sdss_balmer_df['mass'], sdss_balmer_df['balmer_dec'], color='black', marker='o', markersize=10, label='SDSS median, z~0')
    # ax_balmer_mass.plot(garn_masses, garn_balmer_decs, color='grey', marker='None', markersize=10, label='Garn & Best 2010')
    # ax_balmer_mass.plot(battisti_df['log_mass'], battisti_df['balmer_dec'], color='red', marker='None', markersize=10, label='Battisti 2021')

    # Add SDSS galaxies
    sdss_df = read_and_filter_sdss()
    sdss_df = sdss_df[sdss_df['balmer_dec']>2.7]
    sdss_df = sdss_df[sdss_df['balmer_dec']<6.0]
    sdss_df = sdss_df[sdss_df['log_mass']>9.5]
    sdss_df = sdss_df[sdss_df['log_mass']<10.5]
    # sdss_df = sdss_df[sdss_df['LGM_FIB_P50']>9.5]
    # sdss_df = sdss_df[sdss_df['LGM_FIB_P50']<10.5]
    # ax.plot(extra_df['log_mass'], extra_df['balmer_dec'], ls='None', marker='o', color='black', alpha=0.01)
    # ax_balmer_mass.hist2d(sdss_df['LGM_FIB_P50'], sdss_df['balmer_dec'], bins=(100, 100), cmap=plt.cm.gray_r)
    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
        return new_cmap
    cmap = plt.get_cmap('gray_r')
    new_cmap = truncate_colormap(cmap, 0, 0.7)
    # ax_balmer_mass.hexbin(sdss_df['log_mass'], sdss_df['balmer_dec'], gridsize=20, cmap=plt.cm.gray_r, alpha = 0.75)
    ax_balmer_mass.hexbin(sdss_df['log_mass'], sdss_df['balmer_dec'], gridsize=20, cmap=new_cmap)
    ax_balmer_mass.plot([0], [0], marker='h', label='SDSS, z~0', ls='None', color='gray', markersize=25)

    plot_balmer_on_axis(ax_balmer_mass, 'log_mass_median', color='sfr', use_cbar_axis=True, cbar_axis=ax_cbar_mass)
    ax_balmer_mass.set_xlabel(stellar_mass_label, fontsize=axis_fontsize)
    ax_balmer_mass.set_ylabel(balmer_label, fontsize=axis_fontsize)
    ax_balmer_mass.tick_params(labelsize=axis_fontsize)
    ax_cbar_mass.tick_params(labelsize=axis_fontsize)
    ax_balmer_mass.legend(fontsize=16, loc=2)
    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/balmer_plots/balmer_mass.pdf',bbox_inches='tight')
    plt.close('all')

    ## PAPER FIGURE but with SFR and metallicity
    # fig, axarr = plt.subplots(1, 2, figsize=(15,8))
    # ax_balmer_mass = axarr[0]
    # ax_balmer_ssfr = axarr[1]
    # fig.subplots_adjust(right=0.85)
    # ax_cbar = fig.add_axes([0.90, 0.2, 0.02, 0.60])
    fig = plt.figure(figsize=(17, 8))
    ax_balmer_metallicity = fig.add_axes([0.01, 0.2, 0.45, 0.6])
    ax_balmer_sfr = fig.add_axes([0.50, 0.2, 0.45, 0.6])
    ax_cbar_sfr = fig.add_axes([0.40, 0.2, 0.02, 0.60])
    ax_cbar_metallicity = fig.add_axes([0.89, 0.2, 0.02, 0.60])
    plot_balmer_on_axis(ax_balmer_sfr, 'log_use_sfr_median', color='mass', use_cbar_axis=True, cbar_axis=ax_cbar_sfr, use_balmer_av=True)
    plot_balmer_on_axis(ax_balmer_metallicity, 'metallicity_median', color='mass', use_cbar_axis=True, cbar_axis = ax_cbar_metallicity, use_balmer_av=True)
    # plot_balmer_on_axis(ax_balmer_sfr, 'log_use_sfr_median', color='sfr', use_cbar_axis=True, cbar_axis=ax_cbar_sfr, use_balmer_av=True)
    # plot_balmer_on_axis(ax_balmer_metallicity, 'metallicity_median', color='sfr', use_cbar_axis=True, cbar_axis = ax_cbar_metallicity, use_balmer_av=True)
    ax_balmer_sfr.set_xlabel(sfr_label, fontsize=axis_fontsize)
    ax_balmer_metallicity.set_xlabel(metallicity_label, fontsize=axis_fontsize)
    ax_balmer_sfr.set_ylabel(balmer_av_label, fontsize=axis_fontsize)
    ax_balmer_metallicity.set_ylabel(balmer_av_label, fontsize=axis_fontsize)
    ax_balmer_sfr.tick_params(labelsize=axis_fontsize)
    ax_balmer_metallicity.tick_params(labelsize=axis_fontsize)
    ax_cbar_sfr.tick_params(labelsize=axis_fontsize)
    ax_cbar_metallicity.tick_params(labelsize=axis_fontsize)
    ax_balmer_sfr.axhline(0.85, ls='--', color='#8E248C')
    ax_balmer_metallicity.axhline(0.85, ls='--', color='#8E248C')
    ax_balmer_sfr.axhline(1.9, ls='--', color='#FF640A')
    ax_balmer_metallicity.axhline(1.9, ls='--', color='#FF640A')
    plt.setp(ax_balmer_metallicity.get_yticklabels()[0], visible=False)   
    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/balmer_plots/balmer_sfr_metallicity.pdf',bbox_inches='tight')
    plt.close('all')



# plot_balmer_vs_all('norm_1_sn5_filtered')
