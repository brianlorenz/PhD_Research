# Plots the balmer decrement vs a vairiety of properies
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


def plot_balmer_sfr_metallicity():
    cluster_summary_df = ascii.read(imd.loc_cluster_summary_df).to_pandas()
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
    balmer_av = cluster_summary_df['balmer_av_with_limit']
    
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