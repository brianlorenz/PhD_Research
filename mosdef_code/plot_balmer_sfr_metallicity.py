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
ignore_groups = imd.ignore_groups

def plot_balmer_sfr_metallicity(n_clusters, ignore_groups):
    cluster_summary_df = ascii.read(imd.loc_cluster_summary_df).to_pandas()
    fig = plt.figure(figsize=(17, 8))
    ax_balmer_metallicity = fig.add_axes([0.01, 0.2, 0.35, 0.6])
    ax_balmer_sfr = fig.add_axes([0.50, 0.2, 0.35, 0.6])
    ax_cbar_sfr = fig.add_axes([0.37, 0.2, 0.02, 0.60])
    ax_cbar_metallicity = fig.add_axes([0.86, 0.2, 0.02, 0.60])

    balmer_av = cluster_summary_df['balmer_av_with_limit']
    err_balmer_av_low = cluster_summary_df['err_balmer_av_with_limit_low']
    err_balmer_av_high = cluster_summary_df['err_balmer_av_with_limit_high']
    sfr = cluster_summary_df['computed_log_sfr_with_limit']
    metallicity = cluster_summary_df['O3N2_metallicity']
    err_metallicity_low = cluster_summary_df['err_O3N2_metallicity_low']
    err_metallicity_high = cluster_summary_df['err_O3N2_metallicity_high']
    mass = cluster_summary_df['median_log_mass']


    for groupID in range(n_clusters):
        if groupID in ignore_groups:
            continue
        
        ## PAPER FIGURE but with SFR and metallicity
        # fig, axarr = plt.subplots(1, 2, figsize=(15,8))
        # ax_balmer_mass = axarr[0]
        # ax_balmer_ssfr = axarr[1]
        # fig.subplots_adjust(right=0.85)
        # ax_cbar = fig.add_axes([0.90, 0.2, 0.02, 0.60])
        
        
        cmap = mpl.cm.inferno
        norm = mpl.colors.Normalize(vmin=9, vmax=11) 
        rgba = cmap(norm(mass.iloc[groupID]))
        
        if cluster_summary_df.iloc[groupID]['flag_balmer_lower_limit']==1:
            marker='^'
        else:
            marker='o'

        ax_balmer_sfr.errorbar(sfr.iloc[groupID], balmer_av.iloc[groupID], yerr=np.array([[err_balmer_av_low.iloc[groupID], err_balmer_av_high.iloc[groupID]]]).T, color=rgba, marker=marker)
        ax_balmer_metallicity.errorbar(metallicity.iloc[groupID], balmer_av.iloc[groupID], xerr=np.array([[err_metallicity_low.iloc[groupID], err_metallicity_high.iloc[groupID]]]).T, yerr=np.array([[err_balmer_av_low.iloc[groupID], err_balmer_av_high.iloc[groupID]]]).T, color=rgba, marker=marker)
        
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax_cbar_sfr, fraction=0.046, pad=0.04)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax_cbar_metallicity, fraction=0.046, pad=0.04)
    ax_balmer_sfr.set_xlabel(sfr_label, fontsize=single_column_axisfont)
    ax_balmer_metallicity.set_xlabel(metallicity_label, fontsize=single_column_axisfont)
    ax_balmer_sfr.set_ylabel(balmer_av_label, fontsize=single_column_axisfont)
    ax_balmer_metallicity.set_ylabel(balmer_av_label, fontsize=single_column_axisfont)
    ax_balmer_sfr.tick_params(labelsize=single_column_axisfont)
    ax_balmer_metallicity.tick_params(labelsize=single_column_axisfont)
    ax_cbar_sfr.tick_params(labelsize=single_column_axisfont)
    ax_cbar_metallicity.tick_params(labelsize=single_column_axisfont)
    ax_balmer_metallicity.set_xlim(8,9)
    # ax_balmer_sfr.axhline(0.85, ls='--', color='#8E248C')
    # ax_balmer_metallicity.axhline(0.85, ls='--', color='#8E248C')
    # ax_balmer_sfr.axhline(1.9, ls='--', color='#FF640A')
    # ax_balmer_metallicity.axhline(1.9, ls='--', color='#FF640A')
    plt.setp(ax_balmer_metallicity.get_yticklabels()[0], visible=False)   
    fig.savefig(imd.cluster_dir + f'/cluster_stats/balmer_sfr_metallicity.pdf',bbox_inches='tight')
    plt.close('all')

# plot_balmer_sfr_metallicity(20, ignore_groups)