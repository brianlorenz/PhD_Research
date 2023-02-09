import initialize_mosdef_dirs as imd
import matplotlib.pyplot as plt
from astropy.io import ascii
import numpy as np
from matplotlib.patches import Ellipse
from ellipses_for_plotting import get_ellipse_shapes
import matplotlib as mpl
from plot_vals import *
from dust_model import *
from sympy.solvers import solve
from sympy import Symbol


def plot_av_extra():
    summary_df = ascii.read(imd.loc_cluster_summary_df).to_pandas()
    
    summary_df['AV_Extra'] = summary_df['balmer_av'] - summary_df['AV']
    summary_df['err_AV'] = 0.1
    summary_df['err_balmer_AV'] = (summary_df['err_balmer_av_low'] + summary_df['err_balmer_av_high'])/2
    summary_df['err_AV_Extra'] = np.sqrt(0.1**2+summary_df['err_balmer_AV']**2)
    summary_df['err_AV_Extra_low'] = np.sqrt(0.1**2+summary_df['err_balmer_av_low']**2)
    summary_df['err_AV_Extra_high'] = np.sqrt(0.1**2+summary_df['err_balmer_av_high']**2)


    #AV Extra vs mass
    fig, ax = plt.subplots(figsize = (8,8))

    ax.set_xlim(9.25, 10.75)
    ax.set_xlabel(stellar_mass_label, fontsize=single_column_axisfont)
    ax_x_len=1.5

    ax_y_top = 2.5
    ax.set_ylim(0, ax_y_top)
    ax.set_ylabel(av_extra_label, fontsize=single_column_axisfont)
    ax_y_len = ax_y_top

    for i in range(len(summary_df)):
        row = summary_df.iloc[i]
        x_points = row['log_mass']

        cmap = mpl.cm.inferno
        norm = mpl.colors.Normalize(vmin=0, vmax=2.0) 
        rgba = cmap(norm(row['log_sfr']))

        ax.errorbar(x_points, row['AV_Extra'], yerr=np.array([[row['err_AV_Extra_low'], row['err_AV_Extra_high']]]).T, color=rgba, marker='o', ls='None', zorder=3)
    
    ax.tick_params(labelsize=single_column_axisfont)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
    # cbar.set_label(balmer_label, fontsize=fontsize)
    cbar.set_label(sfr_label, fontsize=single_column_axisfont)
    cbar.ax.tick_params(labelsize=single_column_axisfont)
    fig.savefig(imd.cluster_dir + f'/cluster_stats/av_extra_mass.pdf',bbox_inches='tight')

    plt.close('all')

    #AV Extra vs ssfr
    
    fig, ax = plt.subplots(figsize = (8,8))

    ax.set_xlim(-9.2, -8.0)
    ax.set_xlabel(ssfr_label, fontsize=single_column_axisfont)
    ax_x_len=1.2

    ax_y_top = 2.5
    ax.set_ylim(0, ax_y_top)
    ax.set_ylabel(av_extra_label, fontsize=single_column_axisfont)
    ax_y_len = ax_y_top

    for i in range(len(summary_df)):
        row = summary_df.iloc[i]
        x_points = row['log_ssfr']

        cmap = mpl.cm.inferno
        norm = mpl.colors.Normalize(vmin=9, vmax=11) 
        rgba = cmap(norm(row['log_mass']))

        ax.errorbar(x_points, row['AV_Extra'], yerr=np.array([[row['err_AV_Extra_low'], row['err_AV_Extra_high']]]).T, color=rgba, marker='o', ls='None', zorder=3)
    
    ax.tick_params(labelsize=single_column_axisfont)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
    # cbar.set_label(balmer_label, fontsize=fontsize)
    cbar.set_label(sfr_label, fontsize=single_column_axisfont)
    cbar.ax.tick_params(labelsize=single_column_axisfont)
    fig.savefig(imd.cluster_dir + f'/cluster_stats/av_extra_ssfr.pdf',bbox_inches='tight')


# plot_av_extra()