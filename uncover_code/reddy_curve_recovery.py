from full_phot_read_data import read_merged_lineflux_cat, read_final_sample, read_possible_sample, read_paper_df
from full_phot_merge_lineflux import filter_bcg_flags
import matplotlib.pyplot as plt
from uncover_read_data import read_SPS_cat_all, read_bcg_surface_brightness, read_supercat, read_morphology_cat
from compute_av import compute_ratio_from_av, compute_balmer_ratio_from_av
import pandas as pd
import numpy as np
import sys
import random
from plot_vals import *
import matplotlib as mpl
from matplotlib.lines import Line2D
import shutil
from simple_sample_selection import truncate_colormap
from read_mosdef_data import get_shapley_sample, get_mosdef_compare_sample
from full_phot_plot_mass_nebcurve import add_cbar_nebcurve


random.seed(85801479823) 

R_V_value = 7.1


def av_mass_relation(log_M, scatter = 0.35):
    A_V = 1.6 * log_M - 14.10 
    # A_V_scattered = np.random.normal(loc=A_V, scale=scatter)
    A_V_scattered = A_V
    return A_V_scattered


def recovery_test(n_gals=66):
    sample_df = read_final_sample()
    mass_distribution = np.array(sample_df['mstar_50'])
    av_distribution = av_mass_relation(mass_distribution)

    predicted_ha_hb = compute_balmer_ratio_from_av(av_distribution, law='reddy', R_V_value=R_V_value)
    predicted_pab_ha = compute_ratio_from_av(av_distribution, law='reddy', R_V_value=R_V_value)

    median_bins = [[9.,9.5], [9.5,9.9], [9.9,10.3], [10.3,10.8]]
    median_masses, median_ha_hbs = get_median_points(median_bins, mass_distribution, predicted_ha_hb)
    _, median_pab_has = get_median_points(median_bins, mass_distribution, predicted_pab_ha)

    # Plotting
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_axes([0.09, 0.08, 0.65, 0.65])
    ax_cbar = fig.add_axes([0.76, 0.08, 0.03, 0.65])

    cbar_bounds = [9, 9.5, 9.9, 10.3, 10.8]
    cbar_ticks = [9, 9.5, 9.9, 10.3, 10.8]
    cbar_ticklabels = [str(tick) for tick in cbar_ticks]

    cmap = mpl.cm.inferno
    cmap = truncate_colormap(cmap, minval=0.25, maxval=0.85)
    norm = mpl.colors.BoundaryNorm(cbar_bounds, cmap.N)
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar_label = stellar_mass_label

    for k in range(len(median_masses)):
        # norm = mpl.colors.LogNorm(vmin=9, vmax=10.5) 
        
        rgba = cmap(norm(median_masses[k]))
        

        # lineratio_err_plot = np.array([[err_median_pab_ha_ratios[0][k], err_median_pab_ha_ratios[1][k]]]).T
        # mosdef_err_plot = np.array([[mosdef_err_low[k], mosdef_err_high[k]]]).T

        ax.plot(median_ha_hbs[k], median_pab_has[k], marker='o', mec='black', ms=8, color=rgba, ls='None', zorder=10) #, ecolor='gray'
    ax.plot(predicted_ha_hb, predicted_pab_ha, marker='o', mec='None', ms=6, color='gray', ls='None', zorder=9) 


    
    y_label = f'Pa$\\beta$ / H$\\alpha$'
    x_label = f'H$\\alpha$ / H$\\beta$'


    av_values = np.arange(0, 6, 0.01)
    def add_attenuation_curveby_av(av_values, curve_name, color, style):
        pab_ha_ratio = compute_ratio_from_av(av_values, law=curve_name)
        balmer_dec = compute_balmer_ratio_from_av(av_values, law=curve_name)
        ax.plot(balmer_dec, pab_ha_ratio, color=color, ls=style, marker='None')
        legend_line = Line2D([0], [0], color=color, marker='None', ls=style)
        return legend_line
    legend_line_reddy = add_attenuation_curveby_av(av_values, 'reddy', 'black', '--')
    legend_line_cardelli = add_attenuation_curveby_av(av_values, 'cardelli', 'black', '-.')
    ax.text(4.46, 0.19, 'Reddy+25', fontsize=10, rotation=42)
    ax.text(4.4, 0.105, 'Cardelli+89', fontsize=10, rotation=16)

    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.tick_params(labelsize=14)
    
    


    

    sm =  mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    add_cbar_nebcurve(fig, ax_cbar, norm, cmap, cbar_label, cbar_ticks, cbar_ticklabels, mappable)
    save_str=''

    # Duplicating y axis for AV 
    labelsize=14
    fontsize=14
    ax.tick_params(labelsize=labelsize)

    main_ax_lims = np.array([0.055, 0.3])    
    x_lims = [3, 4.8]
    ax.set_ylim(main_ax_lims)
    # ax.set_yscale('log')
    y_tick_locs = [1/10, 1/5, 0.3]
    y_tick_labs = ['0.1', '0.2', '0.3']
    ax.set_yticks(y_tick_locs)
    ax.set_yticklabels(y_tick_labs)
    
    ax.set_xlim(x_lims)    
    x_tick_labs = ['3', '3.5', '4', '4.5']
    x_tick_locs = [float(rat) for rat in x_tick_labs]
    ax.set_xticks(x_tick_locs)
    ax.set_xticklabels(x_tick_labs)
    ax.minorticks_off()

    
    fig.savefig(f'/Users/brianlorenz/uncover/Figures/PHOT_paper/neb_curve_recovery/test.pdf', bbox_inches='tight')
    

def get_median_points(median_bins, x_col, y_col): # bin by xcol
    median_idxs = [np.logical_and(x_col > median_bins[k][0], x_col < median_bins[k][1]) for k in range(len(median_bins))]
    median_xvals = [np.median(x_col[median_idxs[k]]) for k in range(len(median_bins))]
    median_yvals = [np.median(y_col[median_idxs[k]]) for k in range(len(median_bins))]

    return median_xvals, median_yvals

recovery_test()