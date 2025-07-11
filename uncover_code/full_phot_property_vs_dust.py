from full_phot_read_data import read_merged_lineflux_cat, read_final_sample, read_possible_sample, read_paper_df
from full_phot_merge_lineflux import filter_bcg_flags
import matplotlib.pyplot as plt
from uncover_read_data import read_SPS_cat_all, read_bcg_surface_brightness, read_supercat, read_morphology_cat
from compute_av import compute_ha_pab_av, compute_pab_paa_av, compute_paalpha_pabeta_av
import pandas as pd
import numpy as np
import random
from plot_vals import *
import matplotlib as mpl
from matplotlib.lines import Line2D
import shutil
from simple_sample_selection import truncate_colormap
from compute_av import compute_ratio_from_av, avneb_str

def plot_paper_dust_vs_prop(color_var='snr'):
    sample_df = read_final_sample()
    possible_df = read_possible_sample()

    cmap = mpl.cm.inferno

   
    sfr_lims = [-2, 2.5]
    mass_lims = [7, 11.5]
    save_str2 = ''

   
    # fig, ax = plt.subplots(figsize=(7,6))
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_axes([0.09, 0.08, 0.65, 0.65])
    ax_cbar = fig.add_axes([0.09, 0.78, 0.65, 0.03])

    sps_all_df = read_SPS_cat_all()
    # Gray background
    # all_masses = sps_all_df['mstar_50']
    # all_sfr100s = sps_all_df['sfr100_50']
    # all_log_sfr100s = np.log10(all_sfr100s)
    # # ax.plot(all_redshifts, all_masses, marker='o', ls='None', markersize=background_markersize, color='gray')
    # cmap = plt.get_cmap('gray_r')
    # new_cmap = truncate_colormap(cmap, 0, 0.7)
    # good_mass_idx = np.logical_and(all_masses > mass_lims[0], all_masses < mass_lims[1])
    # good_sfr_idx = np.logical_and(all_log_sfr100s > sfr_lims[0], all_log_sfr100s <  sfr_lims[1])

    # if show_hexes:
    #     hexbin_norm = mpl.colors.Normalize(vmin=1, vmax=200) 
    #     good_both_idx = np.logical_and(good_sfr_idx, good_mass_idx)  
    #     ax.hexbin(all_masses[good_both_idx], all_log_sfr100s[good_both_idx], gridsize=15, cmap=new_cmap, norm=hexbin_norm, label='Photometric Sample')
    #     save_str2 = '_hexes'


    # Compute errs
    low_sfr_err = np.log10(sample_df['sfr100_50']) - np.log10(sample_df['sfr100_16'])
    high_sfr_err = np.log10(sample_df['sfr100_84']) - np.log10(sample_df['sfr100_50'])
    low_mass_err = sample_df['mstar_50'] - sample_df['mstar_16']
    high_mass_err = sample_df['mstar_84'] - sample_df['mstar_50']
    err_av_low_plot = sample_df[f'err_AV_pab_ha_low']
    err_av_high_plot = sample_df[f'err_AV_pab_ha_high']
    err_ratio_low_plot = sample_df[f'err_lineratio_pab_ha_low']
    err_ratio_high_plot = sample_df[f'err_lineratio_pab_ha_high']

    snrs = np.min([sample_df[f'Halpha_snr'], sample_df[f'PaBeta_snr']], axis=0)
    sample_df['min_snr'] = snrs
    
    for j in range(len(sample_df)):
        id_dr3 = sample_df['id_dr3'].iloc[j]

        sfr_err_plot = np.array([[low_sfr_err.iloc[j], high_sfr_err.iloc[j]]]).T
        mass_err_plot = np.array([[low_mass_err.iloc[j], high_mass_err.iloc[j]]]).T
        av_err_plot = np.array([[err_av_low_plot.iloc[j], err_av_high_plot.iloc[j]]]).T
        lineratio_err_plot = np.array([[err_ratio_low_plot.iloc[j], err_ratio_high_plot.iloc[j]]]).T

        shape = 'o'
        mec = 'black'
        if color_var == 'snr':
            norm = mpl.colors.LogNorm(vmin=0.5, vmax=50) 
            rgba = cmap(norm(sample_df['min_snr'].iloc[j]))
        if color_var == 'redshift':
            norm = mpl.colors.Normalize(vmin=1.2, vmax=2.5) 
            rgba = cmap(norm(sample_df['z_50'].iloc[j]))
        

        
        #  np.log10(sample_df['sfr100_50'].iloc[j])
        ax.errorbar(sample_df['mstar_50'].iloc[j], sample_df[f'lineratio_pab_ha'].iloc[j], xerr=mass_err_plot, yerr=lineratio_err_plot, marker=shape, mec=mec, ms=6, color=rgba, ls='None', ecolor='gray')
    
    ax.set_xlabel(stellar_mass_label, fontsize=14)
    ax.set_ylabel(f'(Pa$\\beta$ / H$\\alpha$)', fontsize=14)
    ax.tick_params(labelsize=14)
    
    
    add_cbar(fig, ax_cbar, norm, cmap, color_var)
    save_str='test'
    
    # line_sample = Line2D([0], [0], color='orange', marker='o', markersize=8, ls='None', mec='black')
    # line_bcg = Line2D([0], [0], color='blue', marker='o', markersize=4, ls='None', mec='black')
    # line_snr = Line2D([0], [0], color='red', marker='o', markersize=4, ls='None', mec='black')
    # line_chi2 = Line2D([0], [0], color='red', marker='x', markersize=4, ls='None', mec='red')
    # custom_lines = [line_sample, line_bcg, line_snr, line_chi2]
    # custom_labels = ['Sample', 'Close to bcg', 'Low snr', 'Bad cont slope']
    # ax.legend(custom_lines, custom_labels, bbox_to_anchor=(1.05, 1.14))
    # save_str = '_color'

    # Duplicating y axis for AV 
    labelsize=14
    fontsize=14
    ax2 = ax.twinx()
    ax.tick_params(labelsize=labelsize)
    ax2.tick_params(labelsize=labelsize)
    # ax.plot([-10, 100], [-10, 100], ls='--', color='red', marker='None')
    ax.set_xlim(mass_lims)
    main_ax_lims = np.array([0.02, 1])
    ax.set_ylim(main_ax_lims)
    ax2.set_ylim(1/main_ax_lims)
    ax.set_yscale('log')
    ax2.set_yscale('log')
    # x_tick_locs = [0.03, 0.055, 1/10, 1/5]
    # x_tick_labs = ['0.03', '0.055', '0.1', '0.2']
    y_tick_locs = [0.025, 0.055, 1/10, 1/5, 1/2, 1]
    y_tick_labs = ['0.025', '0.055', '0.1', '0.2', '0.5', '1']
    ax.set_yticks(y_tick_locs)
    ax.set_yticklabels(y_tick_labs)
    # ax.set_xticks(x_tick_locs)
    # ax.set_xticklabels(x_tick_labs)
    twin_y_tick_labs = ['-2', '-1', '0', '1', '2', '3', '4', '5', '6']
    twin_y_tick_locs = [1/compute_ratio_from_av(float(rat)) for rat in twin_y_tick_labs]
    # breakpoint()
    ax2.set_yticks(twin_y_tick_locs)
    ax2.set_yticklabels(twin_y_tick_labs)
    ax2.set_ylabel(f'Inferred {avneb_str}', fontsize=fontsize, rotation=270, labelpad=20)
    ax2.minorticks_off()
    ax.minorticks_off()
        
    
    fig.savefig(f'/Users/brianlorenz/uncover/Figures/PHOT_paper/dust_vs_prop_{save_str}{save_str2}{color_var}.pdf', bbox_inches='tight')
    plt.close('all')


def add_cbar(fig, ax_cbar, norm, cmap, cbar_name):
    #SNR cbar
    sm =  mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar_ticks = [1.2, 1.5, 1.8, 2.1, 2.4]
    cbar_ticklabels = [str(tick) for tick in cbar_ticks]
    cbar = fig.colorbar(sm, cax=ax_cbar, orientation='horizontal', ticks=cbar_ticks)
    cbar.ax.set_xticklabels(cbar_ticklabels) 
    cbar.ax.xaxis.minorticks_off()
    cbar.set_label(cbar_name, fontsize=14, labelpad=-55)
    cbar.ax.tick_params(labelsize=14)


if __name__ == '__main__':

    plot_paper_dust_vs_prop()
    plot_paper_dust_vs_prop(color_var='redshift')