from full_phot_read_data import read_merged_lineflux_cat, read_final_sample, read_possible_sample, read_paper_df, read_phot_df
from full_phot_merge_lineflux import filter_bcg_flags
import matplotlib.pyplot as plt
from uncover_read_data import read_SPS_cat_all, read_bcg_surface_brightness, read_supercat, read_morphology_cat
from compute_av import compute_ha_pab_av, compute_pab_paa_av, compute_paalpha_pabeta_av, compute_balmer_av, compute_ratio_from_av, compute_balmer_ratio_from_av
from read_mosdef_data import get_shapley_sample, get_mosdef_compare_sample
import pandas as pd
import numpy as np
import random
import sys
from plot_vals import *
import matplotlib as mpl
from matplotlib.lines import Line2D
import shutil
from simple_sample_selection import truncate_colormap
from compute_av import compute_ratio_from_av, avneb_str
import time
from read_mosdef_data import get_shapley_sample, get_mosdef_compare_sample
from read_data import linemeas_df
import random

# Set the seed for reproducibility
random.seed(5842750384) 



def plot_paper_mass_match_neb_curve(color_var='snr', shapley=2):
    shaded = 0
    
    sample_df = read_final_sample()
    sample_df['log_sfr100_50'] = np.log10(sample_df['sfr100_50'])
    sample_df['log_sfr100_16'] = np.log10(sample_df['sfr100_16'])
    sample_df['log_sfr100_84'] = np.log10(sample_df['sfr100_84'])

    var_name = 'mstar_50'            
    x_lims = [3, 4.8]
    # median_bins = [[9,9.5], [9.5,9.85], [9.85,10.2], [10.2,10.75]]
    median_bins = [[9.,9.5], [9.5,9.9], [9.9,10.3], [10.3,10.8]]
    # mosdef percentiles: 9.2-9.6, 9.6-9.9, 9.9-10.2, 10.2-
    # sample percentiles: 9.2-9.45, 9.45-9.6, 9.6-10.0, 10.0-
    if shaded == 1:
        cbar_bounds = [8.5, 9, 9.5, 9.9, 10.3, 10.8]
        cbar_ticks = [8.5, 9, 9.5, 9.9, 10.3, 10.8]

    else:
        cbar_bounds = [9, 9.5, 9.9, 10.3, 10.8]
        cbar_ticks = [9, 9.5, 9.9, 10.3, 10.8]


    cmap = mpl.cm.inferno
    cmap = truncate_colormap(cmap, minval=0.25, maxval=0.85)
    norm = mpl.colors.BoundaryNorm(cbar_bounds, cmap.N)
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar_label = stellar_mass_label
   
    prospector_av_lims = [-0.1, 4]
    prospector_neb_av_lims = [-0.1, 2.5]
    save_str2 = ''

   
    # fig, ax = plt.subplots(figsize=(7,6))
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_axes([0.09, 0.08, 0.65, 0.65])
    ax_cbar = fig.add_axes([0.09, 0.75, 0.65, 0.03])
    # Compute errs
    add_err_cols(sample_df, 'mstar_50')
    err_av_low_plot = sample_df[f'err_AV_pab_ha_low']
    err_av_high_plot = sample_df[f'err_AV_pab_ha_high']
    err_ratio_low_plot = sample_df[f'err_lineratio_pab_ha_low']
    err_ratio_high_plot = sample_df[f'err_lineratio_pab_ha_high']

    snrs = np.min([sample_df[f'Halpha_snr'], sample_df[f'PaBeta_snr']], axis=0)
    sample_df['min_snr'] = snrs
    
   

   
 
    y_label = f'MegaScience (Pa$\\beta$ / H$\\alpha$)'
    x_label = f'MOSDEF (H$\\alpha$ / H$\\beta$)'

    shape = 'o'
    mec = 'black'
    # if color_var == 'redshift':
    #     norm = mpl.colors.Normalize(vmin=1.2, vmax=2.5) 
    #     rgba = cmap(norm(sample_df['z_50'].iloc[j]))
    #     cbar_ticks = [1.2, 1.5, 1.8, 2.1, 2.4]
    #     cbar_label = 'Redshift'
    
    
    
   
    
    median_masses, median_pab_ha_ratios, err_median_masses, err_median_pab_ha_ratios, n_gals_per_bin = get_median_points(sample_df, median_bins, var_name)
    # Plot shaded region for lowmass bin
    add_str4=''
    if shaded:
        cbar_bounds = [8.5, 9, 9.5, 9.85, 10.2, 10.75]
        median_mass_low, median_pab_ha_ratio_low, err_median_mass_low, err_median_pab_ha_ratio_lows, n_gals_per_bin_low = get_median_points(sample_df, [[6, 9]], var_name)
        shaded_color = cmap(norm(median_mass_low[0]))
        ymin_shaded = median_pab_ha_ratio_low[0]-err_median_pab_ha_ratio_lows[0][0]
        ymax_shaded = median_pab_ha_ratio_low[0]+err_median_pab_ha_ratio_lows[1][0]
        ax.axhspan(ymin=ymin_shaded, ymax=ymax_shaded, facecolor=shaded_color, alpha=0.3)
        ax.axhline(median_pab_ha_ratio_low[0],  ls='-', color=shaded_color, marker='None')
        add_str4='_shaded'


    # shapley's data
    if shapley == 0:
        mosdef_data_mass = np.array([9.252764612954188, 9.73301737756714, 10.0173775671406, 10.437598736176936]) #Shapley 2022
        mosdef_data_decs = np.array([3.337349397590363, 3.4548192771084363, 3.7801204819277103, 4.512048192771086])
        mosdef_data_decs_low = np.array([3.26601519662585, 3.306461008231343, 3.5317649560995976, 4.347472583356616])
        mosdef_data_decs_high = np.array([3.4090408842992668, 3.6017019238471697, 4.022429020682048, 4.686355720952147])
        mosdef_err_low = mosdef_data_decs - mosdef_data_decs_low
        mosdef_err_high = mosdef_data_decs_high - mosdef_data_decs

        mosdef_data_balmeravs = compute_balmer_av(mosdef_data_decs, law='calzetti')
        mosdef_data_lineratios = compute_ratio_from_av(mosdef_data_balmeravs, law='calzetti')
        save_str3 = '_shapley'
    if shapley > 0:
        if shapley == 1:
            mosdef_df, linemeas_df = get_shapley_sample()
            save_str3 = '_shapley_'
        if shapley == 2:
            mosdef_df, linemeas_df = get_mosdef_compare_sample()
            save_str3 = '_mosdef_'
        result_df = pd.concat([mosdef_df, linemeas_df], axis=1)
        result_df['balmer_dec'] = result_df['HA6565_FLUX'] / result_df['HB4863_FLUX']
        median_masses_mosdef, median_balmer_ratios, err_median_masses_mosdef, err_median_balmer_ratios, n_gals_per_bin_mosdef = get_median_points(result_df, median_bins, 'LMASS', y_var_name='balmer_dec')
        mosdef_data_decs = median_balmer_ratios
        mosdef_err_low = err_median_balmer_ratios[0]
        mosdef_err_high = err_median_balmer_ratios[1]
        # mosdef_err_low = mosdef_data_decs - mosdef_data_decs_low
        # mosdef_err_high = mosdef_data_decs_high - mosdef_data_decs

        # mosdef_data_balmeravs = compute_balmer_av(mosdef_data_decs, law='calzetti')
        # mosdef_data_lineratios = compute_ratio_from_av(mosdef_data_balmeravs, law='calzetti')

    
    
    # ax.plot(mosdef_data_mass, mosdef_data_lineratios, color='black', marker='s', ms=10, mec='black', ls='dotted', zorder=1000000, label='z=2.3 MOSDEF (Shapley+ 2022)')
    
    # runco_data_mass = np.array([9.04029773256327, 9.341541353064535	, 9.507356359477967, 9.660911296972452, 9.76852663271054, 9.882224549023732, 10.039519064690833, 10.177858006949581, 10.35957669226384, 10.679835800016289]) #Runco 2022
    # runco_data_decs = np.array([3.282805987024606, 3.6142358130258136, 2.633209874253258, 4.096971898622865, 4.597955149928179, 4.213816474455239, 4.5059129818220125, 4.514513937180409, 5.778564129951793, 5.644147137838686])
    # runco_data_balmeravs = compute_balmer_av(runco_data_decs, law='calzetti')
    # runco_data_lineratios = compute_ratio_from_av(runco_data_balmeravs, law='calzetti')
    # ax.plot(runco_data_mass, runco_data_lineratios, color='#f2f2f2', marker='d', ms=8, mec='black', ls='None', zorder=1000000, label='MOSDEF Stacks (Runco+ 2022)')
    for k in range(len(median_pab_ha_ratios)):
        # norm = mpl.colors.LogNorm(vmin=9, vmax=10.5) 
        
        rgba = cmap(norm(median_masses[k]))
        

        lineratio_err_plot = np.array([[err_median_pab_ha_ratios[0][k], err_median_pab_ha_ratios[1][k]]]).T
        mosdef_err_plot = np.array([[mosdef_err_low[k], mosdef_err_high[k]]]).T

        ax.errorbar(mosdef_data_decs[k], median_pab_ha_ratios[k], xerr=mosdef_err_plot, yerr=lineratio_err_plot, marker=shape, mec=mec, ms=6, color=rgba, ls='None', ecolor='gray')
    
  
    # Add attenuation curves to prospector plots
    av_values = np.arange(0, 6, 0.01)
    def add_attenuation_curveby_av(av_values, curve_name, color, style):
        pab_ha_ratio = compute_ratio_from_av(av_values, law=curve_name)
        balmer_dec = compute_balmer_ratio_from_av(av_values, law=curve_name)
        ax.plot(balmer_dec, pab_ha_ratio, color=color, ls=style, marker='None')
        legend_line = Line2D([0], [0], color=color, marker='None', ls=style)
        return legend_line
    legend_line_reddy = add_attenuation_curveby_av(av_values, 'reddy', 'black', '--')
    # legend_line_calzetti = add_attenuation_curveby_av(av_values, 'calzetti', 'red')
    legend_line_cardelli = add_attenuation_curveby_av(av_values, 'cardelli', 'black', '-.')
    ax.text(4.46, 0.19, 'Reddy+25', fontsize=10, rotation=42)
    ax.text(4.4, 0.105, 'Cardelli+89', fontsize=10, rotation=16)

    
        
    # custom_lines = [legend_line_cardelli, legend_line_reddy]
    # custom_labels = ['Cardelli+89', 'Reddy+25']
    # ax.legend(custom_lines, custom_labels, loc=4)

        
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.tick_params(labelsize=14)
    
    sm =  mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    add_cbar(fig, ax_cbar, norm, cmap, cbar_label, cbar_ticks, mappable)
    save_str=''
    

    # Duplicating y axis for AV 
    labelsize=14
    fontsize=14
    
    # ax.plot([-10, 100], [-10, 100], ls='--', color='red', marker='None') 
    ax.tick_params(labelsize=labelsize)
    
    # x_tick_locs = [0.03, 0.055, 1/10, 1/5]
    # x_tick_labs = ['0.03', '0.055', '0.1', '0.2']
    # ax.set_xticks(x_tick_locs)
    # ax.set_xticklabels(x_tick_labs)
    
    main_ax_lims = np.array([0.055, 0.3])
        
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

    
  
    fig.savefig(f'/Users/brianlorenz/uncover/Figures/PHOT_paper/neb_curve/neb_curve{save_str}{save_str2}{save_str3}{color_var}{add_str4}.pdf', bbox_inches='tight')
    plt.close('all')


def get_median_points(sample_df, median_bins, x_var_name, y_var_name='lineratio_pab_ha', n_boots=1000):
    median_idxs = [np.logical_and(sample_df[x_var_name] > median_bins[k][0], sample_df[x_var_name] < median_bins[k][1]) for k in range(len(median_bins))]
    median_xvals = [np.median(sample_df[median_idxs[k]][x_var_name]) for k in range(len(median_bins))]
    median_yvals = [np.median(sample_df[median_idxs[k]][y_var_name]) for k in range(len(median_bins))]
    
    # median_xerr_low = np.array([median_xvals[k]-np.percentile(sample_df[median_idxs[k]][x_var_name], 16) for k in range(len(median_bins))])
    # median_xerr_high = np.array([np.percentile(sample_df[median_idxs[k]][x_var_name], 84)-median_xvals[k] for k in range(len(median_bins))])
    # median_xerr_plot = np.vstack([median_xerr_low, median_xerr_high])

    # median_yerr_low = np.array([median_yvals[k]-np.percentile(sample_df[median_idxs[k]][y_var_name], 16) for k in range(len(median_bins))])
    # median_yerr_high = np.array([np.percentile(sample_df[median_idxs[k]][y_var_name], 84)-median_yvals[k] for k in range(len(median_bins))])
    # median_yerr_plot = np.vstack([median_yerr_low, median_yerr_high])

    # bootstrap the errors
    median_xerr_low = np.zeros(len(median_bins))
    median_xerr_high = np.zeros(len(median_bins))
    median_yerr_low = np.zeros(len(median_bins))
    median_yerr_high = np.zeros(len(median_bins))
    for k in range(len(median_bins)):
        df = sample_df[median_idxs[k]]
        row_idxs = np.arange(0, len(df), 1)
        mass_boots = []
        line_boots = []
        for boot in range(n_boots):
            selected_idxs = np.random.choice(row_idxs, len(df))
            mass_boots.append(np.median(df.iloc[selected_idxs][x_var_name]))
            line_boots.append(np.median(df.iloc[selected_idxs][y_var_name]))
        median_xerr_low[k] = median_xvals[k] - np.percentile(mass_boots, 16)
        median_xerr_high[k] = np.percentile(mass_boots, 84) - median_xvals[k]
        median_yerr_low[k] = median_yvals[k] - np.percentile(line_boots, 16)
        median_yerr_high[k] = np.percentile(line_boots, 84) - median_yvals[k]
    median_xerr_plot = np.vstack([median_xerr_low, median_xerr_high])
    median_yerr_plot = np.vstack([median_yerr_low, median_yerr_high])
    n_gals_per_bin = [len(sample_df[median_idxs[k]]) for k in range(len(median_bins))]

    return median_xvals, median_yvals, median_xerr_plot, median_yerr_plot, n_gals_per_bin

def add_cbar(fig, ax_cbar, norm, cmap, cbar_name, cbar_ticks, sm):
    #SNR cbar
    cbar_ticklabels = [str(tick) for tick in cbar_ticks]
    cbar = fig.colorbar(sm, cax=ax_cbar, orientation='horizontal', ticks=cbar_ticks, ticklocation='top', spacing='proportional')
    cbar.ax.set_xticklabels(cbar_ticklabels) 
    cbar.ax.xaxis.minorticks_off()
    cbar.set_label(cbar_name, fontsize=14, labelpad=10) # -55 pad if ticks flip back to bottom
    cbar.ax.tick_params(labelsize=14)

def prepare_axis_ratios(sample_df):
    bands = ['f444w', 'f150w', 'f162m', 'f182m', 'f210m', 'f250m', 'f300m', 'f335m', 'f360m', 'f410m', 'f430m', 'f460m']
    for band in bands:
        sample_df[f'{band}_axisratio_84'] = 1-sample_df[f'{band}_ellip_16']
        sample_df[f'{band}_axisratio_50'] = 1-sample_df[f'{band}_ellip_50']
        sample_df[f'{band}_axisratio_16'] = 1-sample_df[f'{band}_ellip_84']
        add_err_cols(sample_df, f'{band}_axisratio_50')

def prepare_prospector_dust(sample_df):
    dust2_50 = sample_df['dust2_50']
    dust1_frac_50 = sample_df['dust1_fraction_50']
    sample_df['prospector_total_av_50'] = 1.086 * (dust2_50 + (dust2_50*dust1_frac_50))
    sample_df['prospector_neb_av_50'] = 1.086 * dust2_50*dust1_frac_50
    dust_boots = []
    av_neb_boots = []
    for ijkl in range(1000):
        x = random.uniform(0, 1)
        if x < 0.5:
            scale = dust2_50 - sample_df['dust2_16']
            dust2_boot = dust2_50 - np.abs(np.random.normal(loc = 0, scale=scale))
        if x > 0.5:
            scale = sample_df['dust2_84'] - dust2_50 
            dust2_boot = dust2_50 + np.abs(np.random.normal(loc = 0, scale=scale))
        x = random.uniform(0, 1)
        if x < 0.5:
            scale = dust1_frac_50 - sample_df['dust1_fraction_16']
            dust1_frac_boot = dust1_frac_50 - np.abs(np.random.normal(loc = 0, scale=scale))
        if x > 0.5:
            scale = sample_df['dust1_fraction_84'] - dust1_frac_50 
            dust1_frac_boot = dust1_frac_50 + np.abs(np.random.normal(loc = 0, scale=scale))
        dust_boot = 1.086 * (dust2_boot + (dust2_boot*dust1_frac_boot))
        dust_boots.append(dust_boot)
        av_neb_boots.append(1.086*dust2_boot*dust1_frac_boot)
    sample_df['err_prospector_total_av_50_low'] =  sample_df['prospector_total_av_50'] - np.percentile(dust_boots, 1, axis=0)
    sample_df['err_prospector_total_av_50_high'] =  np.percentile(dust_boots, 84, axis=0) - sample_df['prospector_total_av_50']
    sample_df['err_prospector_neb_av_50_low'] =  sample_df['prospector_neb_av_50'] - np.percentile(av_neb_boots, 16, axis=0)
    sample_df['err_prospector_neb_av_50_high'] =  np.percentile(av_neb_boots, 84, axis=0) - sample_df['prospector_neb_av_50']
    
    # hoiw to best handle negative errorbars?
    def set_neg_errs_to_99(sample_df, colname):
        sample_df.loc[sample_df[colname] < 0, colname] = 99
    set_neg_errs_to_99(sample_df, 'err_prospector_total_av_50_low')
    set_neg_errs_to_99(sample_df, 'err_prospector_total_av_50_high')
    set_neg_errs_to_99(sample_df, 'err_prospector_neb_av_50_low')
    set_neg_errs_to_99(sample_df, 'err_prospector_neb_av_50_high')

def add_err_cols(sample_df, var_name):
    sample_df[f'err_{var_name}_low'] = sample_df[f'{var_name}'] - sample_df[var_name[:-2]+'16']
    sample_df[f'err_{var_name}_high'] = sample_df[var_name[:-2]+'84'] - sample_df[f'{var_name}']
    return sample_df

if __name__ == '__main__':
    
    plot_paper_mass_match_neb_curve(color_var='mass', shapley=2) # Shapley = 2 for the paper fig

    