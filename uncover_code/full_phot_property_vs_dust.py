from full_phot_read_data import read_merged_lineflux_cat, read_final_sample, read_possible_sample, read_paper_df, read_phot_df
from full_phot_merge_lineflux import filter_bcg_flags
import matplotlib.pyplot as plt
from uncover_read_data import read_SPS_cat_all, read_bcg_surface_brightness, read_supercat, read_morphology_cat
from compute_av import compute_ha_pab_av, compute_pab_paa_av, compute_paalpha_pabeta_av, compute_balmer_av, compute_ratio_from_av
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
from full_phot_plot_mass_nebcurve import get_median_points
import time

random.seed(80148273) 


def plot_paper_dust_vs_prop(prop='mass', color_var='snr', phot_df=[], axisratio_vs_prospector = 0, ax_in='None', bin_type='dex'):
    sample_df = read_final_sample()
    sample_df['log_sfr100_50'] = np.log10(sample_df['sfr100_50'])
    sample_df['log_sfr100_16'] = np.log10(sample_df['sfr100_16'])
    sample_df['log_sfr100_84'] = np.log10(sample_df['sfr100_84'])
    possible_df = read_possible_sample()
    if len(phot_df)<1:
        phot_df = read_phot_df()

    cmap = mpl.cm.inferno
    cmap = truncate_colormap(cmap, 0.2, 1)


   
    sfr_lims = [-1, 2.5]
    mass_lims = [7, 11.5]
    axisratio_lims = [0, 1]
    prospector_av_lims = [-0.1, 4]
    prospector_neb_av_lims = [-0.1, 2.5]
    save_str2 = ''

   
    # fig, ax = plt.subplots(figsize=(7,6))
    if ax_in == 'None':
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_axes([0.09, 0.08, 0.65, 0.65])
        if color_var != 'None':
            ax_cbar = fig.add_axes([0.09, 0.75, 0.65, 0.03])
    else: 
        ax = ax_in

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

    # Calculations for prsoecptor av and errs
    prepare_prospector_dust(sample_df)

    # turn ellipticity into axis ratio and comupte errs
    prepare_axis_ratios(sample_df)


    # Compute errs
    add_err_cols(sample_df, 'mstar_50')
    add_err_cols(sample_df, 'log_sfr100_50')
    add_err_cols(sample_df, 'f444w_ellip_50')
    add_err_cols(sample_df, 'f150w_ellip_50')
    sample_df['halpha_axisratio_50'] = np.zeros(len(sample_df))
    sample_df['halpha_axisratio_16'] = np.zeros(len(sample_df))
    sample_df['halpha_axisratio_84'] = np.zeros(len(sample_df))
    sample_df['err_halpha_axisratio_50_low'] = np.zeros(len(sample_df))
    sample_df['err_halpha_axisratio_50_high'] = np.zeros(len(sample_df))
    err_av_low_plot = sample_df[f'err_AV_pab_ha_low']
    err_av_high_plot = sample_df[f'err_AV_pab_ha_high']
    err_ratio_low_plot = sample_df[f'err_lineratio_pab_ha_low']
    err_ratio_high_plot = sample_df[f'err_lineratio_pab_ha_high']

    snrs = np.min([sample_df[f'Halpha_snr'], sample_df[f'PaBeta_snr']], axis=0)
    sample_df['min_snr'] = snrs
    
    for j in range(len(sample_df)):
        id_dr3 = sample_df['id_dr3'].iloc[j]

        av_err_plot = np.array([[err_av_low_plot.iloc[j], err_av_high_plot.iloc[j]]]).T
        lineratio_err_plot = np.array([[err_ratio_low_plot.iloc[j], err_ratio_high_plot.iloc[j]]]).T

        # Typical y vars 
        y_plot = sample_df[f'lineratio_pab_ha'].iloc[j]
        y_err = lineratio_err_plot
        y_label = f'(Pa$\\beta$ / H$\\alpha$)'

        shape = 'o'
        mec = 'black'
        if color_var == 'snr':
            norm = mpl.colors.LogNorm(vmin=3, vmax=50) 
            rgba = cmap(norm(sample_df['min_snr'].iloc[j]))
            cbar_ticks = [3, 5, 10, 20, 50]
            cbar_label = 'min(SNR)'
        if color_var == 'redshift':
            norm = mpl.colors.Normalize(vmin=1.2, vmax=2.5) 
            rgba = cmap(norm(sample_df['z_50'].iloc[j]))
            cbar_ticks = [1.2, 1.5, 1.8, 2.1, 2.4]
            cbar_label = 'Redshift'
        if color_var == 'None':
            rgba = 'black'
        
        if prop == 'mass':
            var_name = 'mstar_50'            
            x_label = stellar_mass_label
            x_lims = mass_lims
            
            if bin_type == 'dex':
                median_bins = [[7,8], [8,9], [9,10], [10,11]]

            if bin_type == 'galaxies':
                mass_20, mass_40, mass_60, mass_80 = np.percentile(sample_df['mstar_50'], [20, 40, 60, 80])
                median_bins = [[7,mass_20], [mass_20,mass_40], [mass_40,mass_60], [mass_60,mass_80], [mass_80,11]]

        
        elif prop == 'sfr':
            var_name = 'log_sfr100_50'
            x_label = log_sfr_label_sedmethod
            x_lims = sfr_lims

            if bin_type == 'dex':
                median_bins = [[-0.5,0], [0,0.5], [0.5,1], [1,1.5], [1.5,2]]
            if bin_type == 'galaxies':
                sfr_20, sfr_40, sfr_60, sfr_80 = np.percentile(sample_df['log_sfr100_50'], [20, 40, 60, 80])
                median_bins = [[sfr_lims[0],sfr_20], [sfr_20,sfr_40], [sfr_40,sfr_60], [sfr_60,sfr_80], [sfr_80,sfr_lims[1]]]
        
        if 'prospector' in prop:
            var_name = prop   
            if prop.split('_')[1] == 'total':
                prosp_label = 'Prospector A$_{\\mathrm{V,tot}}$'
                x_lims = prospector_av_lims
            if prop.split('_')[1] == 'neb':
                prosp_label = 'Prospector A$_{\\mathrm{V,neb}}$'
                x_lims = prospector_neb_av_lims
            x_label = prosp_label
            median_bins = []
            if axisratio_vs_prospector!=0: 
                y_var_name = var_name
                y_lims = x_lims
                y_label = x_label
                y_plot = sample_df[var_name].iloc[j]
                y_err = np.array([[sample_df[f'err_{var_name}_low'].iloc[j], sample_df[f'err_{var_name}_high'].iloc[j]]]).T
        if 'axisratio' in prop or axisratio_vs_prospector!=0:
            if axisratio_vs_prospector!=0:
                if axisratio_vs_prospector==1: band_in='f150w'
                if axisratio_vs_prospector==2: band_in='f444w'
                if axisratio_vs_prospector==3: band_in='halpha'
            else:
                band_in = prop.split('_')[1]
            if band_in=='halpha':
                phot_row = phot_df[phot_df['id'] == id_dr3]
                halpha_filt = phot_row['Halpha_filter_obs'].iloc[0]
                band = halpha_filt.split('_')[1]
                #determine which band halpha is in for this object
            else:
                band = band_in
            var_name = 'halpha_axisratio_50'
            sample_df.loc[j, var_name] = sample_df[f'{band}_axisratio_50'].iloc[j]
            sample_df.loc[j, 'halpha_axisratio_16'] = sample_df[f'{band}_axisratio_16'].iloc[j]
            sample_df.loc[j, 'halpha_axisratio_84'] = sample_df[f'{band}_axisratio_84'].iloc[j]
            sample_df.loc[j, 'err_halpha_axisratio_50_low'] = sample_df[f'err_{band}_axisratio_50_low'].iloc[j]
            sample_df.loc[j, 'err_halpha_axisratio_50_high'] = sample_df[f'err_{band}_axisratio_50_high'].iloc[j]
            x_label = band_in.upper() + ' b/a'
            if band_in == 'f150w':
                x_label = 'Axis Ratio'
            x_lims = axisratio_lims
            if bin_type=='dex':
                median_bins = [[0,0.4], [0.4,0.6], [0.6,1]]
                median_bins = [[0,0.333], [0.333,0.666], [0.666,1]]
                
            if bin_type =='galaxies':
                ar_no_null = sample_df[~pd.isnull(sample_df[f'{band}_axisratio_50'])][f'{band}_axisratio_50']
                ar_20, ar_40, ar_60, ar_80 = np.percentile(ar_no_null, [20, 40, 60, 80])
                median_bins = ([0,ar_20], [ar_20,ar_40], [ar_40,ar_60], [ar_60,ar_80], [ar_80, 1])
        
        x_plot = sample_df[var_name].iloc[j]
        x_err = np.array([[sample_df[f'err_{var_name}_low'].iloc[j], sample_df[f'err_{var_name}_high'].iloc[j]]]).T

        
        #  np.log10(sample_df['sfr100_50'].iloc[j])
        ax.errorbar(x_plot, y_plot, xerr=x_err, yerr=y_err, marker=shape, mec=mec, ms=5, color=rgba, ls='None', ecolor='gray')
    
    # Add medians if there are bins
    if len(median_bins) > 0:
        if axisratio_vs_prospector != 0:
            median_xvals, median_yvals, median_xerr, median_yerr, ngals = get_median_points(sample_df, median_bins, var_name, y_var_name=y_var_name)
        else:
            median_xvals, median_yvals, median_xerr, median_yerr, ngals = get_median_points(sample_df, median_bins, var_name)
        med_color = 'cornflowerblue'
        med_color = '#ff7f00'
        med_legend = ax.errorbar(median_xvals, median_yvals, xerr=median_xerr, yerr=median_yerr, marker='s', ms=9, color=med_color, ls='--', zorder=50, mec='black', ecolor=med_color)
        for median_tuple in median_bins:
            ax.axvline(x=median_tuple[0], ymin=0, ymax=0.09, color='gray', linestyle='--')
            ax.axvline(x=median_tuple[1], ymin=0, ymax=0.09, color='gray', linestyle='--')

    # Add attenuation curves to prospector plots
    if 'prospector' in prop and axisratio_vs_prospector==0:
        lineratio_points = np.arange(0.025, 1, 0.001)
        def add_attenuation_curve(lineratio_points, curve_name, color):
            av_points = compute_ha_pab_av(lineratio_points, law=curve_name)
            ax.plot(av_points, lineratio_points, color=color, ls='--', marker='None')
            legend_line = Line2D([0], [0], color=color, marker='None', ls='--')
            return legend_line
        legend_line_calzetti = add_attenuation_curve(lineratio_points, 'calzetti', 'red')
        legend_line_reddy = add_attenuation_curve(lineratio_points, 'reddy', 'blue')
        
        custom_lines = [legend_line_calzetti, legend_line_reddy]
        custom_labels = ['Calzetti+2000', 'Reddy+2025']
        ax.legend(custom_lines, custom_labels, loc=4)

        
    
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14, labelpad=-10)
    ax.tick_params(labelsize=14)
    
    if color_var != 'None':
        add_cbar(fig, ax_cbar, norm, cmap, cbar_label, cbar_ticks)
    save_str=''
    
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
    
    # ax.plot([-10, 100], [-10, 100], ls='--', color='red', marker='None') 
    ax.tick_params(labelsize=labelsize)
    ax.set_xlim(x_lims)    
    # x_tick_locs = [0.03, 0.055, 1/10, 1/5]
    # x_tick_labs = ['0.03', '0.055', '0.1', '0.2']
    # ax.set_xticks(x_tick_locs)
    # ax.set_xticklabels(x_tick_labs)
    if axisratio_vs_prospector != 0:
        ax.set_ylim(y_lims)
        save_str2 = str(axisratio_vs_prospector)
    if axisratio_vs_prospector == 0:
        ax2 = ax.twinx()
        ax2.tick_params(labelsize=labelsize)
        main_ax_lims = np.array([0.02, 1])
        ax2.set_ylim(1/main_ax_lims)
        ax2.set_yscale('log')
        ax.set_ylim(main_ax_lims)
        ax.set_yscale('log')
        y_tick_locs = [0.025, 0.055, 1/10, 1/5, 1/2, 1]
        y_tick_labs = ['0.025', '0.055', '0.1', '0.2', '0.5', '1']
        ax.set_yticks(y_tick_locs)
        ax.set_yticklabels(y_tick_labs)
        ax.minorticks_off()
        twin_y_tick_labs = ['-1', '0', '1', '2', '3', '4']
        twin_y_tick_locs = [1/compute_ratio_from_av(float(rat), law='reddy') for rat in twin_y_tick_labs]
        twin_y_tick_locs_calz = [1/compute_ratio_from_av(float(rat), law='calzetti') for rat in twin_y_tick_labs]
        ax2.set_yticks(twin_y_tick_locs)
        ax2.set_yticklabels(twin_y_tick_labs)
        ax2.set_ylabel(f'Inferred {avneb_str}', fontsize=fontsize, rotation=270, labelpad=20)
        ax2.minorticks_off()

    if prop == 'mass':
        # # shapley's data
        mosdef_data_mass = np.array([9.252764612954188, 9.73301737756714, 10.0173775671406, 10.437598736176936]) #Shapley 2022
        mosdef_data_decs = np.array([3.337349397590363, 3.4548192771084363, 3.7801204819277103, 4.512048192771086])
        mosdef_data_balmeravs = compute_balmer_av(mosdef_data_decs, law='cardelli') # Cardelli to get the MOSDEF AV
        mosdef_data_y_plot = compute_ratio_from_av(mosdef_data_balmeravs, law='reddy')  # Reddy to pull it back into this version of the plot. Matches on y-axis
        legend_shapley, = ax.plot(mosdef_data_mass, mosdef_data_y_plot, color='cornflowerblue', marker='d', ms=8, mec='black', ls='dotted', zorder=10, label='z=2.3 MOSDEF (Shapley+ 2022)')
        # legend_shapley = Line2D([0], [0], color=color, marker='None', ls='--')
        label_shapley = 'z=2.3 MOSDEF (Shapley+22)'

        runco_data_mass = np.array([9.04029773256327, 9.341541353064535	, 9.507356359477967, 9.660911296972452, 9.76852663271054, 9.882224549023732, 10.039519064690833, 10.177858006949581, 10.35957669226384, 10.679835800016289]) #Runco 2022
        runco_data_decs = np.array([3.282805987024606, 3.6142358130258136, 2.633209874253258, 4.096971898622865, 4.597955149928179, 4.213816474455239, 4.5059129818220125, 4.514513937180409, 5.778564129951793, 5.644147137838686])
        runco_data_balmeravs = compute_balmer_av(runco_data_decs, law='cardelli')
        runco_data_y_plot = compute_ratio_from_av(runco_data_balmeravs, law='reddy')
        legend_runco, = ax.plot(runco_data_mass, runco_data_y_plot, color='limegreen', marker='*', ms=12, mec='black', ls='None', zorder=11, label='MOSDEF Stacks (Runco+ 2022)')
        # legend_runco = Line2D([0], [0], color=color, marker='None', ls='--')
        label_runco = 'MOSDEF Stacks (Runco+22)'

        # legend_mega = Line2D([0], [0], color=med_color, marker='o', ls='--')
        label_med = 'MegaScience'
        
        custom_lines = [med_legend, legend_shapley, legend_runco]
        custom_labels = [label_med, label_shapley, label_runco]
        ax.legend(custom_lines, custom_labels, loc=2, fontsize=9)
    
    prop_folder = prop
    if 'axisratio' in prop:
        prop_folder = 'axis_ratios' # to save all axisratios to same directory
    if 'prospector' in prop:
        prop_folder = 'prospector_compare'
    if axisratio_vs_prospector != 0:
        prop_folder = 'axisratio_vs_prospector'
    
    if ax_in == 'None':
        fig.savefig(f'/Users/brianlorenz/uncover/Figures/PHOT_paper/dust_vs_prop/{prop_folder}/dust_vs_{prop}_{save_str}{save_str2}{color_var}.pdf', bbox_inches='tight')
        plt.close('all')



def add_cbar(fig, ax_cbar, norm, cmap, cbar_name, cbar_ticks, fontsize=14, ticklocation='top', labelpad=10):
    #SNR cbar
    sm =  mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar_ticklabels = [str(tick) for tick in cbar_ticks]
    cbar = fig.colorbar(sm, cax=ax_cbar, orientation='horizontal', ticks=cbar_ticks, ticklocation=ticklocation)
    cbar.ax.set_xticklabels(cbar_ticklabels) 
    cbar.ax.xaxis.minorticks_off()
    cbar.set_label(cbar_name, fontsize=fontsize, labelpad=labelpad) # -55 pad if ticks flip back to bottom
    cbar.ax.tick_params(labelsize=fontsize)

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


def two_panel(bin_type='dex'):
    fig = plt.figure(figsize=(12, 6))
    ax_mass = fig.add_axes([0.09, 0.08, 0.35, 0.70])
    ax_sfr = fig.add_axes([0.61, 0.08, 0.35, 0.70])
    plot_paper_dust_vs_prop(prop='mass',color_var='None',ax_in=ax_mass, bin_type=bin_type)
    plot_paper_dust_vs_prop(prop='sfr',color_var='None',ax_in=ax_sfr, bin_type=bin_type)
    fig.savefig(f'/Users/brianlorenz/uncover/Figures/PHOT_paper/dust_vs_prop/multiple/dust_vs_mass_sfr_twopanel_{bin_type}.pdf', bbox_inches='tight')

if __name__ == '__main__':
    # two_panel(bin_type='dex')
    # two_panel(bin_type='galaxies')
    # plot_paper_dust_vs_prop(prop='axisratio_f150w',color_var='None', bin_type='galaxies')
    plot_paper_dust_vs_prop(prop='mass',color_var='None', bin_type='galaxies')

    
    # props = ['mass', 'sfr', 'axisratio_f444w', 'axisratio_f150w', 'axisratio_halpha', 'prospector_total_av_50', 'prospector_neb_av_50']
    # color_vars = ['snr', 'redshift']
    # phot_df = read_phot_df()
    # for prop in props:
    #     for color_var in color_vars:
    #         plot_paper_dust_vs_prop(prop=prop, color_var=color_var, phot_df=phot_df)

    # for ar_value in [1,2,3]:
    #     plot_paper_dust_vs_prop(prop='prospector_total_av_50',color_var='redshift', axisratio_vs_prospector=ar_value)
    #     plot_paper_dust_vs_prop(prop='prospector_neb_av_50',color_var='redshift', axisratio_vs_prospector=ar_value)
    
    