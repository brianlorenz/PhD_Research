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
from full_phot_digitized_vals import *
from full_phot_read_data import  read_canucs_compare, read_bluejay_compare
import time
from scipy.stats import linregress

random.seed(80148273) 


def plot_paper_dust_vs_prop(prop='mass', color_var='snr', phot_df=[], axisratio_vs_prospector = 0, ax_in='None', bin_type='dex', compare=0, plot_av=0, hide_twin=0):

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
        mec = 'None'
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
            rgba = '#8b8b8b'
        ecolor = '#b3b3b3'
        
        if prop == 'mass':
            var_name = 'mstar_50'            
            x_label = stellar_mass_label
            x_lims = mass_lims
            x_regress = np.arange(7, 12, 0.1)
            
            if bin_type == 'dex':
                median_bins = [[7,8], [8,9], [9,10], [10,11]]

            if bin_type == 'galaxies':
                mass_20, mass_40, mass_60, mass_80 = np.percentile(sample_df['mstar_50'], [20, 40, 60, 80])
                median_bins = [[7,mass_20], [mass_20,mass_40], [mass_40,mass_60], [mass_60,mass_80], [mass_80,11]]

        
        elif prop == 'sfr':
            var_name = 'log_sfr100_50'
            x_label = log_sfr_label_sedmethod
            x_lims = sfr_lims
            x_regress = np.arange(-2, 3, 0.1)

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
            x_regress = np.arange(0, 1.1, 0.05)
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
                x_label = 'F150W Axis Ratio'
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

        if plot_av:
            if compare == 1:
                law='cardelli'
            if compare == 2:
                law='reddy'
            
            yerr_low = compute_ha_pab_av(y_plot, law=law) - compute_ha_pab_av(y_plot-y_err[0][0], law=law) 
            yerr_high = compute_ha_pab_av(y_plot+y_err[1][0], law=law) - compute_ha_pab_av(y_plot, law=law) 
            y_plot = compute_ha_pab_av(y_plot, law=law) 
            y_err = np.array([[yerr_low, yerr_high]]).T

        if compare == 0:

            # sfr_20, sfr_40, sfr_60, sfr_80 = np.percentile(sample_df['log_sfr100_50'], [20, 40, 60, 80])
            # if sample_df['log_sfr100_50'].iloc[j] > sfr_80:
            #     rgba='magenta'

            ax.errorbar(x_plot, y_plot, xerr=x_err, yerr=y_err, marker=shape, mec=mec, ms=5, color=rgba, ls='None', ecolor=ecolor)
            
            
            # ax.text(x_plot, y_plot, f'{id_dr3}', fontsize=6)

    # Add regression line if 
    if plot_av == 0:
        sample_df['log_lineratio_pab_ha'] = np.log10(sample_df['lineratio_pab_ha'])
        sample_df['err_log_lineratio_pab_ha_low'] = np.log10(sample_df['lineratio_pab_ha']) - np.log10(sample_df['lineratio_pab_ha'] - sample_df['err_lineratio_pab_ha_low']) 
        sample_df['err_log_lineratio_pab_ha_high'] = np.log10(sample_df['lineratio_pab_ha'] + sample_df['err_lineratio_pab_ha_low']) - np.log10(sample_df['lineratio_pab_ha']) 
        sample_df_nonan = sample_df[~pd.isnull(sample_df[var_name])]
        regress_res, points_16, points_84 = bootstrap_fit(sample_df_nonan, var_name, 'log_lineratio_pab_ha', x_regress)
        points_16 = [10**point for point in points_16]
        points_84 = [10**point for point in points_84]
        ax.plot(x_regress, 10**(regress_res.intercept + regress_res.slope*x_regress), color='#4c4c4c', ls='--')
        ax.fill_between(x_regress, points_16, points_84, facecolor='#82BD7A', alpha=0.4)

    # Add medians if there are bins
    n_boots=1000
    if len(median_bins) > 0:
        if axisratio_vs_prospector != 0:
            median_xvals, median_yvals, median_xerr, median_yerr, ngals = get_median_points(sample_df, median_bins, var_name, y_var_name=y_var_name, n_boots=n_boots)
        else:
            median_xvals, median_yvals, median_xerr, median_yerr, ngals = get_median_points(sample_df, median_bins, var_name,  n_boots=n_boots)
        if plot_av:
            yerrs_low = [compute_ha_pab_av(median_yvals[k] - median_yerr[0][k], law=law) for k in range(len(median_yvals))]
            yerrs_high = [compute_ha_pab_av(median_yvals[k] + median_yerr[1][k], law=law) for k in range(len(median_yvals))]
            median_yvals = [compute_ha_pab_av(median_yval, law=law) for median_yval in median_yvals]
            yerrs_low = np.array([median_yvals[i] - yerrs_low[i] for i in range(len(median_yvals))])
            yerrs_high = np.array([yerrs_high[i] - median_yvals[i] for i in range(len(median_yvals))])
            median_yerr = np.vstack([yerrs_low, yerrs_high])
        med_color = 'cornflowerblue'
        med_color = '#ff7f00'
        med_legend = ax.errorbar(median_xvals, median_yvals, xerr=median_xerr, yerr=median_yerr, marker='s', ms=9, color=med_color, ls='None', zorder=50, mec='black', ecolor=med_color, capsize=3)
        # for median_tuple in median_bins:
        #     ax.axvline(x=median_tuple[0], ymin=0, ymax=0.09, color='gray', linestyle='--')
        #     ax.axvline(x=median_tuple[1], ymin=0, ymax=0.09, color='gray', linestyle='--')

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
    if hide_twin:
        main_ax_lims = np.array([0.02, 1])
        ax.set_ylim(main_ax_lims)
        ax.set_yscale('log')
        y_tick_locs = [0.025, 0.056, 1/10, 1/5, 1/2, 1]
        y_tick_labs = ['0.025', '0.056', '0.1', '0.2', '0.5', '1']
        ax.set_yticks(y_tick_locs)
        ax.set_yticklabels(y_tick_labs)
        ax.minorticks_off()
    elif axisratio_vs_prospector == 0 and compare == 0:
        ax2 = ax.twinx()
        ax2.tick_params(labelsize=labelsize)
        main_ax_lims = np.array([0.02, 1])
        ax2.set_ylim(1/main_ax_lims)
        ax2.set_yscale('log')
        ax.set_ylim(main_ax_lims)
        ax.set_yscale('log')
        y_tick_locs = [0.025, 0.056, 1/10, 1/5, 1/2, 1]
        y_tick_labs = ['0.025', '0.056', '0.1', '0.2', '0.5', '1']
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
    if compare > 0:
        main_ax_lims = np.array([-1.5, 4.5])
        ax.set_ylim(main_ax_lims)
        # ax.set_ylim(main_ax_lims)
        # ax.set_yscale('log')
        y_tick_locs = [-1, 0, 1, 2, 3, 4]
        y_tick_labs = ['-1', '0', '1', '2', '3', '4']
        ax.set_yticks(y_tick_locs)
        ax.set_yticklabels(y_tick_labs)
        # ax.minorticks_off()

    if prop == 'mass' and compare>0:
        shapley = 0
        # compare_colors = ['cornflowerblue', 'limegreen', 'yellow', 'red']
        # compare_colors = ['#BCDAB8', '#79AC72', '#4B7346', '#273B24']
        compare_colors = ['#D8F0D4', '#82BD7A', '#42733C', '#273B24']
        compare_mec = 'black'

        # if shapley:
        #     # # shapley's data
        #     mosdef_data_mass = np.array([9.252764612954188, 9.73301737756714, 10.0173775671406, 10.437598736176936]) #Shapley 2022
        #     mosdef_data_decs = np.array([3.337349397590363, 3.4548192771084363, 3.7801204819277103, 4.512048192771086])
        #     mosdef_data_balmeravs = compute_balmer_av(mosdef_data_decs, law=law) # Cardelli to get the MOSDEF AV
        #     mosdef_data_y_plot = compute_ratio_from_av(mosdef_data_balmeravs, law=law)  # Reddy to pull it back into this version of the plot. Matches on y-axis
        #     if plot_av:
        #         mosdef_data_y_plot = mosdef_data_balmeravs
        #     legend_shapley, = ax.plot(mosdef_data_mass, mosdef_data_y_plot, color='cornflowerblue', marker='d', ms=8, mec='black', ls='dotted', zorder=10, label='z=2.3 MOSDEF (Shapley+ 2022)')
        #     # legend_shapley = Line2D([0], [0], color=color, marker='None', ls='--')
        #     label_shapley = 'z=2.3 MOSDEF (Shapley+22)'

        median_av = np.median(compute_ha_pab_av(sample_df['lineratio_pab_ha'], law=law))
        max_ax = np.max(compute_ha_pab_av(sample_df['lineratio_pab_ha'], law=law))
        print(f'Median av: {median_av}, law = {law}')
        print(f'Max av: {max_ax}, law = {law}')

        legend_runco = plot_comparison(ax, runco_data_mass, runco_data_decs, runco_data_low_errs, runco_data_high_errs, law=law, color=compare_colors[2], marker='*', size=10, mec=compare_mec)
        label_runco = 'MOSDEF z~2.3'

        matharu_df = read_canucs_compare()
        matharu_bin_pcts = [0, 33, 66, 100]
        legend_matharu_med, matharu_mass, matharu_decs = plot_comparison_mybins(ax, matharu_df, 'mass', 'BD', matharu_bin_pcts, law=law, color=compare_colors[1], marker='^', size=6, mec=compare_mec)
        label_matharu_med = 'CANUCS z~1.7'

        legend_battisti = plot_comparison(ax, battisti_masses, battisti_decs, battisti_lows, battisti_highs, law=law, color=compare_colors[0], marker='d', size=6, mec=compare_mec)
        label_battisti = 'WISP+3D-HST z~1.3'

        # maheson_df = pd.DataFrame(zip(maheson_all_masses, maheson_all_decs), columns=['mass', 'balmer_dec'])
        maheson_df = read_bluejay_compare()
        maheson_df = maheson_df[maheson_df['redshift']<2.4]
        median_z = np.median(maheson_df['redshift'])
        maheson_bin_pcts = [0, 33, 66, 100]
        legend_maheson, maheson_mass, maheson_decs = plot_comparison_mybins(ax, maheson_df, 'stellar_mass_50', 'BD_50', maheson_bin_pcts, law=law, color=compare_colors[3], marker='o', size=6, mec=compare_mec)
        # legend_maheson, maheson_mass, maheson_decs = plot_comparison_mybins(ax, maheson_df, 'mass', 'balmer_dec', maheson_bin_pcts, law=law, color=compare_colors[3], marker='o', size=6, mec=compare_mec)
        label_maheson = f'Blue Jay z~{median_z:0.2f}'
        

        all_mass = np.concatenate([runco_data_mass, battisti_masses, matharu_mass, maheson_mass])
        all_decs = np.concatenate([runco_data_decs, battisti_decs, matharu_decs, maheson_decs])
        all_avs = compute_balmer_av(all_decs, law=law)
        neb_x_regress = np.arange(8, 12, 0.05)
        regress_res, points_16, points_84 = bootstrap_fit(0, 0, 0, neb_x_regress, xvals=all_mass, yvals=all_avs)
        # ax.plot(neb_x_regress, regress_res.intercept + regress_res.slope*neb_x_regress, color='#466343', ls='--')
        ax.fill_between(neb_x_regress, points_16, points_84, facecolor='#82BD7A', alpha=0.3)
        
        # Assess offset amount
        predicted_av_neb_from_compare = regress_res.intercept + regress_res.slope*np.array(median_xvals)
        measured_avneb_megascience = np.array(median_yvals)
        offsets = measured_avneb_megascience - predicted_av_neb_from_compare
        median_offset = np.median(offsets)
        print(f'Median offset between measured and predicted: {median_offset} mag')
        
        # gorup all plotted points


        # legend_mega = Line2D([0], [0], color=med_color, marker='o', ls='--')
        label_med = 'MegaScience z~1.8'

        # # Add SDSS galaxies
        # from read_sdss import read_and_filter_sdss
        # sdss_df = read_and_filter_sdss()
        # sdss_df = sdss_df[sdss_df['balmer_dec']>2.7]
        # sdss_df = sdss_df[sdss_df['balmer_dec']<10.0]
        # sdss_df = sdss_df[sdss_df['log_mass']>8]
        # sdss_df = sdss_df[sdss_df['log_mass']<11.0]
        # sdss_df['balmer_av_cardelli'] = compute_balmer_av(sdss_df['balmer_dec'], law='cardelli')
        # sdss_df['y_plot'] = compute_ratio_from_av(sdss_df['balmer_av_cardelli'], law='reddy')
        # cmap_hex = plt.get_cmap('gray_r')
        # cmap_hex = truncate_colormap(cmap_hex, 0, 0.7)
        # ax.hexbin(sdss_df['log_mass'], sdss_df['y_plot'], gridsize=20, cmap=cmap_hex, yscale='log')
        # # ax.plot(sdss_df['log_mass'], sdss_df['y_plot'], color='black', marker='s', alpha=0.5, markersize=4, ls='None')
        # line_sdss = Line2D([0], [0], color='black', marker='s', alpha=0.5, markersize=4, ls='None')
        # label_sdss ='SDSS'
        
        if shapley:
            # custom_lines = [med_legend, legend_shapley, legend_runco]#, line_sdss]
            # custom_labels = [label_med, label_shapley, label_runco]#, label_sdss]
            pass
        else:
            custom_lines = [med_legend, legend_battisti, legend_matharu_med, legend_runco, legend_maheson]
            custom_labels = [label_med, label_battisti, label_matharu_med, label_runco, label_maheson]
        ax.legend(custom_lines, custom_labels, loc=2, fontsize=11)
    
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
    ax_sfr = fig.add_axes([0.47, 0.08, 0.35, 0.70])
    plot_paper_dust_vs_prop(prop='mass',color_var='None',ax_in=ax_mass, bin_type=bin_type, hide_twin=1)
    plot_paper_dust_vs_prop(prop='sfr',color_var='None',ax_in=ax_sfr, bin_type=bin_type, hide_twin=1)
    ax_sfr.tick_params(axis='y', labelleft=False)   
    ax_sfr.set_ylabel('')   
    fig.savefig(f'/Users/brianlorenz/uncover/Figures/PHOT_paper/dust_vs_prop/multiple/dust_vs_mass_sfr_twopanel_{bin_type}.pdf', bbox_inches='tight')

def neb_curve_diff(bin_type='galaxies'):
    fig = plt.figure(figsize=(12, 6))
    ax_cardelli = fig.add_axes([0.09, 0.08, 0.35, 0.70])
    ax_reddy = fig.add_axes([0.47, 0.08, 0.35, 0.70])
    plot_paper_dust_vs_prop(prop='mass',color_var='None',ax_in=ax_cardelli, bin_type=bin_type, compare=1, plot_av=1)
    plot_paper_dust_vs_prop(prop='mass',color_var='None',ax_in=ax_reddy, bin_type=bin_type, compare=2, plot_av=1)
    ax_reddy.tick_params(axis='y', labelleft=False)   
    ax_reddy.set_ylabel('')   
    ax_reddy.get_legend().remove()
    ax_reddy.set_title('Reddy+25', fontsize=16)
    ax_cardelli.set_title('Cardelli+89', fontsize=16)
    ax_cardelli.set_ylabel(f'Inferred {avneb_str}', labelpad=2)   
    for ax in [ax_reddy, ax_cardelli]:
        ax.set_ylim(-0.6, 3.2)
        ax.set_xlim(8.1, 11)
        # ax.axhline(0, color='gray', alpha=0.3)
        # ax.axhline(1, color='gray', alpha=0.3)
        # ax.axhline(2, color='gray', alpha=0.3)
        # ax.axhline(3, color='gray', alpha=0.3)
        ax.hlines([0,1,2,3], 7, 12, color='darkgray', zorder=1, lw=0.7)
        
    fig.savefig(f'/Users/brianlorenz/uncover/Figures/PHOT_paper/dust_vs_prop/multiple/dust_vs_mass_nebcurve.pdf', bbox_inches='tight')

def plot_comparison_mybins(ax, df, massname, decname, bin_pcts, law, color, marker, size, mec='black'):
    binlist = np.percentile(df[massname], bin_pcts)
    bins = [[binlist[i], binlist[i+1]] for i in range(len(binlist)-1)]
    mass, dec, _, dec_err, ngals = get_median_points(df, bins, massname, y_var_name=decname, n_boots=1000)
    legend_line = plot_comparison(ax, np.array(mass), np.array(dec), np.array(dec)-np.array(dec_err[0]), np.array(dec)+np.array(dec_err[1]), law=law, color=color, marker=marker, size=size, mec=mec)
    return legend_line, mass, dec


def bootstrap_fit(df, x_col, y_col, xpoints, exclude_limit=True, bootstrap=1000, xvals=[], yvals=[]):
    if len(xvals) + len(yvals) == 0:
        xvals = df[x_col].to_numpy()
        yvals = df[y_col].to_numpy()
    # yerr_low = df['err_' + y_col + '_low']
    # yerr_high = df['err_' + y_col + '_high']
    boot_slopes = []
    boot_yints = []
    idx_pool = np.arange(0, len(yvals), 1)
    for boot in range(bootstrap):
        selected_idxs = np.random.choice(idx_pool, len(yvals))
        new_xs = xvals[selected_idxs]
        new_ys = yvals[selected_idxs]
        # new_ys = [draw_asymettric_error(yvals.iloc[i], yerr_low.iloc[i], yerr_high.iloc[i]) for i in range(len(yvals))]
        regress_res = linregress(new_xs, new_ys)
        boot_slopes.append(regress_res.slope)
        boot_yints.append(regress_res.intercept)
    all_points = [boot_yints[i] + boot_slopes[i]*xpoints for i in range(len(boot_slopes))]
    def get_points(percentile):
        percentile_points = []
        for i in range(len(all_points[0])):
            point_js = [all_points[j][i] for j in range(len(all_points))]
            percentile_points.append(np.percentile(point_js, percentile))
        return percentile_points
    points_16 = get_points(16)
    points_84 = get_points(84)
    regress_res = linregress(xvals, yvals)
  
    return regress_res, points_16, points_84

def draw_asymettric_error(center, low_err, high_err):
    """Draws a point from two asymmetric normal distributions"""
    x = random.uniform(0,1)
    if x < 0.5:
        draw = np.random.normal(loc=0, scale=low_err)
        new_value = center - np.abs(draw)
    else:
        draw = np.random.normal(loc=0, scale=high_err)
        new_value = center + np.abs(draw)
    return new_value

if __name__ == '__main__':
    # The 3 final paper figures
    neb_curve_diff()
    # two_panel(bin_type='galaxies')
    # plot_paper_dust_vs_prop(prop='axisratio_f150w',color_var='None', bin_type='galaxies')
    
    
    
    # plot_paper_dust_vs_prop(prop='mass',color_var='None', bin_type='galaxies', compare=1)



    
    # props = ['mass', 'sfr', 'axisratio_f444w', 'axisratio_f150w', 'axisratio_halpha', 'prospector_total_av_50', 'prospector_neb_av_50']
    # color_vars = ['snr', 'redshift']
    # phot_df = read_phot_df()
    # for prop in props:
    #     for color_var in color_vars:
    #         plot_paper_dust_vs_prop(prop=prop, color_var=color_var, phot_df=phot_df)

    # for ar_value in [1,2,3]:
    #     plot_paper_dust_vs_prop(prop='prospector_total_av_50',color_var='redshift', axisratio_vs_prospector=ar_value)
    #     plot_paper_dust_vs_prop(prop='prospector_neb_av_50',color_var='redshift', axisratio_vs_prospector=ar_value)
    
    