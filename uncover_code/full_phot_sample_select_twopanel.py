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
from read_mosdef_data import get_shapley_sample, get_mosdef_compare_sample

def plot_paper_sample_select_twopanel(show_hexes=True, show_point_color=False, shapley=0):
    sample_df = read_final_sample()
    possible_df = read_possible_sample()
    shapley_df = get_shapley_sample()
    mosdef_df = get_mosdef_compare_sample()
    
    supercat_df = read_supercat()

    snr_cut_gals = read_paper_df('snr_cut_only')
    bcg_cut_gals = read_paper_df('bcg_cut_only')
    chi2_cut_gals = read_paper_df('chi2_cut_only')
    snr_cut_ids = snr_cut_gals['id_dr3'].to_list()
    bcg_cut_ids = bcg_cut_gals['id_dr3'].to_list()
    chi2_cut_ids = chi2_cut_gals['id_dr3'].to_list()

    sample_ids = sample_df['id_dr3'].to_list()
    if shapley==0:
        dfs = [possible_df, sample_df]
        save_str2 = ''
        mass_lims = [5.5, 11.5]
    if shapley>0:
        def add_cols_to_mosdef(df):
            df['z_50'] = df['Z_MOSFIRE']
            df['z_16'] = df['z_50']
            df['z_84'] = df['z_50']
            df['mstar_50'] = df['LMASS']
            df['mstar_16'] = df['L68_LMASS']
            df['mstar_84'] = df['U68_LMASS']
            df['id_dr3'] = np.zeros(len(df))
        add_cols_to_mosdef(mosdef_df)
        add_cols_to_mosdef(shapley_df)
        if shapley == 1:
            dfs = [shapley_df, sample_df]
            save_str2 = '_shapley'    
        if shapley == 2:
            dfs = [mosdef_df, sample_df]
            save_str2 = '_mosdef'    
        mass_lims = [7.5, 11.5]  
        median_bins = [[9,9.5], [9.5,9.85], [9.85,10.2], [10.2,10.75]]
        

    

    sizes = [4, 8]
    colors = ['black', 'orange']
    hist_alphas = [1, 0.7]

    redshift_lims = [1.2, 2.4]
    sfr_lims = [-2, 2.5]
    mags_lims = [18, 32]
    

    

    # Sample select figure to match
    # fig, ax = plt.subplots(figsize=(7,6))
    fig = plt.figure(figsize=(12,6))
    height = 0.72
    width = height/2
    start_level = 0.12
    ax_mass = fig.add_axes([0.10, start_level, width, height])
    ax_mag = fig.add_axes([0.55, start_level, width, height])
    cb_ax = fig.add_axes([0.935, start_level,0.03,height])



    sps_all_df = read_SPS_cat_all()
    # Gray background
    all_masses = sps_all_df['mstar_50']
    all_sfr100s = sps_all_df['sfr100_50']
    all_log_sfr100s = np.log10(all_sfr100s)
    all_redshifts = sps_all_df['z_50']
    # ax.plot(all_redshifts, all_masses, marker='o', ls='None', markersize=background_markersize, color='gray')
    cmap = plt.get_cmap('gray_r')
    new_cmap = truncate_colormap(cmap, 0, 0.7)
    good_redshift_idx = np.logical_and(all_redshifts > redshift_lims[0], all_redshifts < redshift_lims[1])
    good_mass_idx = np.logical_and(all_masses > mass_lims[0], all_masses < mass_lims[1])
    good_sfr_idx = np.logical_and(all_log_sfr100s > sfr_lims[0], all_log_sfr100s < sfr_lims[1])

    if show_hexes:
        hexbin_norm = mpl.colors.Normalize(vmin=1, vmax=200) 
        good_both_idx = np.logical_and(good_sfr_idx, good_mass_idx)    
        ax_mass.hexbin(all_masses[good_both_idx], all_log_sfr100s[good_both_idx], gridsize=15, cmap=new_cmap, norm=hexbin_norm, label='Photometric Sample')
       
        
        hexbin_norm = mpl.colors.Normalize(vmin=1, vmax=200) 
        f444w_fluxes = supercat_df['f_f444w'] * 1e-8
        f444w_mags = -2.5 * np.log10(f444w_fluxes) + 8.9
        good_mag_idx = np.logical_and(f444w_mags>mags_lims[0], f444w_mags<mags_lims[1])
        good_both_idx = np.logical_and(good_redshift_idx, good_mag_idx)
        ax_mag.hexbin(all_redshifts[good_both_idx], f444w_mags[good_both_idx], gridsize=15, cmap=new_cmap, norm=hexbin_norm, label='Photometric Sample')

    cmap = mpl.cm.viridis
    for i in range(len(dfs)): 
        df = dfs[i]
        df = pd.merge(df, supercat_df, left_on='id_dr3', right_on='id', how='left')
        df['f444w_mag'] =  -2.5 * np.log10(df['f_f444w'] * 1e-8) + 8.9
        df['err_f444w_mag_high'] =  (-2.5 * np.log10((df['f_f444w'] - df['e_f444w']) * 1e-8) + 8.9) - df['f444w_mag']
        df['err_f444w_mag_low'] =  df['f444w_mag'] - (-2.5 * np.log10((df['f_f444w'] + df['e_f444w']) * 1e-8) + 8.9) 
        
        # Compute errs
        low_zerr = df['z_50'] - df['z_16']
        high_zerr = df['z_84'] - df['z_50']
        low_sfr_err = np.log10(df['sfr100_50']) - np.log10(df['sfr100_16'])
        high_sfr_err = np.log10(df['sfr100_84']) - np.log10(df['sfr100_50'])
        low_mass_err = df['mstar_50'] - df['mstar_16']
        high_mass_err = df['mstar_84'] - df['mstar_50']
        low_mag_err = df['err_f444w_mag_low'] 
        high_mag_err = df['err_f444w_mag_high'] 
        
    
        
        for j in range(len(df)):

            size = sizes[i]
            color = colors[i]
            shape = 'o'
            mec = 'black'
            # if color_var == 'snr':
            #     norm = mpl.colors.LogNorm(vmin=0.5, vmax=50) 
            #     rgba = cmap(norm(df['min_snr'].iloc[j]))
            # elif color_var == 'ha_qual':
            #     norm = mpl.colors.Normalize(vmin=0, vmax=20) 
            #     rgba = cmap(norm(df['Halpha_quality_factor'].iloc[j]))
            # elif color_var == 'pab_qual':
            #     norm = mpl.colors.Normalize(vmin=0, vmax=5) 
            #     rgba = cmap(norm(df['PaBeta_quality_factor'].iloc[j]))
            id_dr3 = df['id_dr3'].iloc[j]
            if i == 0 and show_point_color:
                if id_dr3 in chi2_cut_ids:
                    shape = 'x'
                    mec = 'red'
                if id_dr3 in snr_cut_ids:
                    color='red'
                if id_dr3 in bcg_cut_ids:
                    color = 'blue'
                    if shape == 'x':
                        mec='blue'
                
            if i == 1:
                norm = mpl.colors.LogNorm(vmin=100, vmax=1000) 
                color = cmap(norm(df['Halpha_eqw'].iloc[j]))

        
            ax_mass.errorbar(df['mstar_50'].iloc[j], np.log10(df['sfr100_50'].iloc[j]), xerr=np.array([[low_mass_err.iloc[j], high_mass_err.iloc[j]]]).T, yerr=np.array([[low_sfr_err.iloc[j], high_sfr_err.iloc[j]]]).T, marker=shape, mec=mec, ms=size, color=color, ls='None', ecolor='gray')
            ax_mag.errorbar(df['z_50'].iloc[j], df['f444w_mag'].iloc[j], xerr=np.array([[low_zerr.iloc[j], high_zerr.iloc[j]]]).T, yerr=np.array([[low_mag_err.iloc[j], high_mag_err.iloc[j]]]).T, marker=shape, mec=mec, ms=size, color=color, ls='None', ecolor='gray')
            # ax.text(df['mstar_50'].iloc[j], df['z_50'].iloc[j], f'{id_dr3}')
    ax_mass.set_xlabel(stellar_mass_label, fontsize=14)
    ax_mass.set_ylabel('Prospector '+log_sfr_label_sedmethod, fontsize=14)
    ax_mag.set_xlabel('Redshift', fontsize=14)
    ax_mag.set_ylabel('F444W Apparent Magnitude', fontsize=14)
    for ax in [ax_mass, ax_mag]:
        ax.tick_params(labelsize=14)
    ax_mag.set_ylim(mags_lims[1], mags_lims[0])
    ax_mass.set_ylim(sfr_lims[0], sfr_lims[1])
    ax_mass.set_xlim(mass_lims[0], mass_lims[1])

    cbar_label = 'H$\\alpha$ Equivalent Width'
    cbar_ew_ticks = [100, 500, 1000]
    cbar_ew_ticklabels = [str(tick) for tick in cbar_ew_ticks]
        
    sm =  mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, orientation='vertical', cax=cb_ax, ticks=cbar_ew_ticks)
    cbar.ax.set_yticklabels(cbar_ew_ticklabels)  
    cbar.set_label(cbar_label, fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    minor_labels = ['', '', '', ''] 
    cbar.ax.set_yticklabels(minor_labels, minor=True)
    # breakpoint()
    # cbar.ax.tick_params(axis='y', which='minor', labelleft=False)
    
    save_str=''
    if show_point_color:
        line_sample = Line2D([0], [0], color='orange', marker='o', markersize=8, ls='None', mec='black')
        line_bcg = Line2D([0], [0], color='blue', marker='o', markersize=4, ls='None', mec='black')
        line_snr = Line2D([0], [0], color='red', marker='o', markersize=4, ls='None', mec='black')
        line_chi2 = Line2D([0], [0], color='red', marker='x', markersize=4, ls='None', mec='red')
        custom_lines = [line_sample, line_bcg, line_snr, line_chi2]
        custom_labels = ['Sample', 'Close to bcg', 'Low snr', 'Bad cont slope']
        ax.legend(custom_lines, custom_labels, bbox_to_anchor=(1.05, 1.14))
        save_str = '_color'

    if shapley>0:
        for median_tuple in median_bins:
            ax.axvline(x=median_tuple[0], ymin=0, ymax=1, color='magenta', linestyle='--')
            ax.axvline(x=median_tuple[1], ymin=0, ymax=1, color='magenta', linestyle='--')  
            
    fig.savefig(f'/Users/brianlorenz/uncover/Figures/PHOT_paper/sample_select/sample_select_twopanel.pdf', bbox_inches='tight')
    scale_aspect(ax)
    plt.close('all')


if __name__ == '__main__':
    plot_paper_sample_select_twopanel()
    pass