from full_phot_read_data import read_merged_lineflux_cat, read_final_sample, read_possible_sample, read_paper_df
from full_phot_merge_lineflux import filter_bcg_flags
import matplotlib.pyplot as plt
from uncover_read_data import read_SPS_cat_all, read_bcg_surface_brightness, read_supercat, read_morphology_cat
from compute_av import compute_ha_pab_av, compute_pab_paa_av, compute_paalpha_pabeta_av
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


def plot_paper_sample_select_sfr_mass(show_hexes=True, show_point_color=False, mass_cut=0, shapley=0):
    sample_df = read_final_sample()
    possible_df = read_possible_sample()
    shapley_df, linemeas_df_shapley = get_shapley_sample()
    mosdef_df, linemeas_df = get_mosdef_compare_sample()

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
        mass_lims = [np.max([5.5, mass_cut]), 11.5]
    if shapley>0:
        def add_cols_to_mosdef(df):
            df['sfr100_50'] = 10**df['LSFR']
            df['sfr100_16'] = 10**df['L68_LSFR']
            df['sfr100_84'] = 10**df['U68_LSFR']
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
        
        make_av_figure = 1
        density = True
        if make_av_figure:
            fig2, ax2 = plt.subplots(figsize=(6,6))
            sample_df['prospector_total_av_50'] = 1.086 * (sample_df['dust2_50'] + (sample_df['dust2_50']*sample_df['dust1_fraction_50']))
            sample_df['prospector_neb_av_50'] = 1.086 * sample_df['dust2_50']*sample_df['dust1_fraction_50']
            sample_df['prospector_stellar_av_50'] = 1.086 * sample_df['dust2_50']
            mass_gt9_idx = sample_df['mstar_50'] > 9
            mass_str = '_mass_gt9_'
            avbins = np.arange(0,3,0.2)
            ar_df = ascii.read(imd.loc_axis_ratio_cat).to_pandas()
            merged_df = pd.merge(dfs[0], ar_df, left_on=['V4ID', 'FIELD_STR'], right_on=['v4id', 'field'], how='left')
            ax2.hist(sample_df[mass_gt9_idx]['prospector_stellar_av_50'], bins=avbins, color='orange', density=True)
            ax2.hist(merged_df['AV'], bins=avbins, color='black', alpha=0.5, density=True)
            ax2.set_ylabel('Number', fontsize=14)
            ax2.set_xlabel('AV', fontsize=14)
            ax2.tick_params(labelsize=14)
            if density:
                save2 = '_dense'
            else:
                save2 = ''
            fig2.savefig(f'/Users/brianlorenz/uncover/Figures/PHOT_paper/sample_compare/AV_hist{save2}{save_str2}{mass_str}.pdf')
            sys.exit()

        

    cmap = mpl.cm.inferno

    sizes = [4, 8]
    colors = ['black', 'orange']
    hist_alphas = [1, 0.7]

    sfr_lims = [-2, 2.5]

    # Sample select figure to match
    # fig, ax = plt.subplots(figsize=(7,6))
    fig, axs = plt.subplot_mosaic([['histx', '.'],['scatter', 'histy']], figsize=(6, 6), width_ratios=(4, 1), height_ratios=(1, 4), layout='constrained')
    ax = axs['scatter']
    ax_histx = axs['histx']
    ax_histy = axs['histy']
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histx.get_yaxis().set_visible(False)
    ax_histy.get_xaxis().set_visible(False)
    xbins = np.arange(mass_lims[0], mass_lims[1], 0.5)
    ybins = np.arange(sfr_lims[0], sfr_lims[1], 0.2)

    ax.set_ylim(sfr_lims[0], sfr_lims[1])
    ax_histy.set_ylim(sfr_lims[0], sfr_lims[1])
    ax.set_xlim(mass_lims[0], mass_lims[1])
    ax_histx.set_xlim(mass_lims[0], mass_lims[1])

    sps_all_df = read_SPS_cat_all()
    # Gray background
    all_masses = sps_all_df['mstar_50']
    all_sfr100s = sps_all_df['sfr100_50']
    all_log_sfr100s = np.log10(all_sfr100s)
    # ax.plot(all_redshifts, all_masses, marker='o', ls='None', markersize=background_markersize, color='gray')
    cmap = plt.get_cmap('gray_r')
    new_cmap = truncate_colormap(cmap, 0, 0.7)
    good_mass_idx = np.logical_and(all_masses > mass_lims[0], all_masses < mass_lims[1])
    good_sfr_idx = np.logical_and(all_log_sfr100s > sfr_lims[0], all_log_sfr100s <  sfr_lims[1])

    if show_hexes:
        hexbin_norm = mpl.colors.Normalize(vmin=1, vmax=200) 
        good_both_idx = np.logical_and(good_sfr_idx, good_mass_idx)  
        mass_cut_idx = all_masses>mass_cut
        good_both_idx = np.logical_and(good_both_idx, mass_cut_idx)
        ax.hexbin(all_masses[good_both_idx], all_log_sfr100s[good_both_idx], gridsize=15, cmap=new_cmap, norm=hexbin_norm, label='Photometric Sample')
        ax_histx.hist(all_masses[good_both_idx], bins=xbins, color='grey', density=True, alpha=0.6, zorder=10)
        ax_histy.hist(all_log_sfr100s[good_both_idx], bins=ybins, color='grey',  orientation='horizontal', density=True, alpha=0.6, zorder=10) 
        save_str2 = '_hexes'

    for i in range(len(dfs)): 
        df = dfs[i]

        # Compute errs
        low_sfr_err = np.log10(df['sfr100_50']) - np.log10(df['sfr100_16'])
        high_sfr_err = np.log10(df['sfr100_84']) - np.log10(df['sfr100_50'])
        
        for j in range(len(df)):

            mass = df['mstar_50'].iloc[j]
            if mass < mass_cut:
                continue

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
                    

            ax.errorbar(df['mstar_50'].iloc[j], np.log10(df['sfr100_50'].iloc[j]), yerr=np.array([[low_sfr_err.iloc[j], high_sfr_err.iloc[j]]]).T, marker=shape, mec=mec, ms=size, color=color, ls='None', ecolor='gray')
            # ax.text(df['mstar_50'].iloc[j], df['z_50'].iloc[j], f'{id_dr3}')
        ax_histx.hist(df['mstar_50'], bins=xbins, color=color, density=True, alpha=hist_alphas[i])
        ax_histy.hist(np.log10(df['sfr100_50']), bins=ybins, color=color,  orientation='horizontal', density=True, alpha=hist_alphas[i])  
    ax.set_xlabel(stellar_mass_label, fontsize=14)
    ax.set_ylabel(f'Prospector log10(SFR)', fontsize=14)
    ax.tick_params(labelsize=14)

    # # Plot SFMS
    # masses = np.arange(6,11,0.1)
    # predicted_log_sfrs = pop_sfms(masses, 1.96)
    # ax.plot(masses, predicted_log_sfrs, color='red', ls='--', marker='None')
    
    # if color_var == 'snr':
    #     add_snr_cbar(fig, ax, norm, cmap)
    # else:
    #     add_cbar(fig, ax, norm, cmap, color_var)
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
            ax_histx.axvline(x=median_tuple[0], ymin=0, ymax=1, color='magenta', linestyle='--')
            ax_histx.axvline(x=median_tuple[1], ymin=0, ymax=1, color='magenta', linestyle='--')      
    
    fig.savefig(f'/Users/brianlorenz/uncover/Figures/PHOT_paper/sample_select/sample_select_sfrmass_{save_str}{save_str2}_mass{mass_cut}.pdf')
    scale_aspect(ax)
    plt.close('all')

def pop_sfms(mass, redshift): # https://academic.oup.com/mnras/article/519/1/1526/6815739#equ10
    from astropy.cosmology import WMAP9 as cosmo
    age_of_universe = cosmo.age(redshift).value

    a0 = 0.20
    a1 = -0.034
    b0 = -26.134
    b1 = 4.722
    b2 = -0.1925

    sfms = (a1*age_of_universe+b1) * mass + b2*mass**2 + (b0+a0*age_of_universe)
    return sfms

if __name__ == '__main__':
    # plot_paper_sample_select_sfr_mass(show_hexes=False, shapley=1)
    # plot_paper_sample_select_sfr_mass(show_hexes=False, shapley=2)
    # plot_paper_sample_select_sfr_mass(show_hexes=True)
    plot_paper_sample_select_sfr_mass(show_hexes=True, mass_cut=7.5)