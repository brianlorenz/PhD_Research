from full_phot_read_data import read_merged_lineflux_cat, read_final_sample, read_possible_sample, read_paper_df, read_ha_sample
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
from full_phot_property_vs_dust import add_cbar
from matplotlib.legend_handler import HandlerTuple
import matplotlib.patches as mpatches


def plot_paper_sample_select_sfr_mass(show_hexes=True, show_point_color=False, mass_cut=0, shapley=0):
    sample_df = read_final_sample()
    possible_df = read_possible_sample()
    ha_sample = read_ha_sample()
    shapley_df, linemeas_df_shapley = get_shapley_sample()
    mosdef_df, linemeas_df = get_mosdef_compare_sample()

    snr_cut_gals = read_paper_df('snr_cut_only')
    bcg_cut_gals = read_paper_df('bcg_cut_only')
    chi2_cut_gals = read_paper_df('chi2_cut_only')
    snr_cut_ids = snr_cut_gals['id_dr3'].to_list()
    bcg_cut_ids = bcg_cut_gals['id_dr3'].to_list()
    chi2_cut_ids = chi2_cut_gals['id_dr3'].to_list()

    sample_ids = sample_df['id_dr3'].to_list()
    ha_sample_only = ha_sample[~ha_sample['id_dr3'].isin(sample_df['id_dr3'])]

    if shapley==0:
        dfs = [possible_df, sample_df, ha_sample_only]
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
        median_bins = [[9.,9.5], [9.5,9.9], [9.9,10.3], [10.3,10.8]]
        
        make_av_figure = 0
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

        

    
    # format is [everything, selected, ha_only]
    sizes = [4, 8, 4]
    colors = ['#2f2f2f', 'redshift', 'redshift']
    ecolors = ['#2f2f2f', 'redshift', 'redshift']
    mecs = ['None', 'black', 'black']
    zorders = [11, 30, 20]
    shapes = ['o', 'o', 's']
    labels = ['_', 'H$\\alpha$+Pa$\\beta$', 'H$\\alpha$ only']

    hist_zorders = [11, 20, 33]
    hist_alphas = [1, 0.7, 1]
    hist_colors = ['#2f2f2f', 'redshift', 'black']
    histtypes = ['bar', 'bar', 'step']

    sfr_lims = [-2, 2.5]

    # Sample select figure to match
    # fig, ax = plt.subplots(figsize=(7,6))
    fig, axs = plt.subplot_mosaic([['histx', '.'],['scatter', 'histy']], figsize=(6, 6), width_ratios=(4, 1), height_ratios=(1, 4), layout='constrained')
    ax = axs['scatter']
    ax_histx = axs['histx']
    ax_histy = axs['histy']
    cbar_ax = fig.add_axes([0.17, 0.725, 0.3, 0.025])
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
    new_cmap = truncate_colormap(cmap, 0, 0.5)
    good_mass_idx = np.logical_and(all_masses > mass_lims[0], all_masses < mass_lims[1])
    good_sfr_idx = np.logical_and(all_log_sfr100s > sfr_lims[0], all_log_sfr100s <  sfr_lims[1])

    if show_hexes:
        hexbin_norm = mpl.colors.Normalize(vmin=1, vmax=300) 
        good_both_idx = np.logical_and(good_sfr_idx, good_mass_idx)  
        mass_cut_idx = all_masses>mass_cut
        good_both_idx = np.logical_and(good_both_idx, mass_cut_idx)
        ax.hexbin(all_masses[good_both_idx], all_log_sfr100s[good_both_idx], gridsize=15, cmap=new_cmap, norm=hexbin_norm, label='UNCOVER')
        ax_histx.hist(all_masses[good_both_idx], bins=xbins, color='grey', density=True, alpha=0.6, zorder=10)
        ax_histy.hist(all_log_sfr100s[good_both_idx], bins=ybins, color='grey',  orientation='horizontal', density=True, alpha=0.6, zorder=10)
        save_str2 = '_hexes'

    cmap = mpl.cm.viridis
    cmap = truncate_colormap(cmap, 0.2, 1.0)
    for i in range(len(dfs)): 
        if i == 0:
            continue
        df = dfs[i]

        # Compute errs
        if i == 0:
            df['zeros'] = np.zeros(len(df['sfr100_50'])) 
            low_sfr_err = df['zeros']
            high_sfr_err = df['zeros']
        else:
            low_sfr_err = np.log10(df['sfr100_50']) - np.log10(df['sfr100_16'])
            high_sfr_err = np.log10(df['sfr100_84']) - np.log10(df['sfr100_50'])
        
        for j in range(len(df)):

            mass = df['mstar_50'].iloc[j]
            if mass < mass_cut:
                continue

            size = sizes[i]
            color = colors[i]
            ecolor = ecolors[i]
            shape = shapes[i]
            mec = mecs[i]
            if color == 'redshift':
                norm = mpl.colors.Normalize(vmin=1.2, vmax=2.4) 
                color = cmap(norm(df['z_50'].iloc[j]))
                ecolor = color
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
            
            ax.errorbar(df['mstar_50'].iloc[j], np.log10(df['sfr100_50'].iloc[j]), yerr=np.array([[low_sfr_err.iloc[j], high_sfr_err.iloc[j]]]).T, marker=shape, mec=mec, ms=size, color=color, ls='None', ecolor=ecolor, zorder=zorders[i], label=labels[i])
            # ax.text(df['mstar_50'].iloc[j], df['z_50'].iloc[j], f'{id_dr3}')
        if 'redshift' in hist_colors[i]:
            value = 2.05
            if 'redshift_2' in hist_colors[i]:
                value = 1.3
            norm = mpl.colors.Normalize(vmin=1.2, vmax=2.4) 
            color = cmap(norm(value))
        else: 
            color = hist_colors[i]
        ax_histx.hist(df['mstar_50'], bins=xbins, color=color, density=True, alpha=hist_alphas[i], histtype=histtypes[i], linewidth=2, zorder=hist_zorders[i])
        ax_histy.hist(np.log10(df['sfr100_50']), bins=ybins, color=color,  orientation='horizontal', density=True, alpha=hist_alphas[i], histtype=histtypes[i], linewidth=2, zorder=hist_zorders[i])
    ax.set_xlabel(stellar_mass_label, fontsize=14)
    ax.set_ylabel('Prospector log$_{10}$(SFR)', fontsize=14)
    ax.tick_params(labelsize=14)
    
    line_sample = Line2D([0], [0], color=cmap(norm(2.05)), marker='o', markersize=8, ls='None', mec='black')
    line_colorhist = mpatches.Patch(color=cmap(norm(2.05)))
    line_snr = Line2D([0], [0], color=cmap(norm(2.05)), marker='s', markersize=4, ls='None', mec='black')
    line_blackhist = Line2D([0], [0], color='black', marker='None', ls='-', linewidth=2)  
    line_hexes = Line2D([0], [0], color='grey', marker='h', markersize=12, ls='None')
    # line_hexhist = mpatches.Patch(color='grey')
    custom_lines = [(line_sample, line_colorhist), (line_snr, line_blackhist), line_hexes]

    custom_labels = ['H$\\alpha$ + Pa$\\beta$', 'H$\\alpha$ only', 'UNCOVER']
    ax.legend(custom_lines, custom_labels, loc=4, handler_map={tuple: HandlerTuple(ndivide=None)})

    # # Plot SFMS
    # masses = np.arange(6,11,0.1)
    # predicted_log_sfrs = pop_sfms(masses, 1.96)
    # ax.plot(masses, predicted_log_sfrs, color='red', ls='--', marker='None')
    
    # if color_var == 'snr':
    #     add_snr_cbar(fig, ax, norm, cmap)
    # else:
    cbar_ticks = [1.2, 1.6, 2.0, 2.4]
    add_cbar(fig, cbar_ax, norm, cmap, 'Redshift', cbar_ticks, fontsize=10, ticklocation='bottom', labelpad=-37)
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
    
    if shapley == 2:
        sample_masses = []
        mosdef_masses = []
        sample_sfrs = []
        mosdef_sfrs = []
        sample_redshifts = []
        mosdef_redshifts = []
        for bin in median_bins:
            sample_cut = sample_df[np.logical_and(sample_df['mstar_50']>bin[0], sample_df['mstar_50']<bin[1])]
            mosdef_cut = mosdef_df[np.logical_and(mosdef_df['mstar_50']>bin[0], mosdef_df['mstar_50']<bin[1])]
            sample_masses.append(np.median(sample_cut['mstar_50']))
            mosdef_masses.append(np.median(mosdef_cut['mstar_50']))
            sample_sfrs.append(np.median(np.log10(sample_cut['sfr100_50'])))
            mosdef_sfrs.append(np.median(np.log10(mosdef_cut['sfr100_50'])))
            sample_redshifts.append(np.median(sample_cut['z_50']))
            mosdef_redshifts.append(np.median(mosdef_cut['Z_MOSFIRE']))
        median_df = pd.DataFrame(zip(sample_masses, mosdef_masses, sample_sfrs, mosdef_sfrs, sample_redshifts, mosdef_redshifts), columns=['sample_mass', 'mosdef_mass', 'sample_log_sfr', 'mosdef_log_sfr', 'sample_redshift', 'mosdef_redshift'])
        median_df.to_csv('/Users/brianlorenz/uncover/Data/generated_tables/mosdef_compare/median_props.csv', index=False)
        breakpoint()
        sys.exit()
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
    # plot_paper_sample_select_sfr_mass(show_hexes=True, mass_cut=7.5, shapley=2)

    plot_paper_sample_select_sfr_mass(show_hexes=True, mass_cut=7.5)