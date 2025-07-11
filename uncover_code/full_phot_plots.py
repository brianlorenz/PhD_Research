from full_phot_read_data import read_merged_lineflux_cat
from full_phot_merge_lineflux import filter_bcg_flags
import matplotlib.pyplot as plt
from uncover_read_data import read_SPS_cat_all, read_bcg_surface_brightness, read_supercat, read_morphology_cat
from compute_av import compute_ha_pab_av, compute_pab_paa_av, compute_paalpha_pabeta_av
import pandas as pd
import numpy as np
import random
from plot_vals import *
import sys
import matplotlib as mpl
from matplotlib.lines import Line2D
import shutil


# Set thresholds here
redshift_sigma_thresh = 2 # sigma or higher
bcg_thresh = 0.04 # this value or lower

show_ids = False

chi2_dict = {
            'Halpha_chi2_thresh' : 0.01, 
            'PaBeta_chi2_thresh' : 0.01, 
            'PaAlpha_chi2_thresh' : 3, 
        }

def plot_mass_vs_dust(snr_dict, copy_sample=False, color_var='snr', use_ew=False, use_chi2=False):
    # Set thresholds above
    print('plotting mass vs dust')

    lineflux_df = read_merged_lineflux_cat()    

    sps_df = read_SPS_cat_all()
    bcg_df = read_bcg_surface_brightness() 
    morph_df = read_morphology_cat()

    # Merge with sps
    lineflux_df = pd.merge(lineflux_df, sps_df, left_on='id_dr3', right_on='id', how='left')
    lineflux_df = pd.merge(lineflux_df, bcg_df, left_on='id_dr3', right_on='id_dr3', how='left')
    lineflux_df = pd.merge(lineflux_df, morph_df, left_on='id_dr3', right_on='id', how='left')
    """Things to check for morph catalog - f444w_r_eff_50, f444w_n_50, f444w_ellip_50"""

    # Halpha to PaB dust measurement
    lineflux_ha_pab = lineflux_df[lineflux_df['lines_measured']==7]
    make_hist(lineflux_ha_pab['Halpha_chi2_scaled'], 'Halpha')
    make_hist(lineflux_ha_pab['PaBeta_chi2_scaled'], 'PaBeta')

    # Other possible samples
    full_possible_sample = lineflux_df[lineflux_df['lines_measured']==7]

    # PaB to PaA dust measurement
    lineflux_pab_paa = lineflux_df[lineflux_df['lines_measured']==9]
    lineflux_pab_paa['PaAlpha_chi2_scaled'] = np.ones(len(lineflux_pab_paa))

    lineflux_pab_paa = make_cuts_lineflux_df(lineflux_pab_paa, snr_dict, pab_paa_ratio=True, ew_cut=use_ew, chi2_cut=use_chi2)
    lineflux_ha_pab = make_cuts_lineflux_df(lineflux_ha_pab, snr_dict, ha_pab_ratio=True, ew_cut=use_ew, chi2_cut=use_chi2)
    # No changes to the dataframes after this point for sample selection

    

    if copy_sample:
        for i in range(len(lineflux_ha_pab)):
            id_dr3 = lineflux_ha_pab['id_dr3'].iloc[i]
            ha_snr = lineflux_ha_pab['Halpha_snr'].iloc[i]
            pab_snr = lineflux_ha_pab['PaBeta_snr'].iloc[i]
            shutil.copy('/Users/brianlorenz/uncover/Figures/PHOT_sample/sed_images/Halpha_sed_images_prospector_method' + f'/{id_dr3}_Halpha_sed.pdf', '/Users/brianlorenz/uncover/Figures/PHOT_sample/sed_images/sample/Halpha'+f'/{id_dr3}_Halpha_snr_{ha_snr:0.2f}.pdf')
            shutil.copy('/Users/brianlorenz/uncover/Figures/PHOT_sample/sed_images/Halpha_sed_images_prospector_method' + f'/{id_dr3}_Halpha_sed.pdf', '/Users/brianlorenz/uncover/Figures/PHOT_sample/sed_images/sample/Halpha_snrsort'+f'/snr_{ha_snr:0.2f}_{id_dr3}_Halpha.pdf')
            shutil.copy('/Users/brianlorenz/uncover/Figures/PHOT_sample/sed_images/PaBeta_sed_images_prospector_method' + f'/{id_dr3}_PaBeta_sed.pdf', '/Users/brianlorenz/uncover/Figures/PHOT_sample/sed_images/sample/PaBeta'+f'/{id_dr3}_PaBeta_snr_{pab_snr:0.2f}.pdf')
            shutil.copy('/Users/brianlorenz/uncover/Figures/PHOT_sample/sed_images/PaBeta_sed_images_prospector_method' + f'/{id_dr3}_PaBeta_sed.pdf', '/Users/brianlorenz/uncover/Figures/PHOT_sample/sed_images/sample/PaBeta_snrsort'+f'/snr_{pab_snr:0.2f}_{id_dr3}_PaBeta.pdf')

    # Compute dust measurements
    lineflux_ha_pab['lineratio_pab_ha'] = lineflux_ha_pab['fe_cor_PaBeta_flux'] / lineflux_ha_pab['nii_cor_Halpha_flux']
    lineflux_ha_pab['AV_pab_ha'] = compute_ha_pab_av(lineflux_ha_pab['fe_cor_PaBeta_flux'] / lineflux_ha_pab['nii_cor_Halpha_flux'])
    err_av_lows = []
    err_av_highs = []
    err_ratio_lows = []
    err_ratio_highs = []
    for i in range(len(lineflux_ha_pab)):
        boot_vals, err_av_low, err_av_high, boot_ratios, err_ratio_low, err_ratio_high = boot_errs(lineflux_ha_pab['AV_pab_ha'].iloc[i], lineflux_ha_pab['fe_cor_PaBeta_flux'].iloc[i], lineflux_ha_pab['err_fe_cor_PaBeta_flux_low'].iloc[i], lineflux_ha_pab['err_fe_cor_PaBeta_flux_high'].iloc[i], lineflux_ha_pab['nii_cor_Halpha_flux'].iloc[i], lineflux_ha_pab['err_nii_cor_Halpha_flux_low'].iloc[i], lineflux_ha_pab['err_nii_cor_Halpha_flux_high'].iloc[i], ha_pab=True)
        err_av_lows.append(err_av_low)
        err_av_highs.append(err_av_high)
        err_ratio_lows.append(err_ratio_low)
        err_ratio_highs.append(err_ratio_high)
    lineflux_ha_pab['err_AV_pab_ha_low'] = err_av_lows
    lineflux_ha_pab['err_AV_pab_ha_high'] = err_av_highs
    lineflux_ha_pab['err_lineratio_pab_ha_low'] = err_ratio_lows
    lineflux_ha_pab['err_lineratio_pab_ha_high'] = err_ratio_highs

    # Saveing for later access
    lineflux_ha_pab.to_csv('/Users/brianlorenz/uncover/Data/generated_tables/paper_data/final_sample.csv', index=False)
    full_possible_sample.to_csv('/Users/brianlorenz/uncover/Data/generated_tables/paper_data/possible_sample.csv', index=False)
    sys.exit()
    # pab_ha_av_errs = pandas_cols_to_matplotlib_errs(lineflux_ha_pab['err_AV_pab_ha_low'], lineflux_ha_pab['err_AV_pab_ha_high'])
    
    lineflux_pab_paa['AV_paa_pab'] = compute_paalpha_pabeta_av(lineflux_pab_paa['PaAlpha_flux'] / lineflux_pab_paa['fe_cor_PaBeta_flux'])
    err_av_lows = []
    err_av_highs = []
    for i in range(len(lineflux_pab_paa)):
        boot_vals, err_av_low, err_av_high, boot_ratios, err_ratio_low, err_ratio_high = boot_errs(lineflux_pab_paa['AV_paa_pab'].iloc[i], lineflux_pab_paa['PaAlpha_flux'].iloc[i], lineflux_pab_paa['err_PaAlpha_flux_low'].iloc[i], lineflux_pab_paa['err_PaAlpha_flux_high'].iloc[i], lineflux_pab_paa['fe_cor_PaBeta_flux'].iloc[i], lineflux_pab_paa['err_fe_cor_PaBeta_flux_low'].iloc[i], lineflux_pab_paa['err_fe_cor_PaBeta_flux_high'].iloc[i], paa_pab=True)
        err_av_lows.append(err_av_low)
        err_av_highs.append(err_av_high)
    lineflux_pab_paa['err_AV_paa_pab_low'] = err_av_lows
    lineflux_pab_paa['err_AV_paa_pab_high'] = err_av_highs
    pab_paa_av_errs = []
    # pab_paa_av_errs = pandas_cols_to_matplotlib_errs(lineflux_pab_paa['err_AV_paa_pab_low'], lineflux_pab_paa['err_AV_paa_pab_high'])

    # Dust figure - both mass and sfr
    for fig_type in ['axis_ratio','prospector_compare', 'mass', 'sfr']:
        fig, ax = plt.subplots(figsize=(7,6))

        # dfs = [lineflux_ha_pab, lineflux_pab_paa]
        dfs = [lineflux_ha_pab]
        colors = ['blue', 'red']
        shapes = ['o', 's']
        labels = ['PaB/Ha AV', 'PaA/PaB AV']
        av_names = ['AV_pab_ha', 'AV_paa_pab']
        line_names = [('Halpha', 'PaBeta'), ('PaBeta', 'PaAlpha')]
        
        cmap = mpl.cm.inferno
        

        for i in range(len(dfs)):
            color = colors[i]
            df = dfs[i]
            label = labels[i]
            # Compute snr for coloring
            snrs = np.min([df[f'{line_names[i][0]}_snr'], df[f'{line_names[i][1]}_snr']], axis=0)
            df['min_snr'] = snrs
            for j in range(len(df)):
                if color_var == 'snr':
                    norm = mpl.colors.LogNorm(vmin=0.5, vmax=50) 
                    rgba = cmap(norm(df['min_snr'].iloc[j]))
                elif color_var == 'ha_qual':
                    norm = mpl.colors.Normalize(vmin=0, vmax=20) 
                    rgba = cmap(norm(df['Halpha_quality_factor'].iloc[j]))
                elif color_var == 'pab_qual':
                    norm = mpl.colors.Normalize(vmin=0, vmax=5) 
                    rgba = cmap(norm(df['PaBeta_quality_factor'].iloc[j]))
                    
                err_av_low_plot = df[f'err_{av_names[i]}_low'].iloc[j]
                err_av_high_plot = df[f'err_{av_names[i]}_high'].iloc[j]
                # ax.errorbar(df['mstar_50'].iloc[j], df['z_50'].iloc[j], yerr=np.array([[low_zerr.iloc[j], high_zerr.iloc[j]]]).T, marker=shapes[i], mec='black', ms=6, color=rgba, ls='None', ecolor='gray')
                if fig_type == 'mass':
                    # ax.plot(df['mstar_50'].iloc[j], df[av_names[i]].iloc[j], marker=shapes[i], color=rgba, markersize=6, mec='black', ls='None')
                    ax.errorbar(df['mstar_50'].iloc[j], df[av_names[i]].iloc[j], yerr=np.array([[err_av_low_plot, err_av_high_plot]]).T, marker=shapes[i], color=rgba, markersize=6, mec='black', ls='None')
                    if show_ids == True:
                        id_dr3 = df['id_dr3'].iloc[j]
                        ax.text(df['mstar_50'].iloc[j], df[av_names[i]].iloc[j], f'{id_dr3}')
                if fig_type == 'sfr':
                    # ax.plot(np.log10(df['sfr100_50'].iloc[j]), df[av_names[i]].iloc[j], marker=shapes[i], color=rgba, markersize=6, mec='black', ls='None')
                    ax.errorbar(np.log10(df['sfr100_50'].iloc[j]), df[av_names[i]].iloc[j], yerr=np.array([[err_av_low_plot, err_av_high_plot]]).T, marker=shapes[i], color=rgba, markersize=6, mec='black', ls='None')
                if fig_type == 'axis_ratio':
                    ellip_err_low = df['f444w_ellip_50'].iloc[j] - df['f444w_ellip_16'].iloc[j]
                    ellip_err_high = df['f444w_ellip_84'].iloc[j] - df['f444w_ellip_50'].iloc[j]
                    ellip_errs = np.array([[ellip_err_low, ellip_err_high]]).T
                    ax.errorbar(df['f444w_ellip_50'].iloc[j], df[av_names[i]].iloc[j], xerr=ellip_errs, yerr=np.array([[err_av_low_plot, err_av_high_plot]]).T, marker=shapes[i], color=rgba, markersize=6, mec='black', ls='None')
                if fig_type == 'prospector_compare':
                    dust2_50 = df['dust2_50'].iloc[j]
                    dust1_frac_50 = df['dust1_fraction_50'].iloc[j]
                    prospector_dust = 1.086 * (dust2_50 + (dust2_50*dust1_frac_50))
                    dust_boots = []
                    for ijkl in range(1000):
                        x = random.uniform(0, 1)
                        if x < 0.5:
                            scale = dust2_50 - df['dust2_16'].iloc[j]
                            dust2_boot = dust2_50 - np.abs(np.random.normal(loc = 0, scale=scale))
                        if x > 0.5:
                            scale = df['dust2_84'].iloc[j] - dust2_50 
                            dust2_boot = dust2_50 + np.abs(np.random.normal(loc = 0, scale=scale))
                        x = random.uniform(0, 1)
                        if x < 0.5:
                            scale = dust1_frac_50 - df['dust1_fraction_16'].iloc[j]
                            dust1_frac_boot = dust1_frac_50 - np.abs(np.random.normal(loc = 0, scale=scale))
                        if x > 0.5:
                            scale = df['dust1_fraction_84'].iloc[j] - dust1_frac_50 
                            dust1_frac_boot = dust1_frac_50 + np.abs(np.random.normal(loc = 0, scale=scale))
                        dust_boot = 1.086 * (dust2_boot + (dust2_boot*dust1_frac_boot))
                        dust_boots.append(dust_boot)
                    prospector_dust_err_low =  prospector_dust - np.percentile(dust_boots, 16)
                    prospector_dust_err_high =  np.percentile(dust_boots, 84) - prospector_dust
                    if prospector_dust_err_low < 0:
                        prospector_dust_err_low = 99
                    prospector_dust_err = np.array([[prospector_dust_err_low, prospector_dust_err_high]]).T
                    ax.errorbar(prospector_dust, df[av_names[i]].iloc[j], xerr=prospector_dust_err, yerr=np.array([[err_av_low_plot, err_av_high_plot]]).T, marker=shapes[i], color=rgba, markersize=6, mec='black', ls='None')
                    
                    id_dr3 = df['id_dr3'].iloc[j]
                    if show_ids == True:
                        ax.text(prospector_dust, df[av_names[i]].iloc[j], f'{id_dr3}')
                    ax.plot([-100, 100], [-100, 100], ls='--', color='red', marker='None')
        # for i in range(len(lineflux_ha_pab)):
        #     plot_lineflux_ha_pab_err=np.array([[lineflux_ha_pab['err_AV_pab_ha_low'].iloc[i], lineflux_ha_pab['err_AV_pab_ha_high'].iloc[i]]]).T
        #     ax.errorbar(lineflux_ha_pab['mstar_50'].iloc[i], lineflux_ha_pab['AV_pab_ha'].iloc[i], yerr=plot_lineflux_ha_pab_err, marker='o', color='blue', ecolor='gray', ls='None', label='PaB/Ha AV')
        
        # for i in range(len(lineflux_pab_paa)):
        #     plot_lineflux_pab_paa_err=np.array([[lineflux_pab_paa['err_AV_paa_pab_low'].iloc[i], lineflux_pab_paa['err_AV_paa_pab_high'].iloc[i]]]).T
        #     ax.errorbar(lineflux_pab_paa['mstar_50'].iloc[i], lineflux_pab_paa['AV_paa_pab'].iloc[i], yerr=plot_lineflux_pab_paa_err, marker='o', color='red', ecolor='gray', ls='None', label='PaA/PaB AV')
        
        line_circles = Line2D([0], [0], color='orange', marker='o', markersize=6, ls='None', mec='black')
        line_squares = Line2D([0], [0], color='orange', marker='s', markersize=6, ls='None', mec='black')
        custom_lines = [line_circles, line_squares]
        custom_labels = ['Pa$\\beta$/H$\\alpha$', 'Pa$\\alpha$/Pa$\\beta$']
        ax.legend(custom_lines, custom_labels, loc=3, bbox_to_anchor=(1.05, 1.14))
        
        if color_var == 'snr':
            add_snr_cbar(fig, ax, norm, cmap)
        else:
            add_cbar(fig, ax, norm, cmap, color_var)

        if fig_type == 'mass':
            ax.set_xlabel(stellar_mass_label, fontsize=14)
            ax.set_xlim(5.5, 11.5)
        if fig_type == 'sfr':
            ax.set_xlabel('Prospector ' + sfr_label, fontsize=14)
            ax.set_xlim(-2, 3)
        if fig_type == 'prospector_compare':
            ax.set_xlabel('Prospector AV', fontsize=14)
            ax.set_xlim(0, 4)
        if fig_type == 'axis_ratio':
            ax.set_xlabel('F444W ellipticity', fontsize=14)
            ax.set_xlim(0, 1)

        ax.set_ylabel('Inferred' + balmer_av_label, fontsize=14)
        ax.set_ylim(-2, 6)
        
        save_str = ''
        save_str = add_thresh_text(ax, snr_dict, ew_cut=use_ew, chi2_cut=use_chi2)
        fig.savefig(f'/Users/brianlorenz/uncover/Figures/PHOT_analysis/dust_plots/{fig_type}_plots/{save_str}{fig_type}_vs_av.pdf')
        # plt.show()
        plt.close('all')

    # Sample select figure to match
    fig, ax = plt.subplots(figsize=(7,6))
    for i in range(len(dfs)):
        df = dfs[i]
        # ax.plot(df['mstar_50'], df['z_50'], marker='o', color=colors[i], ls='None')
        # Compute errs
        low_zerr = df['z_50'] - df['z_16']
        high_zerr = df['z_84'] - df['z_50']
        
    
        for j in range(len(df)):
            if color_var == 'snr':
                norm = mpl.colors.LogNorm(vmin=0.5, vmax=50) 
                rgba = cmap(norm(df['min_snr'].iloc[j]))
            elif color_var == 'ha_qual':
                norm = mpl.colors.Normalize(vmin=0, vmax=20) 
                rgba = cmap(norm(df['Halpha_quality_factor'].iloc[j]))
            elif color_var == 'pab_qual':
                norm = mpl.colors.Normalize(vmin=0, vmax=5) 
                rgba = cmap(norm(df['PaBeta_quality_factor'].iloc[j]))
            ax.errorbar(df['mstar_50'].iloc[j], df['z_50'].iloc[j], yerr=np.array([[low_zerr.iloc[j], high_zerr.iloc[j]]]).T, marker=shapes[i], mec='black', ms=6, color=rgba, ls='None', ecolor='gray')
    ax.set_xlabel(stellar_mass_label, fontsize=14)
    ax.set_ylabel('Redshift', fontsize=14)
    ax.tick_params(labelsize=14)
    save_str = add_thresh_text(ax, snr_dict, ew_cut=use_ew, chi2_cut=use_chi2)
    if color_var == 'snr':
        add_snr_cbar(fig, ax, norm, cmap)
    else:
        add_cbar(fig, ax, norm, cmap, color_var)
    line_circles = Line2D([0], [0], color='orange', marker='o', markersize=6, ls='None', mec='black')
    line_squares = Line2D([0], [0], color='orange', marker='s', markersize=6, ls='None', mec='black')
    custom_lines = [line_circles, line_squares]
    custom_labels = ['Pa$\\beta$/H$\\alpha$', 'Pa$\\alpha$/Pa$\\beta$']
    ax.legend(custom_lines, custom_labels, bbox_to_anchor=(1.05, 1.14))
    ax.set_ylim(1.0, 2.5)
    ax.set_xlim(5.5, 11.5)
    fig.savefig(f'/Users/brianlorenz/uncover/Figures/PHOT_analysis/sample_selects/{save_str}sample_select.pdf')
    scale_aspect(ax)
    plt.close('all')

    # F444W Histogram figure
    fig, ax = plt.subplots(figsize=(7,6))
    supercat_df = read_supercat()
    bins = np.arange(18, 30, 1)
    for i in range(len(dfs)):
        df_super = df.merge(supercat_df, left_on='id_dr3', right_on='id')
        f444w_flux_jy = df_super['f_f444w']*1e-8 # Jy, originally 10 nJy
        f444w_ab_mag = -2.5 * np.log10(f444w_flux_jy) + 8.9
        ax.hist(f444w_ab_mag, color='black', bins=bins)
    ax.set_xlabel('F444W AB Magnitude', fontsize=14)
    ax.set_ylabel('Number of Objects', fontsize=14)
    ax.tick_params(labelsize=14)
    save_str = add_thresh_text(ax, snr_dict, ew_cut=use_ew, chi2_cut=use_chi2)
    # ax.set_ylim(0, 2.5)
    ax.set_xlim(18, 30)
    fig.savefig(f'/Users/brianlorenz/uncover/Figures/PHOT_analysis/f444w_hist/f444w_hist{save_str}.pdf')
    scale_aspect(ax)
    
    plt.close('all')


def add_snr_cbar(fig, ax, norm, cmap):
    #SNR cbar
    sm =  mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar_ticks = [0.5, 2, 5, 10, 50]
    cbar_ticklabels = [str(tick) for tick in cbar_ticks]
    cbar = fig.colorbar(sm, ax=ax, ticks=cbar_ticks)
    cbar.set_label('min(SNR)', fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.set_yticklabels(cbar_ticklabels) 

def add_cbar(fig, ax, norm, cmap, cbar_name):
    #SNR cbar
    sm =  mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    # cbar_ticks = [0.5, 2, 5, 10, 50]
    # cbar_ticklabels = [str(tick) for tick in cbar_ticks]
    cbar = fig.colorbar(sm, ax=ax)#, ticks=cbar_ticks)
    cbar.set_label(cbar_name, fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    # cbar.ax.set_yticklabels(cbar_ticklabels) 

def calc_errs_worst(av, line_num_flux, line_num_err_low, line_num_err_high, line_den_flux, line_den_err_low, line_den_err_high):
    pabeta_value_low = line_num_flux - line_num_err_low
    pabeta_value_high = line_num_flux + line_num_err_high
    halpha_value_low = line_den_flux - line_den_err_low
    halpha_value_high = line_den_flux + line_den_err_high
    av_value_low = compute_ha_pab_av(pabeta_value_low / halpha_value_high)
    av_value_high = compute_ha_pab_av(pabeta_value_high / halpha_value_low)
    err_av_low = av - av_value_low
    err_av_high = av_value_high - av
    return err_av_low, err_av_high

def boot_errs(av, line_num_flux, line_num_err_low, line_num_err_high, line_den_flux, line_den_err_low, line_den_err_high, ha_pab=False, paa_pab=False, draws=1000):
    # Currently having trouble bootstrapping if any go negative, since you can't toake log of a negative number
    # Needs to handle nondetections
    boot_vals = []
    boot_ratios = []
    for i in range(draws):
        x = random.uniform(0, 1)
        if x < 0.5:
            boot_num = line_num_flux - np.abs(np.random.normal(loc = 0, scale=line_num_err_low))
        if x > 0.5:
            boot_num = np.abs(np.random.normal(loc = 0, scale=line_num_err_high)) + line_num_flux
        x = random.uniform(0, 1)
        if x < 0.5:
            boot_den = line_den_flux - np.abs(np.random.normal(loc = 0, scale=line_den_err_low))
        if x > 0.5:
            boot_den = np.abs(np.random.normal(loc = 0, scale=line_den_err_high)) + line_den_flux
        
        boot_ratio = boot_num / boot_den
        if ha_pab:
            boot_val = compute_ha_pab_av(boot_num / boot_den)
        if paa_pab:
            boot_val = compute_paalpha_pabeta_av(boot_num / boot_den)
        boot_ratios.append(boot_ratio)
        boot_vals.append(boot_val)
        # if i == 0:
        #     df = boot_val.to_frame()
        # else:
        #     df[f'{i}'] = boot_val
    boot_vals = np.array(boot_vals)
    boot_vals = np.nan_to_num(boot_vals, nan=-99)
    err_low = av - np.percentile(boot_vals, 16)
    err_high = np.percentile(boot_vals, 84) - av
    if np.percentile(boot_vals, 16) < -98:
        err_low = 1e20
    if err_high < 0:
        err_high = 1e20

    boot_ratios = np.array(boot_ratios)
    ratio = line_num_flux / line_den_flux
    boot_ratios = np.nan_to_num(boot_ratios, nan=-99)
    err_ratio_low = ratio - np.percentile(boot_ratios, 16)
    err_ratio_high = np.percentile(boot_ratios, 84) - ratio
    
    return boot_vals, err_low, err_high, boot_ratios, err_ratio_low, err_ratio_high


def make_cuts_lineflux_df(df, snr_dict, ha_pab_ratio=False, pab_paa_ratio=False, ew_cut=False, chi2_cut=False):
    '''

    ew_cut - if True, cut based on EW values in snr_dict, not snr values
    '''
    original_df = df.copy(deep=True) 
    included_lines = []
    if ha_pab_ratio:
        included_lines = ['Halpha', 'PaBeta']
    if pab_paa_ratio:
        included_lines = ['PaBeta', 'PaAlpha']

    for line_name in included_lines:  # Loop through included lines
        # SNR cut  
        if ew_cut == False and chi2_cut == False:
            snr_thresh = snr_dict[f'{line_name}_snr_thresh']
            print(f'Making cuts {line_name}, with snr_thresh of {snr_thresh} and redshift_sigma of {redshift_sigma_thresh}')
            snr_cut_only_halpha = original_df[original_df[f'Halpha_snr']<=snr_dict[f'Halpha_snr_thresh']]
            snr_cut_only_pabeta = original_df[original_df[f'PaBeta_snr']<=snr_dict[f'PaBeta_snr_thresh']]
            either_snr = pd.concat([snr_cut_only_halpha, snr_cut_only_pabeta])
            snr_cut_only = either_snr.drop_duplicates()
            df = df[df[f'{line_name}_snr']>snr_thresh]
            
        if ew_cut == True:
            snr_thresh = snr_dict[f'{line_name}_ew_thresh']
            print(f'Making cuts {line_name}, with ew_thresh of {snr_thresh} and redshift_sigma of {redshift_sigma_thresh}')
            df = df[df[f'{line_name}_eqw']>snr_thresh]
        # if chi2_cut == True:
        
        if ew_cut == False and chi2_cut == False:
            snr_thresh = chi2_dict[f'{line_name}_chi2_thresh']
            print(f'Making cuts {line_name}, with chi2_thresh of {snr_thresh} and redshift_sigma of {redshift_sigma_thresh}')
            df = df[df[f'{line_name}_chi2_scaled']<snr_thresh]
            chi2_cut_only_ha = original_df[original_df[f'Halpha_chi2_scaled']>=chi2_dict[f'Halpha_chi2_thresh']]
            chi2_cut_only_pab = original_df[original_df[f'PaBeta_chi2_scaled']>=chi2_dict[f'PaBeta_chi2_thresh']]
            either_chi2 = pd.concat([chi2_cut_only_ha, chi2_cut_only_pab])
            chi2_cut_only = either_chi2.drop_duplicates()
            
        # Redshift sig cut
        df = df[df[f'{line_name}_redshift_sigma']>redshift_sigma_thresh]
        redshift_cut_only = original_df[original_df[f'{line_name}_redshift_sigma']<=redshift_sigma_thresh]
    # bcg cut, need to be less than threshold. Doesn't vary by line
    print(f'Making bcg cuts with surface brightness above {bcg_thresh}')
    df = df[df[f'bcg_surface_brightness']<bcg_thresh]
    bcg_cut_only = original_df[original_df[f'bcg_surface_brightness']>=bcg_thresh]

    snr_cut_only.to_csv('/Users/brianlorenz/uncover/Data/generated_tables/paper_data/snr_cut_only.csv', index=False)
    chi2_cut_only.to_csv('/Users/brianlorenz/uncover/Data/generated_tables/paper_data/chi2_cut_only.csv', index=False)
    redshift_cut_only.to_csv('/Users/brianlorenz/uncover/Data/generated_tables/paper_data/redshift_cut_only.csv', index=False)
    bcg_cut_only.to_csv('/Users/brianlorenz/uncover/Data/generated_tables/paper_data/bcg_cut_only.csv', index=False)
    return df

def add_thresh_text(ax, snr_dict, ew_cut=False, chi2_cut=False):
    
    
    text_xstart = 0.01
    text_vstart = 1.11
    text_vsep = 0.04
    text_hsep = 0.35
    if ew_cut == False and chi2_cut==False:
        gt_symbol = '>'
        type_text = 'snr'
    elif ew_cut == True:
        type_text = 'ew'
        gt_symbol = '>'
    elif chi2_cut == True:
        type_text = 'chi2'
        gt_symbol = '<'
    ha_thresh = snr_dict[f'Halpha_{type_text}_thresh']
    pab_thresh = snr_dict[f'PaBeta_{type_text}_thresh']
    paa_thresh = snr_dict[f'PaAlpha_{type_text}_thresh']
    ha_chi2_thresh = chi2_dict[f'Halpha_chi2_thresh']
    pab_chi2_thresh = chi2_dict[f'PaBeta_chi2_thresh']
    ax.text(text_xstart, text_vstart, f'Ha {type_text}   {gt_symbol} {ha_thresh}', transform=ax.transAxes, color='cornflowerblue')
    ax.text(text_xstart, text_vstart - text_vsep, f'PaB {type_text} {gt_symbol} {pab_thresh}', transform=ax.transAxes, color='blue')
    ax.text(text_xstart, text_vstart - 2*text_vsep, f'PaA {type_text} {gt_symbol} {paa_thresh}', transform=ax.transAxes, color='darkblue')
    ax.text(text_xstart+text_hsep, text_vstart, f'z sigma > {redshift_sigma_thresh}', transform=ax.transAxes, color='mediumseagreen')
    ax.text(text_xstart+text_hsep, text_vstart - text_vsep, f'bcg       < {bcg_thresh}', transform=ax.transAxes, color='darkgreen')
    save_str = f'{type_text}_ha{ha_thresh}_pab{pab_thresh}_paa{paa_thresh}_bcg{bcg_thresh}_zsig{redshift_sigma_thresh}_chi2ha{ha_chi2_thresh}_chi2pab{pab_chi2_thresh}'
    return save_str

def plot_mass_vs_dust_all_snrs():
    snrs = [1, 2, 3, 5, 10]
    for snr in snrs:
        snr_dict = {
            'Halpha_snr_thresh' : snr, 
            'PaBeta_snr_thresh' : snr, 
            'PaAlpha_snr_thresh' : snr, 
        }
        plot_mass_vs_dust(snr_dict)

def make_hist(column, line_name):
    fig, ax = plt.subplots(figsize=(6,6))
    logbins = np.logspace(-4, 1, 20)

    ax.hist(column, bins=logbins, color='black')
    ax.axvline(chi2_dict[f'{line_name}_chi2_thresh'], color='red')
    ax.set_xlabel(f'{line_name}_chi2_scaled')
    ax.set_ylabel('Count')
    ax.tick_params(labelsize=14)
    ax.set_xscale('log')

    fig.savefig(f'/Users/brianlorenz/uncover/Figures/PHOT_analysis/chi2_hist/{line_name}_chi2_scaled.pdf', bbox_inches='tight')

if __name__ == "__main__":

    snr_dict = {
            'Halpha_snr_thresh' : 5, 
            'PaBeta_snr_thresh' : 5, 
            'PaAlpha_snr_thresh' : 3, 
        }
    plot_mass_vs_dust(snr_dict, copy_sample=False, color_var='snr')


    # ew_dict = {
    #         'Halpha_ew_thresh' : 500, 
    #         'PaBeta_ew_thresh' : 100, 
    #         'PaAlpha_ew_thresh' : 3, 
    #     }
    # plot_mass_vs_dust(ew_dict, copy_sample=False, color_var='snr', use_ew=True)

    
    # plot_mass_vs_dust(chi2_dict, copy_sample=False, color_var='snr', use_ew=False, use_chi2=True)


    # plot_mass_vs_dust_all_snrs()
    
    pass