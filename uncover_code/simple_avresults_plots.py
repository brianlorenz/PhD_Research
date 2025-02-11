
from astropy.io import fits, ascii
from uncover_read_data import read_supercat, read_raw_spec, read_spec_cat, read_segmap, read_SPS_cat, read_aper_cat, read_fluxcal_spec, get_id_msa_list
from uncover_make_sed import read_sed
from uncover_sed_filters import unconver_read_filters
from sedpy import observate
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import pandas as pd
import numpy as np
import matplotlib as mpl
from plot_vals import scale_aspect, stellar_mass_label, sfr_label
from matplotlib.colors import Normalize, LogNorm
import math
from compute_av import compute_paalpha_pabeta_av

def plot_av_properties():
    fontsize = 14

    sps_df = read_SPS_cat()
    lineratio_df = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/lineratio_av_df.csv').to_pandas()
    merged_df = lineratio_df.merge(sps_df, on='id_msa')

    fig, axarr = plt.subplots(1, 2, figsize=(12,6))
    ax_av_mass = axarr[0]
    ax_av_sfr = axarr[1]

    fig2, axarr2 = plt.subplots(1, 2, figsize=(13,6))
    ax_prospect_sed = axarr2[0]
    ax_prospect_spec = axarr2[1]

    for i in range(len(lineratio_df)):
        merged_df_row = merged_df.iloc[i]
        
        stellar_mass_50 = merged_df_row['mstar_50']
        err_stellar_mass_low = stellar_mass_50 - merged_df_row['mstar_16']
        err_stellar_mass_high = merged_df_row['mstar_84'] - stellar_mass_50
        mass_err=[[err_stellar_mass_low], [err_stellar_mass_high]]

        log_sfr100_50 = np.log10(merged_df_row['sfr100_50'])
        err_log_sfr100_low = log_sfr100_50 - np.log10(merged_df_row['sfr100_16'])
        err_log_sfr100_high = np.log10(merged_df_row['sfr100_84']) - log_sfr100_50
        log_sfr_err=[[err_log_sfr100_low], [err_log_sfr100_high]]

        
        prosp_av_50 =  1.086 * merged_df_row['dust2_50']
        err_prosp_av_50_low = prosp_av_50 - (1.086 *merged_df_row['dust2_16'])
        err_prosp_av_50_high = (1.086 *merged_df_row['dust2_84']) - prosp_av_50
        prosp_av_err=[[err_prosp_av_50_low], [err_prosp_av_50_high]]

        sed_av = merged_df_row['sed_av']
        err_sed_av_low = merged_df_row['err_sed_av_low']
        err_sed_av_high = merged_df_row['err_sed_av_high']
        if math.isnan(err_sed_av_low): 
            err_sed_av_low = 0
        if math.isnan(err_sed_av_high): 
            err_sed_av_high = 0
        av_err=[[err_sed_av_low], [err_sed_av_high]]

        emission_fit_av = merged_df_row['emission_fit_av']
        err_emission_fit_av_low = merged_df_row['err_emission_fit_av_low']
        err_emission_fit_av_high = merged_df_row['err_emission_fit_av_high']
        emission_fit_av_err=[[err_emission_fit_av_low], [err_emission_fit_av_high]]


        rgba = 'black'
        
        ax_av_mass.errorbar(stellar_mass_50, sed_av, xerr=mass_err, yerr=av_err, marker='o', color=rgba, ls='None', mec='black')
        ax_av_sfr.errorbar(log_sfr100_50, sed_av, xerr=log_sfr_err, yerr=av_err, marker='o', color=rgba, ls='None', mec='black')

        ax_prospect_sed.errorbar(prosp_av_50, sed_av, xerr=prosp_av_err, yerr=av_err, marker='o', color=rgba, ls='None', mec='black')
        ax_prospect_spec.errorbar(prosp_av_50, emission_fit_av, xerr=prosp_av_err, yerr=emission_fit_av_err, marker='o', color=rgba, ls='None', mec='black')
    
    ax_av_mass.set_xlabel(stellar_mass_label, fontsize=fontsize)
    ax_av_sfr.set_xlabel(sfr_label, fontsize=fontsize)
    for ax in axarr:
        ax.set_ylabel('Photometric A$_V$', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        scale_aspect(ax)
    fig.savefig('/Users/brianlorenz/uncover/Figures/av_plots/av_mass_sfr.pdf', bbox_inches='tight')


    
    ax_prospect_sed.set_ylabel('Photometric A$_V$', fontsize=fontsize)
    ax_prospect_spec.set_ylabel('Spectroscopic A$_V$', fontsize=fontsize)
    
    # scale_aspect(ax_prospect)
    for ax in axarr2:
        ax.tick_params(labelsize=fontsize)
        ax.set_xlabel('Prospector A$_V$', fontsize=fontsize)
        ax.plot([-100, 10000], [-100, 10000], ls='--', color='red', marker='None')
        ax.set_xlim(-2, 5)
        ax.set_ylim(-2, 5)
    fig2.savefig('/Users/brianlorenz/uncover/Figures/av_plots/av_prospector.pdf', bbox_inches='tight')

    



def compare_pab_paa_avs(id_msa_list):
    import os

    fig, axarr = plt.subplots(1,3,figsize=(18,6))
    ax_hapaa_hapab = axarr[0]
    ax_hapaa_paapab = axarr[1]
    ax_paapab_hapab = axarr[2]
    for id_msa in id_msa_list:
        paa_fit_path = f'/Users/brianlorenz/uncover/Data/emission_fitting_paalpha/{id_msa}_emission_fits.csv'
        if os.path.exists(paa_fit_path):
            print(f'plotting {id_msa}')
            paa_fit_df = ascii.read(paa_fit_path).to_pandas()
            fit_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting/{id_msa}_emission_fits.csv').to_pandas()
            
            paa_snr = paa_fit_df['signal_noise_ratio'].iloc[1]
            if paa_snr < 1:
                print(f'paalpha snr<1, skipping {id_msa}')
                continue

            paa_av = paa_fit_df['ha_paa_av'].iloc[0]
            err_ha_paa_av_low = np.abs(paa_fit_df['err_ha_paa_av_low'].iloc[0])
            err_ha_paa_av_high = np.abs(paa_fit_df['err_ha_paa_av_high'].iloc[0])
            err_paa_av = [[err_ha_paa_av_low], [err_ha_paa_av_high]]

            
            pab_av = fit_df['ha_pab_av'].iloc[0]
            err_ha_pab_av_low = np.abs(fit_df['err_ha_pab_av_low'].iloc[0])
            err_ha_pab_av_high = np.abs(fit_df['err_ha_pab_av_high'].iloc[0])
            err_pab_av = [[err_ha_pab_av_low], [err_ha_pab_av_high]]

            paa_flux = paa_fit_df['flux'].iloc[1]
            err_paa_flux = paa_fit_df['err_flux'].iloc[1]
            pab_flux = fit_df['flux'].iloc[1]
            err_pab_flux = fit_df['err_flux'].iloc[1]
            paa_pab_ratio = paa_flux/pab_flux
            paa_err_pct = err_paa_flux/paa_flux
            pab_err_pct = err_pab_flux/pab_flux
            ratio_err_pct = np.sqrt(paa_err_pct**2 + pab_err_pct**2)
            err_paa_pab_ratio = paa_pab_ratio*ratio_err_pct
            paa_pab_av = compute_paalpha_pabeta_av(paa_pab_ratio)
            paa_pab_av_low = compute_paalpha_pabeta_av(paa_pab_ratio-err_paa_pab_ratio)
            paa_pab_av_high = compute_paalpha_pabeta_av(paa_pab_ratio+err_paa_pab_ratio)
            err_paa_pab_av = [[paa_pab_av-paa_pab_av_low], [paa_pab_av_high-paa_pab_av]]


            ax_hapaa_hapab.errorbar(pab_av, paa_av, xerr=err_pab_av, yerr=err_paa_av, marker='o', color='black', ls='None', mec='black')
            ax_hapaa_paapab.errorbar(paa_pab_av, paa_av, xerr=err_paa_pab_av, yerr=err_paa_av, marker='o', color='black', ls='None', mec='black')
            ax_paapab_hapab.errorbar(pab_av, paa_pab_av, xerr=err_pab_av, yerr=err_paa_pab_av, marker='o', color='black', ls='None', mec='black')
        
    ax_hapaa_hapab.set_xlabel('Spectroscopic Pa$\\beta$/H$\\alpha$ AV', fontsize=14)
    ax_hapaa_hapab.set_ylabel('Spectroscopic Pa$\\alpha$/H$\\alpha$ AV', fontsize=14)
    
    ax_hapaa_paapab.set_xlabel('Spectroscopic Pa$\\beta$/Pa$\\alpha$ AV', fontsize=14)
    ax_hapaa_paapab.set_ylabel('Spectroscopic Pa$\\alpha/H$\\alpha$ AV', fontsize=14)

    ax_paapab_hapab.set_xlabel('Spectroscopic Pa$\\beta$/H$\\alpha$ AV', fontsize=14)
    ax_paapab_hapab.set_ylabel('Spectroscopic Pa$\\beta$/Pa$\\alpha$ AV', fontsize=14)
    for ax in axarr:
        ax.set_xlim(-3, 6)
        ax.set_ylim(-3, 6)
        ax.tick_params(labelsize=14)
        ax.plot([-100, 10000], [-100, 10000], ls='--', color='red', marker='None')
    fig.savefig('/Users/brianlorenz/uncover/Figures/av_plots/pab_paa_ha_compare_spectra.pdf', bbox_inches='tight')

            

if __name__ == '__main__':
    # plot_av_properties()
    id_msa_list = get_id_msa_list(full_sample=True)
    compare_pab_paa_avs(id_msa_list)