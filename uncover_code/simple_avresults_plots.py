
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

def plot_av_properties():
    fontsize = 14

    sps_df = read_SPS_cat()
    lineratio_df = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/lineratio_av_df.csv').to_pandas()
    merged_df = lineratio_df.merge(sps_df, on='id_msa')

    fig, axarr = plt.subplots(1, 2, figsize=(12,6))
    ax_av_mass = axarr[0]
    ax_av_sfr = axarr[1]

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

        sed_av = merged_df_row['sed_av']
        err_sed_av_low = merged_df_row['err_sed_av_low']
        err_sed_av_high = merged_df_row['err_sed_av_high']
        if math.isnan(err_sed_av_low): 
            err_sed_av_low = 0
        if math.isnan(err_sed_av_high): 
            err_sed_av_high = 0
        av_err=[[err_sed_av_low], [err_sed_av_high]]

        rgba = 'black'
        
        ax_av_mass.errorbar(stellar_mass_50, sed_av, xerr=mass_err, yerr=av_err, marker='o', color=rgba, ls='None', mec='black')
        ax_av_sfr.errorbar(log_sfr100_50, sed_av, xerr=log_sfr_err, yerr=av_err, marker='o', color=rgba, ls='None', mec='black')

    ax_av_mass.set_xlabel(stellar_mass_label, fontsize=fontsize)
    ax_av_sfr.set_xlabel(sfr_label, fontsize=fontsize)
    for ax in axarr:
        ax.set_ylabel('Photometric A$_V$', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        scale_aspect(ax)
    fig.savefig('/Users/brianlorenz/uncover/Figures/av_plots/av_mass_sfr.pdf', bbox_inches='tight')

if __name__ == '__main__':
    plot_av_properties()