from uncover_read_data import read_raw_spec
from uncover_make_sed import get_sed
from uncover_sed_filters import unconver_read_filters
from sedpy import observate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import ascii

def compare_sed_flux(id_msa, make_plot=True):
    spec_df = read_raw_spec(id_msa)
    sed_df = get_sed(id_msa)
    filt_dict, filters = unconver_read_filters()
    
    wavelength = spec_df['wave_aa'].to_numpy()
    f_lambda = spec_df['flux_erg_aa'].to_numpy()
    sed_abmag = observate.getSED(wavelength, f_lambda, filterlist=filters)
    sed_jy = 10**(-0.4*(sed_abmag-8.9))
    # breakpoint() # ha 12, 13, 15
    # breakpoint() # pab 19, 21, 22
    ha_cont_pct = 0.5369986399626991
    pab_cont_pct = 0.6381169699210135
    ha_cont = np.percentile([sed_jy[12], sed_jy[15]], ha_cont_pct)
    pab_cont = np.percentile([sed_jy[19], sed_jy[22]], pab_cont_pct)
    ha_line = sed_jy[13] - ha_cont
    pab_line = sed_jy[21] - pab_cont
    breakpoint()
    
    wave_micron = sed_df['eff_wavelength']

    flux_ratio = sed_df['flux']/sed_jy
    full_ratio = np.nanmedian(flux_ratio)
    blue_ratio = np.nanmedian(flux_ratio[0:8])
    red_ratio = np.nanmedian(flux_ratio[-8:])


    if make_plot == True:
        fig, ax = plt.subplots(figsize=(6,6))
        ax.plot(spec_df['wave'], spec_df['flux'], color='blue', marker='None', ls='--', label='Spectrum')
        ax.plot(wave_micron, sed_jy, color='orange', marker='o', ls='None', label='Integrated Spectrum')
        ax.plot(wave_micron, sed_df['flux'], color='black', marker='o', ls='None', label='SED')
        if 'scaled_flux' in spec_df.columns:
             ax.plot(spec_df['wave'], spec_df['scaled_flux'], color='darkblue', marker='None', ls='--', label='Scaled Spectrum')
        fontsize = 14
        ax.legend(fontsize=fontsize-4)
        ax.tick_params(labelsize=fontsize)
        ax.set_xlabel('Wavelength (um)', fontsize=fontsize)
        ax.set_ylabel('Flux (Jy)', fontsize=fontsize)
        fig.savefig(f'/Users/brianlorenz/uncover/Figures/spec_sed_compare/spec_sed_compare_{id_msa}.pdf')
        plt.close('all')

    return full_ratio, blue_ratio, red_ratio

def compare_all_sed_flux(id_msa_list):
    full_ratios = []
    blue_ratios = []
    red_ratios = []
    for id_msa in id_msa_list:
        print(f'Finding scaling factor for {id_msa}')
        full_ratio, blue_ratio, red_ratio = compare_sed_flux(id_msa, make_plot=True)
        full_ratios.append(full_ratio)
        blue_ratios.append(blue_ratio)
        red_ratios.appeend(red_ratio)

    ratio_df = pd.DataFrame(zip(id_msa_list, full_ratios, blue_ratios, red_ratios), columns = ['id_msa', 'full_ratio', 'blue_ratio', 'red_ratio'])
    ratio_df.to_csv('/Users/brianlorenz/uncover/Figures/spec_sed_compare/compare_ratio.csv', index=False)


def make_ratio_hist():
    ratio_df = ascii.read('/Users/brianlorenz/uncover/Figures/spec_sed_compare/compare_ratio.csv').to_pandas()
    bins = np.arange(0, 6, 0.2)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.hist(ratio_df['full_ratio'], bins = bins, color='black')
    fig.savefig('/Users/brianlorenz/uncover/Figures/spec_sed_compare/hist_fullratio.pdf')
    plt.close('all')
    fig, ax = plt.subplots(figsize=(6,6))
    ax.hist(ratio_df['blue_ratio'], bins = bins, color='black')
    fig.savefig('/Users/brianlorenz/uncover/Figures/spec_sed_compare/hist_blueratio.pdf')
    plt.close('all')
    fig, ax = plt.subplots(figsize=(6,6))
    ax.hist(ratio_df['red_ratio'], bins = bins, color='black')
    fig.savefig('/Users/brianlorenz/uncover/Figures/spec_sed_compare/hist_redratio.pdf')
    plt.close('all')
# make_ratio_hist()
# compare_sed_flux(47875)


