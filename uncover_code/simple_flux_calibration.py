from uncover_read_data import read_raw_spec, read_spec_cat, get_id_msa_list
from uncover_make_sed import get_sed
from uncover_sed_filters import unconver_read_filters
from sedpy import observate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import ascii
from scipy.optimize import curve_fit
from uncover_prospector_seds import read_prospector, get_prospect_df_loc

def flux_calibrate_spectrum(id_msa):
    print(f'Flux calibration for {id_msa}')
    spec_df = read_raw_spec(id_msa)
    sed_df = get_sed(id_msa)
    filt_dict, filters = unconver_read_filters()

    wavelength = spec_df['wave_aa'].to_numpy()
    f_lambda = spec_df['flux_erg_aa'].to_numpy()
    sed_abmag = observate.getSED(wavelength, f_lambda, filterlist=filters)
    sed_jy = 10**(-0.4*(sed_abmag-8.9))

    sed_df['int_spec_flux'] = sed_jy
    nan_indices = sed_df[sed_df.isna().any(axis=1)].index
    sed_df_nonan = sed_df.drop(nan_indices)

    
    # Polynomial correction
    def poly5(x, a5, a4, a3, a2, a1, a0):
        return a5 * x**5 + a4 * x**4 + a3 * x**3 + a2 * x**2 + a1 * x + a0
    guess = [0, 0, 0, 0, 2, 0]
    popt, pcov = curve_fit(poly5, sed_df_nonan['eff_wavelength'], sed_df_nonan['flux'] / sed_df_nonan['int_spec_flux'], p0=guess)
    def correct_flux(wavelengths, fluxes, popt):
        correction_factor = poly5(wavelengths, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])
        return fluxes * correction_factor
    sed_df['int_spec_flux_calibrated'] = correct_flux(sed_df['eff_wavelength'], sed_df['int_spec_flux'], popt)
    spec_df['flux_calibrated_jy'] = correct_flux(spec_df['wave'], spec_df['flux'], popt)
    spec_df['err_flux_calibrated_jy'] = correct_flux(spec_df['wave'], spec_df['err'], popt)
    # popt, pcov = curve_fit(poly5, sed_df_nonan['int_spec_flux'], sed_df_nonan['flux'], p0=guess)
    # sed_df['int_spec_flux_calibrated'] = poly5(sed_df['int_spec_flux'], popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])
    # spec_df['flux_calibrated_jy'] = poly5(spec_df['flux'], popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])
    
    # Convert the flux calibrated spectra to wavelegnth
    c = 299792458 # m/s
    spec_df['flux_calibrated_erg_aa'] = spec_df['flux_calibrated_jy'] * (1e-23*1e10*c / (spec_df['wave_aa']**2))
    spec_cat = read_spec_cat()
    redshift = spec_cat[spec_cat['id_msa']==id_msa]['z_spec'].iloc[0]
    spec_df['rest_flux_calibrated_erg_aa'] = spec_df['flux_calibrated_erg_aa'] * (1+redshift)

    spec_df['err_rest_flux_calibrated_erg_aa'] = spec_df['err_flux_calibrated_jy'] * (1e-23*1e10*c / (spec_df['wave_aa']**2)) * (1+redshift)

    print(f'Saving flux calibration for {id_msa}')
    sed_df.to_csv(f'/Users/brianlorenz/uncover/Data/seds/{id_msa}_sed.csv', index=False)
    spec_df_saveloc = f'/Users/brianlorenz/uncover/Data/fluxcal_specs/{id_msa}_fluxcal_spec.csv'
    spec_df.to_csv(spec_df_saveloc, index=False)


    # Plotting
    wave_micron = sed_df['eff_wavelength']
    fig, ax = plt.subplots(figsize=(6,6))
    # Raw Spec
    ax.plot(spec_df['wave'], spec_df['flux'], color='cyan', marker='None', ls='-', label='Spectrum')
    ax.plot(wave_micron, sed_jy, color='blue', marker='o', ls='None', label='Integrated Spectrum')
    
    # Flux Cal spec
    ax.plot(spec_df['wave'], spec_df['flux_calibrated_jy'], color='orange', marker='None', ls='-', label='FluxCal Spectrum')
    ax.plot(wave_micron, sed_df['int_spec_flux_calibrated'], color='red', marker='o', ls='None', label='FluxCal Int Spectrum')

    # Raw SED
    ax.plot(wave_micron, sed_df['flux'], color='black', marker='o', ls='None', label='SED')

    fontsize = 14
    ax.legend(fontsize=fontsize-4)
    ax.tick_params(labelsize=fontsize)
    ax.set_xlabel('Wavelength (um)', fontsize=fontsize)
    ax.set_ylabel('Flux (Jy)', fontsize=fontsize)
    fig.savefig(f'/Users/brianlorenz/uncover/Figures/simple_spec_sed_compare/spec_sed_compare_{id_msa}.pdf')
    plt.close('all')

if __name__ == "__main__":
    flux_calibrate_spectrum(43497)

    # id_msa_list = get_id_msa_list(full_sample=True)
    # for id_msa in id_msa_list:
    #     flux_calibrate_spectrum(id_msa)

