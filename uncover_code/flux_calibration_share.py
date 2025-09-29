import numpy as np
import pandas as pd
from sedpy import observate
from astropy.io import fits
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from astropy.table import Table

# Update to your local locations for these catalog
super_catalog_location = '/Users/brianlorenz/uncover/Catalogs/UNCOVER_v5.2.0_LW_SUPER_CATALOG.fits'
spec_catalog_location = '/Users/brianlorenz/uncover/Catalogs/UNCOVER_v0.6_zspec_zqual_catalog.fits'
spec_file_path = '/Users/brianlorenz/uncover/Catalogs/DR4_spectra'
figure_location = '/Users/brianlorenz/uncover/Figures'

# Place to input ids is at the bottom

def flux_calibrate_spectrum(id_msa, save_fluxcal=True):
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
    # popt, pcov = curve_fit(poly8, sed_df_nonan['eff_wavelength'], sed_df_nonan['flux'] / sed_df_nonan['int_spec_flux'], p0=guess8)
    def correct_flux(wavelengths, fluxes, popt):
        correction_factor = poly5(wavelengths, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])#, popt[6], popt[7], popt[8])
        return fluxes * np.abs(correction_factor)
    sed_df['int_spec_flux_calibrated'] = correct_flux(sed_df['eff_wavelength'], sed_df['int_spec_flux'], popt)
    spec_df['flux_calibrated_jy'] = correct_flux(spec_df['wave'], spec_df['flux'], popt)
    spec_df['err_flux_calibrated_jy'] = correct_flux(spec_df['wave'], spec_df['err'], popt)
  
    # Convert the flux calibrated spectra to wavelegnth
    c = 299792458 # m/s
    spec_df['flux_calibrated_erg_aa'] = spec_df['flux_calibrated_jy'] * (1e-23*1e10*c / (spec_df['wave_aa']**2))
    spec_cat = read_spec_cat()
    redshift = spec_cat[spec_cat['id_msa']==id_msa]['z_spec'].iloc[0]
    spec_df['rest_flux_calibrated_erg_aa'] = spec_df['flux_calibrated_erg_aa'] * (1+redshift)

    spec_df['err_rest_flux_calibrated_erg_aa'] = spec_df['err_flux_calibrated_jy'] * (1e-23*1e10*c / (spec_df['wave_aa']**2)) * (1+redshift)

    if save_fluxcal == True:
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
        fig.savefig(f'{figure_location}/spec_sed_compare_{id_msa}.pdf')
        plt.close('all')

    return correct_flux, popt



def read_supercat():
    supercat_loc = super_catalog_location
    supercat_df = make_pd_table_from_fits(supercat_loc)
    return supercat_df

def read_spec_cat():
    spec_cat_loc = spec_catalog_location
    spec_data_df = make_pd_table_from_fits(spec_cat_loc)
    return spec_data_df

def make_pd_table_from_fits(file_loc):
    with fits.open(file_loc) as hdu:
        data_loc = hdu[1].data
        data_df = Table(data_loc).to_pandas()
        return data_df
    
def read_raw_spec(id_msa):
    """
    read_2d options:
        1 - SPEC1D - 1D Spectrum (this is what we normally read)
        2 - SCI - 2D Spectrum, uJy
        3 - WHT - 2D weights, 1/uJy^2
        4 - PROFILE - 2D Profile
        5 - PROF1D - 1D profile, binary table   
        6 - BACKGROUND - 2D Background
        7 - SLITS - Slit information, binary table
    """
    spec_cat = read_spec_cat()
    redshift = spec_cat[spec_cat['id_msa']==id_msa]['z_spec'].iloc[0]
    raw_spec_loc = f'{spec_file_path}/uncover_DR4_prism-clear_2561_{id_msa}.spec.fits'
    spec_df = make_pd_table_from_fits(raw_spec_loc)
    spec_df['flux'] = spec_df['flux']*1e-6 # Convert uJy to Jy
    spec_df['err'] = spec_df['err']*1e-6
    spec_df['rest_flux'] = spec_df['flux']*(1+redshift)
    spec_df['wave_aa'] = spec_df['wave']*10000
    spec_df['rest_wave_aa'] = spec_df['wave_aa']/(1+redshift)
    
    c = 299792458 # m/s
    spec_df['flux_erg_aa'] = spec_df['flux'] * (1e-23*1e10*c / (spec_df['wave_aa']**2))
    
    spec_df['rest_flux_erg_aa'] = spec_df['flux_erg_aa'] * (1+redshift)
    spec_df['err_rest_flux_erg_aa'] = spec_df['err'] * (1e-23*1e10*c / (spec_df['wave_aa']**2)) * (1+redshift)
    return spec_df

def get_sed(id_msa, use_id_dr3=False):
    supercat_df = read_supercat()
    if use_id_dr3 == True:
        id_col = 'id'
    else:
        id_col = 'id_msa'
    row = supercat_df[supercat_df[id_col] == id_msa]
    if id_msa == 42041:
        row = supercat_df[supercat_df['id'] == 54635]
    filt_dir, filters = unconver_read_filters()
    filt_cols = get_filt_cols(row)   
    fluxes = []
    e_fluxes = []
    eff_waves = []
    filt_names = []
    eff_widths = []
    rect_widths = []
    for col in filt_cols:
        filt_names.append(col)
        flux = row[col].iloc[0]
        eff_wave = filt_dir[col+'_wave_eff']
        ecol = col.replace('f_', 'e_')
        e_flux = row[ecol].iloc[0]
        eff_width = filt_dir[col+'_width_eff']
        rect_width = filt_dir[col+'_width_rect']

        fluxes.append(flux*1e-8) # Jy, originally 10 nJy
        e_fluxes.append(e_flux*1e-8) # Jy
        eff_waves.append(eff_wave/10000) # microns
        eff_widths.append(eff_width) # microns
        rect_widths.append(rect_width) # microns
    sed_df = pd.DataFrame(zip(filt_names, eff_waves, fluxes, e_fluxes, eff_widths, rect_widths), columns=['filter', 'eff_wavelength', 'flux', 'err_flux', 'eff_width', 'rectangular_width'])
    return sed_df

def unconver_read_filters():
    supercat_df = read_supercat()
    filt_cols = get_filt_cols(supercat_df)
    sedpy_filts = []
    uncover_filt_dir = {}
    for filt in filt_cols:
        filtname = filt
        filt = filt.replace('f_', 'jwst_')
        try: 
            sedpy_filt = observate.load_filters([filt])
        except:
            try:
                filt = filt.replace('jwst_', 'wfc3_ir_')
                sedpy_filt = observate.load_filters([filt])
            except:
                filt = filt.replace('wfc3_ir_', 'acs_wfc_')
                sedpy_filt = observate.load_filters([filt])
        uncover_filt_dir[filtname+'_blue'] = sedpy_filt[0].blue_edge
        uncover_filt_dir[filtname+'_red'] = sedpy_filt[0].red_edge
        uncover_filt_dir[filtname+'_wave_eff'] = sedpy_filt[0].wave_effective
        uncover_filt_dir[filtname+'_width_eff'] = sedpy_filt[0].effective_width
        uncover_filt_dir[filtname+'_width_rect'] = sedpy_filt[0].rectangular_width

        scaled_trasm = sedpy_filt[0].transmission / np.max(sedpy_filt[0].transmission)
        trasm_low = scaled_trasm<0.2
        idx_lows = [i for i, x in enumerate(trasm_low) if x]
        idx_lows = np.array(idx_lows)
        max_idx = np.argmax(sedpy_filt[0].transmission)
        lower_cutoff_idx = np.max(idx_lows[idx_lows<max_idx])
        upper_cutoff_idx = np.min(idx_lows[idx_lows>max_idx])
        uncover_filt_dir[filtname+'_lower_20pct_wave'] = sedpy_filt[0].wavelength[lower_cutoff_idx]
        uncover_filt_dir[filtname+'_upper_20pct_wave'] = sedpy_filt[0].wavelength[upper_cutoff_idx]

        sedpy_filts.append(sedpy_filt[0])

    return uncover_filt_dir, sedpy_filts

def get_filt_cols(df, skip_wide_bands=False):
    filt_cols = [col for col in df.columns if 'f_' in col]
    filt_cols = [col for col in filt_cols if 'alma' not in col]
    if skip_wide_bands ==  True:
        filt_cols = [col for col in filt_cols if 'w' not in col]
    return filt_cols


if __name__ == '__main__':
    flux_calibrate_spectrum(14398)