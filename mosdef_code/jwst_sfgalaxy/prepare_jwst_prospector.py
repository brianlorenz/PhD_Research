from read_catalog import spec_loc, sed_loc
from read_jwst_spectrum import z_sfgalaxy
from astropy.io import ascii

def convert_sed_flux_to_maggies():
    """Adds a new column in a file that converts the flux to maggies, the unit needed by Prospector

    Parameters:
    target_file (str) - location of file containing composite SED points

    """
    sed_df = ascii.read(sed_loc).to_pandas()
    
    sed_df['redshifted_peak_wavelength'] = sed_df['peak_wavelength']*(1+z_sfgalaxy)
    sed_df['redshifted_flux'] = sed_df['f_lambda'] / (1+z_sfgalaxy)
    sed_df['err_redshifted_flux'] = sed_df['err_f_lambda'] / (1+z_sfgalaxy)
    f_nu = sed_df['redshifted_flux'] * (sed_df['redshifted_peak_wavelength']**2) * 3.34 * 10**(-19)
    f_jy = f_nu * (10**23)
    maggies = f_jy / 3631
    err = sed_df['err_redshifted_flux'] * \
        ((sed_df['redshifted_peak_wavelength']**2) * 3.34 * 10**(-19)) * (10**23) / 3631
    sed_df['f_maggies_red'] = maggies
    sed_df['err_f_maggies_red'] = err
    

    sed_df.to_csv(f'/Users/brianlorenz/jwst_sfgalaxy/data/128561_sed_maggies.csv', index=False)
    return

convert_sed_flux_to_maggies()