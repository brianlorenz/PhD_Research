from astropy.io import ascii
import pandas as pd
from read_jwst_spectrum import z_sfgalaxy

galaxy_id = 128561
catalog_loc = '/Users/brianlorenz/jwst_sfgalaxy/data/catalog/UVISTA_DR3_master_v1.1_SUSPENSE.cat'
filters_loc = '/Users/brianlorenz/jwst_sfgalaxy/data/catalog/suspense_filterlist.csv'
sed_loc = f'/Users/brianlorenz/jwst_sfgalaxy/data/{galaxy_id}_sed.csv'
spec_loc = f'/Users/brianlorenz/jwst_sfgalaxy/data/{galaxy_id}_spec.csv'

def main():
    cat_df = read_catalog()
    filters_df = read_filters()
    gal_row = cat_df[cat_df['id'] == galaxy_id]
    sed = match_fluxes(gal_row, filters_df)
    sed = sed.sort_values(by='peak_wavelength', ascending=True)
    sed = deredshift(sed)
    sed.to_csv(sed_loc, index=False)


def deredshift(sed):
    sed['rest_wavelength'] = sed['peak_wavelength']/(1+z_sfgalaxy)
    sed['rest_f_lambda'] = sed['f_lambda']*(1+z_sfgalaxy)
    sed['err_rest_f_lambda'] = sed['err_f_lambda']*(1+z_sfgalaxy)
    return sed

def read_catalog():
    cat_df = ascii.read(catalog_loc).to_pandas()
    return cat_df

def read_filters():
    filters_df = ascii.read(filters_loc).to_pandas()
    return filters_df

def match_fluxes(gal_row, filters_df):
    fluxes = [float(gal_row[filtname]) for filtname in filters_df['filter_name']]
    errorfluxes = [float(gal_row['e'+filtname]) for filtname in filters_df['filter_name']]
    flux_tuple = [(fluxes[i], errorfluxes[i]) for i in range(len(fluxes))]
    # Magnitude zeropoint conversion from:
    # http://monoceros.astro.yale.edu/RELEASE_V4.0/Photometry/AEGIS/aegis_3dhst.v4.1.cats/aegis_readme.v4.1.txt
    convert_factor = 3.7325 * 10**(-30)
    # Convert from f_nu to f_lambda
    convert_lambda = 3 * 10**18
    f_lambda_tuple = [(convert_factor * convert_lambda * flux_tuple[i][0] / (filters_df.iloc[i]['peak_wavelength'])**2, convert_factor *
                       convert_lambda * flux_tuple[i][1] / (filters_df.iloc[i]['peak_wavelength'])**2) for i in range(len(flux_tuple))]
    sed = pd.DataFrame(f_lambda_tuple, columns=['f_lambda', 'err_f_lambda'])
    # Concatenate with wavelength
    sed = sed.merge(filters_df,left_index=True, right_index=True)
    flux_df = pd.DataFrame(zip(fluxes, errorfluxes), columns=['flux_ab25', 'flux_error_ab25'])
    sed = sed.merge(flux_df, left_index=True, right_index=True)
    # Continue to set the -99 to -99
    sed.loc[sed['flux_ab25'] == -99, 'f_lambda'] = -99
    sed.loc[sed['flux_ab25'] == -99, 'err_f_lambda'] = -99
    return sed

# main()