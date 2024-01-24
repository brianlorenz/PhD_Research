from astropy.io import ascii
import matplotlib.pyplot as plt
from fit_emission_jwst import fit_continuum, fit_emission
import numpy as np

flux_columns = ['flux_total', 'err_total', 'flux_dither_1', 'err_dither_1', 'flux_dither_2', 'err_dither_2']
z_sfgalaxy = 2.925

def main_read_spec():
    path_to_specs = '/Users/brianlorenz/jwst_sfgalaxy/data/'
    spec_df = read_1d_spectrum(path_to_specs + '128561_comb_x1d.txt')
    spec_df = convert_units(spec_df) 
    spec_df = convert_redshift(spec_df, z_sfgalaxy)
    return spec_df

# def old_main_read_spec_old():
#     path_to_specs = '/Users/brianlorenz/jwst_sfgalaxy/'
#     z = 2.925
#     spec1 = read_1d_spectrum(path_to_specs + '128561_x1d_dither1.txt')
#     spec2 = read_1d_spectrum(path_to_specs + '128561_x1d_dither2.txt')
#     specs = [spec1, spec2]
#     specs = [convert_units(spec) for spec in specs]
#     specs = [convert_redshift(spec, z) for spec in specs]
#     return specs

    # fit_region = [100, 5000]
    # fit_idxs = np.logical_and(specs[0]['rest_wavelength'] > fit_region[0], specs[0]['rest_wavelength'] < fit_region[1]) 


    # for spec in specs:
    #     plt.plot(spec[fit_idxs]['rest_wavelength'], spec[fit_idxs]['rest_flux_normalized'])

    # continuum = fit_continuum(specs[1][fit_idxs]['rest_wavelength'], specs[1][fit_idxs]['rest_flux_normalized'])
    # plt.plot(specs[1][fit_idxs]['rest_wavelength'], continuum, color='black')
    # plt.show()

def read_1d_spectrum(file):
    spec_df = ascii.read(file).to_pandas()
    spec_df = spec_df.rename(columns={"wav": "wavelength"})
    return spec_df

def convert_units(spec_df):
    #convert Jy to erg
    for name in flux_columns:
        spec_df[name] = spec_df[name] / 1.0E+23
    #convert um to angstrom
    spec_df['wavelength'] = spec_df['wavelength'] * 10000
    return spec_df

def convert_redshift(spec_df, z):
    #compute the rest wavelength using Mariska's redshift
    spec_df['rest_wavelength'] = spec_df['wavelength'] / (1+z)
    for name in flux_columns:
        spec_df[f'rest_{name}'] = spec_df[name] * (1+z)
    return spec_df


# main_read_spec()