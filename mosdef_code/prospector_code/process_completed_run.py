'''Codes that should be run on Savio immediately after the run finishes, putting the data in an easy format for plotting'''

from prospector_composite_params_group0 import get_filt_list
from cosmology_calcs import luminosity_to_flux
from convert_flux_to_maggies import prospector_maggies_to_flux, prospector_maggies_to_flux_spec
import prospect.io.read_results as reader
import sys
import os
import numpy as np
import pandas as pd
import pickle



# Directory locations on savio
savio_prospect_out_dir = '/global/scratch/brianlorenz/prospector_h5s'
prospector_plot_dir = '/global/scratch/brianlorenz/prospector_plots'
composite_sed_csvs_dir = '/global/scratch/brianlorenz/composite_sed_csvs'
composite_filter_sedpy_dir = '/global/scratch/brianlorenz/sedpy_par_files'
median_zs_file = '/global/scratch/brianlorenz/median_zs.csv'
mosdef_elines_file = '/global/scratch/brianlorenz/mosdef_elines.txt'

# For saving
prospector_csvs_dir = '/global/scratch/brianlorenz/prospector_csvs'

# Directory locations on home
# import initialize_mosdef_dirs as imd
# savio_prospect_out_dir = imd.prospector_h5_dir
# composite_sed_csvs_dir = imd.composite_sed_csvs_dir
# composite_filter_sedpy_dir = imd.composite_filter_sedpy_dir
# median_zs_file = imd.composite_seds_dir + '/median_zs.csv'
# prospector_plot_dir = imd.prospector_plot_dir
# mosdef_elines_file = imd.loc_mosdef_elines


def main_process(groupID):
    """Runs all of the functions needed to process the prospector outputs

    Parameters:
    groupID (int): The id of the group to run

    Returns:
 

    """
    all_files = os.listdir(savio_prospect_out_dir)
    target_file = [file for file in all_files if f'composite_group{groupID}_' in file]
    print(f'found {target_file}')
    res, obs, mod, sps, file = read_output(savio_prospect_out_dir + '/' + target_file[0])
    all_spec, all_phot, all_mfrac, all_line_fluxes, all_line_fluxes_erg, line_waves, weights, idx_high_weights = gen_phot(res, obs, mod, sps)
    compute_quantiles(res, obs, mod, sps, all_spec, all_phot, all_mfrac, all_line_fluxes, all_line_fluxes_erg, line_waves, weights, idx_high_weights, groupID)
    
    # Now repeat but just with the continuum
    mod.params['add_neb_emission'] = np.array([False])
    all_spec, all_phot, all_mfrac, all_line_fluxes, all_line_fluxes_erg, line_waves, weights, idx_high_weights = gen_phot(
        res, obs, mod, sps)
    compute_quantiles(res, obs, mod, sps, all_spec, all_phot, all_mfrac, all_line_fluxes, all_line_fluxes_erg, line_waves, weights, idx_high_weights, groupID, cont=True)


def read_output(file, get_sps=True):
    """Reads the results and gets the sps model for the fit
    e.g. res, obs, mod, sps = read_output(file)

    Parameters:
    file (str): The .h5 file that stores the results
    get_sps (boolean): set to True to read the sps object, False to skpip it

    Returns:
    obs (prospector): The obs object from the run
    res (prospector): The res object from the run
    mod (prospector): The model object from the run
    sps (prospector): the sps object from the run
    file (str): Name of the file that stores the results

    """
    print(f'Reading from {file}')
    res, obs, mod = reader.results_from(file)

    filt_folder = composite_filter_sedpy_dir + f'/{obs["groupID"]}_sedpy_pars'

    print(f'Setting obs["filters"] to sedpy filters from {filt_folder}')
    obs["filters"] = get_filt_list(filt_folder)

    # These will be the outputs, sps is added if get_sps==True
    outputs = [res, obs, mod]

    if get_sps == True:
        print(f'Loading sps object')
        sps = reader.get_sps(res)
        outputs = [res, obs, mod, sps, file]
    return outputs


def gen_phot(res, obs, mod, sps):
    """Generates the spec and phot objects from a given theta value

    Parameters:

    Returns:

    """

    print('Calculating first spectrum')
    theta = res['chain'][np.argmax(res['weights'])]
    spec, phot, mfrac = mod.mean_model(theta, obs, sps)
    line_waves, line_fluxes = sps.get_galaxy_elines()

    print('Starting to calculate spectra...')
    weights = res.get('weights', None)
    # Get the thousand highest weights
    idx_high_weights = np.argsort(weights)[-1000:]
    # Setup places to store the results
    all_spec = np.zeros((len(spec), len(idx_high_weights)))
    all_phot = np.zeros((len(phot), len(idx_high_weights)))
    all_mfrac = np.zeros((len(idx_high_weights)))
    all_line_fluxes = np.zeros((len(line_fluxes), len(idx_high_weights)))
    all_line_fluxes_erg = np.zeros((len(line_fluxes), len(idx_high_weights)))
    for i, weight_idx in enumerate(idx_high_weights):
        print(f'Finding mean model for {i}')
        theta_val = res['chain'][weight_idx, :]
        all_spec[:, i], all_phot[:, i], all_mfrac[i] = mod.mean_model(
            theta_val, obs, sps=sps)
        line_waves, line_fluxes = sps.get_galaxy_elines()
        all_line_fluxes[:, i] = line_fluxes
        mass = mod.params['mass']
        line_fluxes_erg_s = line_fluxes * mass * 3.846e33
        line_fluxes_erg_s_cm2 = luminosity_to_flux(
            line_fluxes_erg_s, mod.params['zred'])
        all_line_fluxes_erg[:, i] = line_fluxes_erg_s_cm2

    return all_spec, all_phot, all_mfrac, all_line_fluxes, all_line_fluxes_erg, line_waves, weights, idx_high_weights


def compute_quantiles(res, obs, mod, sps, all_spec, all_phot, all_mfrac, all_line_fluxes, all_line_fluxes_erg, line_waves, weights, idx_high_weights, groupID, cont=False):
    # Find the mean photometery, spectra
    phot16 = np.array([quantile(all_phot[i, :], 16, weights=weights[idx_high_weights])
                       for i in range(all_phot.shape[0])])
    phot50 = np.array([quantile(all_phot[i, :], 50, weights=weights[idx_high_weights])
                       for i in range(all_phot.shape[0])])
    phot84 = np.array([quantile(all_phot[i, :], 84, weights=weights[idx_high_weights])
                       for i in range(all_phot.shape[0])])
    spec16 = np.array([quantile(all_spec[i, :], 16, weights=weights[idx_high_weights])
                       for i in range(all_spec.shape[0])])
    spec50 = np.array([quantile(all_spec[i, :], 50, weights=weights[idx_high_weights])
                       for i in range(all_spec.shape[0])])
    spec84 = np.array([quantile(all_spec[i, :], 84, weights=weights[idx_high_weights])
                       for i in range(all_spec.shape[0])])
    spec16 = spec16 / ((1 + obs['z'])**2)
    spec50 = spec50 / ((1 + obs['z'])**2)
    spec84 = spec84 / ((1 + obs['z'])**2)

    lines16 = np.array([quantile(all_line_fluxes[i, :], 16, weights=weights[idx_high_weights])
                        for i in range(all_line_fluxes.shape[0])])
    lines50 = np.array([quantile(all_line_fluxes[i, :], 50, weights=weights[idx_high_weights])
                        for i in range(all_line_fluxes.shape[0])])
    lines84 = np.array([quantile(all_line_fluxes[i, :], 84, weights=weights[idx_high_weights])
                        for i in range(all_line_fluxes.shape[0])])

    lines16_erg = np.array([quantile(all_line_fluxes_erg[i, :], 16, weights=weights[idx_high_weights])
                            for i in range(all_line_fluxes_erg.shape[0])])
    lines50_erg = np.array([quantile(all_line_fluxes_erg[i, :], 50, weights=weights[idx_high_weights])
                            for i in range(all_line_fluxes_erg.shape[0])])
    lines84_erg = np.array([quantile(all_line_fluxes_erg[i, :], 84, weights=weights[idx_high_weights])
                            for i in range(all_line_fluxes_erg.shape[0])])

    # Setup wavelength ranges
    phot_wavelength = np.array(
        [f.wave_effective for f in res['obs']['filters']])
    z0_phot_wavelength = phot_wavelength / (1 + obs['z'])
    spec_wavelength = sps.wavelengths
    

    # Convert to f_lambda
    obs, phot16_flambda = prospector_maggies_to_flux(obs, phot16)
    obs, phot50_flambda = prospector_maggies_to_flux(obs, phot50)
    obs, phot84_flambda = prospector_maggies_to_flux(obs, phot84)
    spec16_flambda = prospector_maggies_to_flux_spec(
        spec_wavelength, spec16)
    spec50_flambda = prospector_maggies_to_flux_spec(
        spec_wavelength, spec50)
    spec84_flambda = prospector_maggies_to_flux_spec(
        spec_wavelength, spec84)
    
    phot_df = pd.DataFrame(zip(z0_phot_wavelength, phot16, phot50, phot84, phot16_flambda, phot50_flambda, phot84_flambda), columns=['rest_wavelength', 'phot16', 'phot50', 'phot84', 'phot16_flambda', 'phot50_flambda', 'phot84_flambda'])
    spec_df = pd.DataFrame(zip(spec_wavelength, spec16, spec50, spec84, spec16_flambda, spec50_flambda, spec84_flambda), columns=['rest_wavelength', 'spec16', 'spec50', 'spec84', 'spec16_flambda', 'spec50_flambda', 'spec84_flambda'])
    line_df = pd.DataFrame(zip(line_waves, lines16, lines50, lines84, lines16_erg, lines50_erg, lines84_erg), columns=['rest_wavelength', 'lines16', 'lines50', 'lines84', 'lines16_erg', 'lines50_erg', 'lines84_erg'])
    

    if cont==False:
        phot_df.to_csv(prospector_csvs_dir + f'/{groupID}_phot.csv', index=False)
        spec_df.to_csv(prospector_csvs_dir + f'/{groupID}_spec.csv', index=False)
        line_df.to_csv(prospector_csvs_dir + f'/{groupID}_lines.csv', index=False)

        def save_obj(obj, name):
            with open(prospector_csvs_dir + '/' + name + '.pkl', 'wb+') as f:
                pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

        save_obj(obs, f'{groupID}_obs')
        save_obj(res, f'{groupID}_res')
    
    else:
        phot_df.to_csv(prospector_csvs_dir + f'/{groupID}_cont_phot.csv', index=False)
        spec_df.to_csv(prospector_csvs_dir + f'/{groupID}_cont_spec.csv', index=False)


# function from tom to get theta values for different percentiles
def quantile(data, percents, weights=None):
    '''
    Parameters:
    data (prospector): The obs object from the run
    percents (array): in unuits of 1%
    weights (sps): The photometry from the sps generation of the model


     percents in units of 1%
    weights specifies the frequency (count) of data.
    '''
    if weights is None:
        return np.percentile(data, percents)
    ind = np.argsort(data)
    d = data[ind]
    w = weights[ind]
    p = 1. * w.cumsum() / w.sum() * 100
    y = np.interp(percents, p, d)
    return y


# Run with sys.argv when called 
groupID = sys.argv[1]
main_process(groupID)
