'''Codes that should be run on Savio immediately after the run finishes, putting the data in an easy format for plotting'''

from prospector_params_jwstgal import get_filt_list
from cosmology_calcs import luminosity_to_flux
from convert_flux_to_maggies import prospector_maggies_to_flux, prospector_maggies_to_flux_spec
import prospect.io.read_results as reader
import sys
import os
import numpy as np
import pandas as pd
import pickle
from scipy.special import gamma, gammainc
import matplotlib.pyplot as plt



# If using non-parametric sfh, we don't calculate the line fluxes in erg/s since I'm not sure which mass value to use
non_par_sfh = False




# For saving
prospector_dir = '/Users/brianlorenz/jwst_sfgalaxy/prospector/'
plot_dir = prospector_dir + 'prospector_plots/'

plot_save = '128561_photspec_newmass'
target_file = prospector_dir + '128561_photspec_newmass_1707765947_mcmc.h5'


# Directory locations on home
# import initialize_mosdef_dirs as imd
# savio_prospect_out_dir = imd.prospector_h5_dir
# composite_sed_csvs_dir = imd.composite_sed_csvs_dir
# composite_filter_sedpy_dir = imd.composite_filter_sedpy_dir
# median_zs_file = imd.composite_seds_dir + '/median_zs.csv'
# prospector_plot_dir = imd.prospector_plot_dir
# mosdef_elines_file = imd.loc_mosdef_elines


def main_process(non_par_sfh):
    """Runs all of the functions needed to process the prospector outputs

    Parameters:
    groupID (int): The id of the group to run
    run_name (str): Name of the run, controls folders to save/read from
    non_par_sfh (boolean): Set to true if using a non-parametric SFH. Skips some calculations if true

    Returns:
 

    """
    
    res, obs, mod, sps, file = read_output(target_file)

    tfig = reader.traceplot(res)
    tfig.savefig(plot_dir + f'/{plot_save}_tfig.pdf')
    cfig = reader.subcorner(res)
    cfig.savefig(plot_dir + f'/{plot_save}_cfig.pdf')


    all_spec, all_phot, all_mfrac, all_line_fluxes, all_line_fluxes_erg, all_total_masses, all_tages, all_logzsols, all_taus, all_dust2s, all_dustindexs, all_sfrs, all_ssfrs, line_waves, weights, idx_high_weights = gen_phot(res, obs, mod, sps, non_par_sfh)
    compute_quantiles(res, obs, mod, sps, all_spec, all_phot, all_mfrac, all_line_fluxes, all_line_fluxes_erg, all_total_masses, all_tages, all_logzsols, all_taus, all_dust2s, all_dustindexs, all_sfrs, all_ssfrs, line_waves, weights, idx_high_weights)
    
    # Now repeat but just with the continuum
    mod.params['add_neb_emission'] = np.array([False])
    print('Set neb emission to false, computing continuum')
    all_spec, all_phot, all_mfrac, all_line_fluxes, all_line_fluxes_erg, all_total_masses, all_tages, all_logzsols, all_taus, all_dust2s, all_dustindexs, all_sfrs, all_ssfrs, line_waves, weights, idx_high_weights = gen_phot(
        res, obs, mod, sps, non_par_sfh)
    compute_quantiles(res, obs, mod, sps, all_spec, all_phot, all_mfrac, all_line_fluxes, all_line_fluxes_erg, all_total_masses, all_tages, all_logzsols, all_taus, all_dust2s, all_dustindexs, all_sfrs, all_ssfrs, line_waves, weights, idx_high_weights, cont=True)

def compute_SFR(mass, tage, tau):
    psi = mass * (tage/tau**2) * np.exp(-tage/tau) / (gamma(2) * gammainc(2, tage/tau)) * 1e-9
    return psi

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
    
    outputs = [res, obs, mod]

    if get_sps == True:
        print(f'Loading sps object')
        sps = reader.get_sps(res)
        outputs = [res, obs, mod, sps, file]
    return outputs



def gen_phot(res, obs, mod, sps, non_par_sfh):
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
    all_total_masses = np.zeros((len(idx_high_weights)))
    all_tages = np.zeros((len(idx_high_weights)))
    all_logzsols = np.zeros((len(idx_high_weights)))
    all_taus = np.zeros((len(idx_high_weights)))
    all_dust2s = np.zeros((len(idx_high_weights)))
    all_dustindexs = np.zeros((len(idx_high_weights)))
    all_sfrs = np.zeros(len(idx_high_weights))
    all_ssfrs = np.zeros(len(idx_high_weights))
    all_line_fluxes = np.zeros((len(line_fluxes), len(idx_high_weights)))
    all_line_fluxes_erg = np.zeros((len(line_fluxes), len(idx_high_weights)))
    # all_sfrs = np.zeros(len(idx_high_weights))
    for i, weight_idx in enumerate(idx_high_weights):
        print(f'Finding mean model for {i}')
        theta_val = res['chain'][weight_idx, :]
        theta_names = res['theta_labels']
        all_total_masses[i] = theta_val[theta_names.index('mass')]
        all_tages[i] = theta_val[theta_names.index('tage')]
        all_logzsols[i] = theta_val[theta_names.index('logzsol')]
        all_taus[i] = theta_val[theta_names.index('tau')]
        all_dust2s[i] = theta_val[theta_names.index('dust2')]
        if 'dust_index' in theta_names:
            all_dustindexs[i] = theta_val[theta_names.index('dust_index')]
        # Compute SFR - https://github.com/bd-j/prospector/issues/166
        all_sfrs[i] = compute_SFR(all_total_masses[i], all_tages[i], all_taus[i])
        all_ssfrs[i] = all_sfrs[i]/all_total_masses[i]
        all_spec[:, i], all_phot[:, i], all_mfrac[i] = mod.mean_model(
            theta_val, obs, sps=sps)
        line_waves, line_fluxes = sps.get_galaxy_elines()
        all_line_fluxes[:, i] = line_fluxes
        if non_par_sfh == False:
            # When there are multiple agebins, I don't know how to find the mass to convert with
            mass = mod.params['mass']
            line_fluxes_erg_s = line_fluxes * mass * 3.846e33
            line_fluxes_erg_s_cm2 = luminosity_to_flux(
                line_fluxes_erg_s, mod.params['zred'])
            all_line_fluxes_erg[:, i] = line_fluxes_erg_s_cm2

    return all_spec, all_phot, all_mfrac, all_line_fluxes, all_line_fluxes_erg, all_total_masses, all_tages, all_logzsols, all_taus, all_dust2s, all_dustindexs, all_sfrs, all_ssfrs, line_waves, weights, idx_high_weights

def take_one_quantile_photspec(array, weights, idx_high_weights):
    """Performs the quantile calculation for photometry of spectra"""
    perc16 = np.array([quantile(array[i, :], 16, weights=weights[idx_high_weights])
                       for i in range(array.shape[0])])
    perc50 = np.array([quantile(array[i, :], 50, weights=weights[idx_high_weights])
                       for i in range(array.shape[0])])
    perc84 = np.array([quantile(array[i, :], 84, weights=weights[idx_high_weights])
                       for i in range(array.shape[0])])
    return perc16, perc50, perc84

def take_one_quantile_thetaprop(theta_array, weights, idx_high_weights):
    """Performs the quantile calculation for something from the chain"""
    perc16, perc50, perc84 = quantile(theta_array, [16,50,84], weights=weights[idx_high_weights])
    return perc16, perc50, perc84

def compute_quantiles(res, obs, mod, sps, all_spec, all_phot, all_mfrac, all_line_fluxes, all_line_fluxes_erg, all_total_masses, all_tages, all_logzsols, all_taus, all_dust2s, all_dustindexs, all_sfrs, all_ssfrs, line_waves, weights, idx_high_weights, cont=False):
    # Find the mean photometery, spectra
    phot16, phot50, phot84 = take_one_quantile_photspec(all_phot, weights, idx_high_weights)
    spec16, spec50, spec84 = take_one_quantile_photspec(all_spec, weights, idx_high_weights)
    lines16, lines50, lines84 = take_one_quantile_photspec(all_line_fluxes, weights, idx_high_weights)
    lines16_erg, lines50_erg, lines84_erg = take_one_quantile_photspec(all_line_fluxes_erg, weights, idx_high_weights)


    if cont == False:
        surviving_masses = all_mfrac*all_total_masses
        surviving_mass16, surviving_mass50, surviving_mass84 = take_one_quantile_thetaprop(surviving_masses, weights, idx_high_weights)
        tage16, tage50, tage84 = take_one_quantile_thetaprop(all_tages, weights, idx_high_weights)
        logzsol16, logzsol50, logzsol84 = take_one_quantile_thetaprop(all_logzsols, weights, idx_high_weights)
        tau16, tau50, tau84 = take_one_quantile_thetaprop(all_taus, weights, idx_high_weights)
        dust2_16, dust2_50, dust2_84 = take_one_quantile_thetaprop(all_dust2s, weights, idx_high_weights)
        dustindex16, dustindex50, dustindex84 = take_one_quantile_thetaprop(all_dustindexs, weights, idx_high_weights)
        sfr16, sfr50, sfr84 = take_one_quantile_thetaprop(all_sfrs, weights, idx_high_weights)
        ssfr16, ssfr50, ssfr84 = take_one_quantile_thetaprop(all_ssfrs, weights, idx_high_weights)

        prop_df = pd.DataFrame(zip([surviving_mass16], [surviving_mass50], [surviving_mass84],  [tage16], [tage50], [tage84], [logzsol16], [logzsol50], [logzsol84], [tau16], [tau50], [tau84], [dust2_16], [dust2_50], [dust2_84], [dustindex16], [dustindex50], [dustindex84], [sfr16], [sfr50], [sfr84], [ssfr16], [ssfr50], [ssfr84]), columns = ['surviving_mass16', 'surviving_mass50', 'surviving_mass84',  'tage16', 'tage50', 'tage84', 'logzsol16', 'logzsol50', 'logzsol84', 'tau16', 'tau50', 'tau84', 'dust2_16', 'dust2_50', 'dust2_84', 'dustindex16', 'dustindex50', 'dustindex84', 'sfr16', 'sfr50', 'sfr84', 'ssfr16', 'ssfr50', 'ssfr84'])
        prop_df.to_csv(prospector_dir + f'/{plot_save}_props.csv', index=False)
    
    # Setup wavelength ranges
    phot_wavelength = np.array(
        [f.wave_effective for f in res['obs']['filters']])
    z0_phot_wavelength = phot_wavelength / (1 + obs['z'])
    spec_wavelength = sps.wavelengths
    

    # Convert to f_lambda
    obs, phot16_flambda = prospector_maggies_to_flux(obs, phot16)
    obs, phot50_flambda = prospector_maggies_to_flux(obs, phot50)
    obs, phot84_flambda = prospector_maggies_to_flux(obs, phot84)

    # The f_lambda fluxes are in the redshifted frame, to bring them to rest we have to mulitply by 1+z
    phot16_flambda_rest = phot16_flambda * (1 + obs['z'])
    phot50_flambda_rest = phot50_flambda * (1 + obs['z'])
    phot84_flambda_rest = phot84_flambda * (1 + obs['z'])



    spec16_flambda = prospector_maggies_to_flux_spec(
        spec_wavelength, spec16)
    spec50_flambda = prospector_maggies_to_flux_spec(
        spec_wavelength, spec50)
    spec84_flambda = prospector_maggies_to_flux_spec(
        spec_wavelength, spec84)
    spec16_flambda_rest = spec16_flambda / (1 + obs['z'])
    spec50_flambda_rest = spec50_flambda / (1 + obs['z'])
    spec84_flambda_rest = spec84_flambda / (1 + obs['z'])
    # Did not redshift, convert, and then redshift back. An equivalent way is to convert and then divide by (1+z)^2 - why is it 1+z and not 1+z^2
    

    # Don't need to copy what is done to the spectra to the lines, since they are in different units

    
    phot_df = pd.DataFrame(zip(z0_phot_wavelength, phot16_flambda_rest, phot50_flambda_rest, phot84_flambda_rest), columns=['rest_wavelength', 'phot16_flambda', 'phot50_flambda', 'phot84_flambda'])
    spec_df = pd.DataFrame(zip(spec_wavelength, spec16_flambda_rest, spec50_flambda_rest, spec84_flambda_rest), columns=['rest_wavelength', 'spec16_flambda', 'spec50_flambda', 'spec84_flambda'])
    line_df = pd.DataFrame(zip(line_waves, lines16_erg, lines50_erg, lines84_erg), columns=['rest_wavelength', 'lines16_erg', 'lines50_erg', 'lines84_erg'])
    


    


    if cont==False:
        phot_df.to_csv(prospector_dir + f'{plot_save}_phot.csv', index=False)
        spec_df.to_csv(prospector_dir + f'{plot_save}_spec.csv', index=False)
        line_df.to_csv(prospector_dir + f'{plot_save}_lines.csv', index=False)

        def save_obj(obj, name):
            with open(plot_save + name + '.pkl', 'wb+') as f:
                pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

        try:
            save_obj(obs, f'{plot_save}_obs')
        except:
            print('Could not pickle obs')
        try:
            save_obj(res, f'{plot_save}_res')
        except:
            print('Could not pickle res')        
        try:
            save_obj(mod, f'{plot_save}_mod')
        except:
            print('Could not pickle mod')
    
    else:
        phot_df.to_csv(prospector_dir + f'{plot_save}_cont_phot.csv', index=False)
        spec_df.to_csv(prospector_dir + f'{plot_save}_cont_spec.csv', index=False)



    

    add_spec = False
    for i in range(2):
        fig, ax = plt.subplots(figsize=(8, 7))

        # Set the wavelength limits
        start_spec = phot_df['rest_wavelength'].iloc[0]
        end_spec = phot_df['rest_wavelength'].iloc[-1]
        spec_idxs = np.logical_and(
            spec_df['rest_wavelength'] > start_spec, spec_df['rest_wavelength'] < end_spec)

        # Plot photometry
        rest_frame_original_phot = obs['f_lambda']*(1+obs['z'])
        rest_frame_original_phot_errs = obs['err_f_lambda']*(1+obs['z'])
        ax.errorbar(phot_df['rest_wavelength'], phot_df['rest_wavelength'] * rest_frame_original_phot, color='black', yerr=phot_df['rest_wavelength'] * rest_frame_original_phot_errs, ls='-', marker='o', label='Observations', zorder=1)

        y_model = np.array(phot_df['rest_wavelength']
                            * phot_df['phot50_flambda'])
        y_model_16 = phot_df['rest_wavelength'] * phot_df['phot16_flambda']
        y_model_84 = phot_df['rest_wavelength'] * phot_df['phot84_flambda']
        model_errs = np.vstack((y_model - y_model_16, y_model_84 - y_model))
        ax.errorbar(np.array(phot_df['rest_wavelength']), y_model,
                    ls='-', marker='o', yerr=model_errs, color='blue', label='Model')
        
        ax.axvspan(phot_df['rest_wavelength'].iloc[0]-100, 1100, alpha=0.6, color='grey')

        ## SPECTRUM IS HERE
        # Plot spectrum
        spec_name = ''
        if add_spec==True:
            ax.plot(spec_df['rest_wavelength'][spec_idxs], spec_df['rest_wavelength'][spec_idxs] * spec_df['spec50_flambda'][spec_idxs], '-',
                    color='orange', label='Model spectrum', zorder=3)
            spec_name='_spec'
        ax.set_xscale('log')
        ax.set_ylim(0.8 * np.percentile(phot_df['rest_wavelength'] * rest_frame_original_phot, 1),
                        1.1 * np.percentile(phot_df['rest_wavelength'] * rest_frame_original_phot, 99))
        ax.set_xlim(phot_df['rest_wavelength'].iloc[0] -
                        30, phot_df['rest_wavelength'].iloc[-1] + 3000)
   
        ax.legend()
        ax.set_ylabel("$\lambda$ F$_\lambda$", fontsize = 14)
        ax.set_xlabel("Wavelength ($\AA$)", fontsize=14)
        ax.tick_params(labelsize=14)
        fig.savefig(prospector_dir + f'{plot_save}_{add_spec}.pdf', bbox_inches='tight')
        add_spec = True
        plt.close('all')


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





main_process(non_par_sfh)
