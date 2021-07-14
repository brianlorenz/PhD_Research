import prospect.io.read_results as reader
import sys
import os
import numpy as np
import pandas as pd
from astropy.io import ascii
from read_data import mosdef_df
from uvj_clusters import plot_uvj_cluster
from emission_measurements import read_emission_df
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from bpt_clusters import plot_bpt
import cluster_data_funcs as cdf
from emission_measurements import read_emission_df
from prospector_composite_params import get_filt_list
from convert_flux_to_maggies import prospector_maggies_to_flux, prospector_maggies_to_flux_spec
from plot_mass_sfr import plot_mass_sfr_cluster, read_sfr_df, get_all_sfrs_masses
from cosmology_calcs import luminosity_to_flux


# Directory locations on savio
savio_prospect_out_dir = '/global/scratch/brianlorenz/prospector_h5s'
prospector_plot_dir = '/global/scratch/brianlorenz/prospector_plots'
composite_sed_csvs_dir = '/global/scratch/brianlorenz/composite_sed_csvs'
composite_filter_sedpy_dir = '/global/scratch/brianlorenz/sedpy_par_files'
median_zs_file = '/global/scratch/brianlorenz/median_zs.csv'
mosdef_elines_file = '/global/scratch/brianlorenz/mosdef_elines.txt'


# Directory locations on home
# import initialize_mosdef_dirs as imd
# savio_prospect_out_dir = imd.prospector_h5_dir
# composite_sed_csvs_dir = imd.composite_sed_csvs_dir
# composite_filter_sedpy_dir = imd.composite_filter_sedpy_dir
# median_zs_file = imd.composite_seds_dir + '/median_zs.csv'
# prospector_plot_dir = imd.prospector_plot_dir
# mosdef_elines_file = imd.loc_mosdef_elines

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


def make_plots(res, obs, mod, sps, all_spec, all_phot, all_line_fluxes, all_line_fluxes_erg, line_waves, weights, idx_high_weights, file, p_plots=False, mask=False, savename='False'):
    """Plots the observations vs the sps model

    Parameters:
    obs (prospector): The obs object from the run
    res (prospector): The res object from the run
    phot (sps): The photometry from the sps generation of the model
    theta (prospector): The values of the free parameters
    file (str): Name of the .h5 file used
    p_plots(boolean): Set to True to make Prospector plots tfig and cfig, False to skip
    mask(boolean): Set to True if there was a mask
    savename (str): Set to the name you want to save the file under

    """
    save_dir = prospector_plot_dir
    if savename == 'False':
        savename = file[:-19]

    groupID = int(obs["groupID"])

    # Read in nebular emission lines from mosdef
    lines_df = ascii.read(mosdef_elines_file).to_pandas()

    # Make tfig and cfig
    if p_plots == True:
        tfig = reader.traceplot(res)
        tfig.savefig(save_dir + savename + '_tfig.pdf')
        cfig = reader.subcorner(res)
        cfig.savefig(save_dir + savename + '_cfig.pdf')

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
    # lines16_erg = (lines16_erg) * ((1 + obs['z'])**2)
    # lines50_erg = (lines50_erg) * ((1 + obs['z'])**2)
    # lines84_erg = (lines84_erg) * ((1 + obs['z'])**2)

    parnames = np.array(res.get('theta_labels', mod.theta_labels()))

    # Figure setup
    fig = plt.figure(figsize=(14, 8))
    ax_main = fig.add_axes([0.04, 0.40, 0.44, 0.57])
    ax_zoomHb = fig.add_axes([0.63, 0.73, 0.20, 0.24])
    ax_zoomHa = fig.add_axes([0.88, 0.73, 0.10, 0.24])
    spectra_axes = [ax_main, ax_zoomHa, ax_zoomHb]

    # Plot the UVJ diagram
    ax_UVJ = fig.add_axes([0.76, 0.08, 0.20, 0.25])
    plot_uvj_cluster(groupID, ax_UVJ)
    ax_UVJ.set_xlabel('V-J')
    ax_UVJ.set_ylabel('U-V')

    # Setup the BPT Diagram
    ax_BPT = fig.add_axes([0.76, 0.38, 0.20, 0.25])

    # Plot the SFR/Mass Diagram
    ax_SFR = fig.add_axes([0.52, 0.08, 0.20, 0.25])
    sfr_df = read_sfr_df()
    all_sfrs_res = get_all_sfrs_masses(sfr_df)
    plot_mass_sfr_cluster(groupID, all_sfrs_res, axis_obj=ax_SFR)
    ax_SFR.set_xlabel('log(Stellar Mass) (M_sun)')
    ax_SFR.set_ylabel('log(SFR) (M_sun/yr)')

    # Axis for residuals between model and observations
    ax_residual = fig.add_axes([0.04, 0.08, 0.44, 0.25])

    # Setup wavelength ranges
    phot_wavelength = np.array(
        [f.wave_effective for f in res['obs']['filters']])
    z0_phot_wavelength = phot_wavelength / (1 + obs['z'])
    # Set the wavelength limits
    start_spec = z0_phot_wavelength[0]
    end_spec = z0_phot_wavelength[-1]
    spec_wavelength = sps.wavelengths
    z0_spec_wavelength = spec_wavelength  # / (1 + obs['z'])
    spec_idxs = np.logical_and(
        z0_spec_wavelength > start_spec, z0_spec_wavelength < end_spec)
    # start_spec = phot_wavelength[0]
    # end_spec = phot_wavelength[-1]
    # spec_wavelength = np.arange(start_spec, end_spec, 4)
    # z0_spec_wavelength = np.arange(start_spec, end_spec, 4) / (1 + obs['z'])

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

    # Plot the BPT diagram:
    bpt_lines = [6585, 6563, 5008, 4861]
    bpt_names = ['NII', 'Ha', 'OIII', 'Hb']
    bpt_fluxes = []
    bpt_errs_pct = []
    # Compute the BPT value for our composite
    for bpt_line in bpt_lines:
        # Find the nearest line in cloudy:
        idx_nearest = np.argmin(np.abs(line_waves - bpt_line))
        line_wave = line_waves[idx_nearest]
        line_flux = lines50[idx_nearest]
        line_errs_pct = (((line_flux - lines16[idx_nearest]) / line_flux),
                         ((lines84[idx_nearest] - line_flux) / line_flux))
        bpt_fluxes.append(line_flux)
        bpt_errs_pct.append(line_errs_pct)
    # Calculate the bpt points
    bpt_x_rat = np.log10(bpt_fluxes[0] / bpt_fluxes[1])
    err_bpt_x_rat = (0.434 * np.sqrt(bpt_errs_pct[0][0]**2 + bpt_errs_pct[1][
                     0]**2), 0.434 * np.sqrt(bpt_errs_pct[0][1]**2 + bpt_errs_pct[1][1]**2))
    bpt_y_rat = np.log10(bpt_fluxes[2] / bpt_fluxes[3])
    err_bpt_y_rat = (0.434 * np.sqrt(bpt_errs_pct[2][0]**2 + bpt_errs_pct[3][
                     0]**2), 0.434 * np.sqrt(bpt_errs_pct[2][1]**2 + bpt_errs_pct[3][1]**2))
    composite_bpt_point = [bpt_x_rat, bpt_y_rat]
    composite_bpt_errs = [err_bpt_x_rat, err_bpt_y_rat]
    # Read in the emission for cluster galaxies
    emission_df = read_emission_df()
    # Find the names and ids of cluster galaxies
    cluster_names, fields_ids = cdf.get_cluster_fields_ids(groupID)
    fields_ids = [(obj[0], int(obj[1])) for obj in fields_ids]
    plot_bpt(emission_df, fields_ids, axis_obj=ax_BPT,
             composite_bpt_point=composite_bpt_point, composite_bpt_errs=composite_bpt_errs)
    ax_BPT.set_xlabel('log(N[II] 6583 / H$\\alpha$)')
    ax_BPT.set_ylabel('log(O[III] 5007 / H$\\beta$)')

    # Find the ranges for Ha and Hb
    Ha_range = [6500, 6625]
    Hb_range = [4800, 5050]
    Ha_idxs = np.logical_and(z0_spec_wavelength > Ha_range[
                             0], z0_spec_wavelength < Ha_range[1])
    Hb_idxs = np.logical_and(z0_spec_wavelength > Hb_range[
                             0], z0_spec_wavelength < Hb_range[1])

    ax_ranges = [[start_spec, end_spec], Ha_range, Hb_range]

    for j in range(len(spectra_axes)):
        ax = spectra_axes[j]
        ax_range = ax_ranges[j]
        # Plot photometry
        ax.errorbar(z0_phot_wavelength, z0_phot_wavelength * obs['f_lambda'], color='black', yerr=z0_phot_wavelength * obs[
                    'err_f_lambda'], ls='-', marker='o', label='Observations', zorder=1)

        y_model = np.array(z0_phot_wavelength * phot50_flambda)
        y_model_16 = z0_phot_wavelength * phot16_flambda
        y_model_84 = z0_phot_wavelength * phot84_flambda
        model_errs = np.vstack((y_model - y_model_16, y_model_84 - y_model))
        ax.errorbar(np.array(z0_phot_wavelength), y_model,
                    ls='-', marker='o', yerr=model_errs, color='blue', label='Model')

        # Plot spectrum
        ax.plot(z0_spec_wavelength[spec_idxs], z0_spec_wavelength[spec_idxs] * spec50_flambda[spec_idxs], '-',
                color='orange', label='Model spectrum', zorder=3)

        # Plot Emission line and labels - plots all from CLOUDY
        # for i in range(len(line_waves)):
        #     if np.logical_and(line_waves[i] > z0_spec_wavelength[0], line_waves[i] < z0_spec_wavelength[-1]):
        #         ax.axvline(line_waves[i],
        #                    ls='--', color='mediumseagreen')

        # Plot Emission line and labels - plots only those form mosdef
        text_height = 0.96
        for i in range(len(lines_df)):
            # Get rest wavelength
            line_rest_wave = int(lines_df.iloc[i]['Wavelength'])

            # Check if the line is in the range - if it's out, loop
            if np.logical_or(line_rest_wave < ax_range[0], line_rest_wave > ax_range[1]):
                continue

            # Find the nearest line in cloudy:
            idx_nearest = np.argmin(np.abs(line_waves - line_rest_wave))
            line_wave = line_waves[idx_nearest]
            line_flux = lines50[idx_nearest]
            # Plot a green line where it is
            ax.axvline(line_wave, ls='--', color='mediumseagreen')
            # Get the line's height
            line_range = np.logical_and(z0_spec_wavelength > (
                line_wave - 20), z0_spec_wavelength < (line_wave + 20))
            # Add a label
            trans = transforms.blended_transform_factory(
                ax.transData, ax.transAxes)
            ax.text(line_wave, text_height, lines_df.iloc[
                    i]['Name'], transform=trans)
            text_height = text_height - 0.025

        ax.set_ylabel("$\lambda$ F$_\lambda$")
        ax.set_xlabel("Wavelength ($\AA$)")

    # Plot residuals
    ax_residual.plot(z0_phot_wavelength, (z0_phot_wavelength * obs['f_lambda']) - (z0_phot_wavelength * phot50_flambda),
                     '-o', color='black')
    ax_residual.plot(z0_phot_wavelength, np.zeros(
        len(z0_phot_wavelength)), '--', color='grey')
    ax_residual.set_ylabel("$\lambda$ F$_\lambda$")
    ax_residual.set_xlabel("Wavelength ($\AA$)")
    ax_residual.set_xscale('log')

    # for i in range(len(lines_df)):
    #     if np.logical_and(lines_df.iloc[i]['Wavelength'] > spec_wavelength[0], lines_df.iloc[i]['Wavelength'] < spec_wavelength[-1]):
    #         ax.axvline(lines_df.iloc[i]['Wavelength'],
    #                    ls='--', color='mediumseagreen')
    ax_main.set_xscale('log')
    # ax_main.set_ylim(0.8 * np.percentile(z0_spec_wavelength[spec_idxs] * spectrum, 1),
    # 1.1 * np.percentile(z0_spec_wavelength[spec_idxs] * spectrum, 99))
    ax_main.set_ylim(0.8 * np.percentile(z0_spec_wavelength[spec_idxs] * spec50_flambda[spec_idxs], 1),
                     1.1 * np.percentile(z0_spec_wavelength[spec_idxs] * spec50_flambda[spec_idxs], 99))
    ax_main.set_xlim(z0_spec_wavelength[spec_idxs][
                     0] - 30, z0_spec_wavelength[spec_idxs][-1] + 3000)

    ax_zoomHa.set_xlim(Ha_range[0], Ha_range[1])
    ax_zoomHb.set_xlim(Hb_range[0], Hb_range[1])
    # ax_zoomHa.set_ylim(
    #     0.8 * np.percentile(z0_spec_wavelength[Ha_idxs] * spectrum[Ha_idxs], 1), 1.1 * np.percentile(z0_spec_wavelength[Ha_idxs] * spectrum[Ha_idxs], 99))
    # ax_zoomHb.set_ylim(
    # 0.8 * np.percentile(z0_spec_wavelength[Hb_idxs] * spectrum[Hb_idxs], 1),
    # 1.1 * np.percentile(z0_spec_wavelength[Hb_idxs] * spectrum[Hb_idxs],
    # 99))
    ax_zoomHa.set_ylim(
        0.8 * np.percentile(z0_spec_wavelength[Ha_idxs] * spec50_flambda[Ha_idxs], 1), 1.1 * np.percentile(z0_spec_wavelength[Ha_idxs] * spec50_flambda[Ha_idxs], 99))
    ax_zoomHb.set_ylim(
        0.8 * np.percentile(z0_spec_wavelength[Hb_idxs] * spec50_flambda[Hb_idxs], 1), 1.1 * np.percentile(z0_spec_wavelength[Hb_idxs] * spec50_flambda[Hb_idxs], 99))

    # Mask for halpha
    if mask == True:
        ax_main.axvspan(4500, 5300, facecolor='r', alpha=0.5, label="Mask")
        ax_main.axvspan(6100, 6900, facecolor='r', alpha=0.5)

    # Save the fluxes of the model lines
    model_lines_df = pd.DataFrame(zip(line_waves, lines50_erg), columns=[
                                  'rest_wavelength', 'flux'])

    # Output the merged spectra and lines
    output_dir = prospector_plot_dir
    spec_df = pd.DataFrame(zip(z0_spec_wavelength, (spec16_flambda) * ((1 + obs['z'])**2), (spec50_flambda) * ((1 + obs['z'])**2), (spec84_flambda) * ((1 + obs['z'])**2)), columns=[
                           'rest_wavelength', 'spec16_flambda', 'spec50_flambda', 'spec84_flambda'])
    phot_df = pd.DataFrame(zip(z0_phot_wavelength, (phot16_flambda) * ((1 + obs['z'])**2), (phot50_flambda) * ((1 + obs['z'])**2), (phot84_flambda) * ((1 + obs['z'])**2)), columns=[
                           'rest_wavelength', 'phot16_flambda', 'phot50_flambda', 'phot84_flambda'])
    spec_df.to_csv(output_dir + savename + '_model_spec.csv', index=False)
    phot_df.to_csv(output_dir + savename + '_model_phot.csv', index=False)
    model_lines_df.to_csv(output_dir + savename +
                          '_model_lines.csv', index=False)

    # Save the plot
    ax_main.legend()
    fig.savefig(save_dir + savename + '_fit.pdf')
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


def get_percentiles(res, mod, ptile=[16, 50, 84], start=0.0, thin=1, **extras):
    """Get get percentiles of the marginalized posterior for each parameter.

    :param res:
        A results dictionary, containing a "chain" and "theta_labels" keys.

    :param ptile: (optional, default: [16, 50, 84])
       A list of percentiles (integers 0 to 100) to return for each parameter.

    :param start: (optional)
       How much of the beginning of chains to throw away before calculating
       percentiles, expressed as a fraction of the total number of iterations.

    :param thin: (optional)
       Only use every ``thin`` iteration when calculating percentiles.

    :returns pcts:
       Dictionary with keys giving the parameter names and values giving the
       requested percentiles for that parameter.
    """

    parnames = np.array(res.get('theta_labels', mod.theta_labels()))
    niter = res['chain'].shape[-2]
    start_index = np.floor(start * (niter - 1)).astype(int)
    if res["chain"].ndim > 2:
        flatchain = res['chain'][:, start_index::thin, :]
        dims = flatchain.shape
        flatchain = flatchain.reshape(dims[0] * dims[1], dims[2])
    elif res["chain"].ndim == 2:
        flatchain = res["chain"][start_index::thin, :]
    pct = np.array([quantile(p, ptile, weights=res.get("weights", None))
                    for p in flatchain.T])
    return dict(zip(parnames, pct))


def read_and_make_plot(file, savename='False', mask=False):
    ''' Does all of the above to get a plot

    '''
    res, obs, mod, sps, file = read_output(file)
    all_spec, all_phot, all_mfrac, all_line_fluxes, all_line_fluxes_erg, line_waves, weights, idx_high_weights = gen_phot(
        res, obs, mod, sps)
    make_plots(res, obs, mod, sps, all_spec, all_phot, all_line_fluxes, all_line_fluxes_erg, line_waves,
               weights, idx_high_weights, file, p_plots=False, mask=mask, savename=savename)

    return [res, obs, mod, sps, all_spec, all_phot, all_line_fluxes, line_waves, weights, idx_high_weights, file]


def gen_continuum_spec(file, mask=False):
    ''' Will run through and make a plot and save the spectrum of the continuum, WITHOUT any nebular emission lines

    '''
    savename = file[:-19] + '_CONT'
    res, obs, mod, sps, file = read_output(file)
    mod.params['add_neb_emission'] = np.array([False])
    all_spec, all_phot, all_mfrac, all_line_fluxes, all_line_fluxes_erg, line_waves, weights, idx_high_weights = gen_phot(
        res, obs, mod, sps)
    make_plots(res, obs, mod, sps, all_spec, all_phot, all_line_fluxes, all_line_fluxes_erg, line_waves,
               weights, idx_high_weights, file, p_plots=False, mask=mask, savename=savename)



# Run with sys.argv when called 
groupID = sys.argv[1]
all_files = os.listdir(savio_prospect_out_dir)
target_file = [file for file in all_files if f'composite_group{groupID}_' in file]
print(f'found {target_file}')
read_and_make_plot(savio_prospect_out_dir + '/' + target_file[0])