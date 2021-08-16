# Combines the spectra and photometry to get one composite where the
# spectra of the lines are on the photometry

import os
import numpy as np
import pandas as pd
from astropy.io import ascii
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.integrate as integrate
import initialize_mosdef_dirs as imd
from scipy.optimize import curve_fit
from prospector_plot import load_obj


# -------------------
# Main
# -------------------

def main(groupID, run_name):
    '''Runs all the functions in this code, taking the output of prospector, merging the photometry with spectra, then saving

    Parameters:
    groupID (int): ID of the group to run
    run_name (str): Name of the run on prospector
    '''

    spec, phot, obs = prepare_data_for_merge(
        groupID, run_name)
    phot_merged, model_phot_df, flux_difference_scale, sum_lines_scale, integral_range, redshift_cor = merge_spec_phot(
        groupID, run_name, spec, phot, obs)
    merge_plot(groupID, run_name, phot_merged, phot, spec,
               model_phot_df, flux_difference_scale, sum_lines_scale, integral_range, redshift_cor)


def merge_spec_phot(groupID, run_name, spec, phot, obs, keep_range=7):
    '''
    keep_range (int): Number of angstroms on either side of the line to keep
    '''

    # Separate the spectrum into just the line regions that we want to use
    ha_range = [6450, 6650]
    ha_lines = [6548, 6563, 6583]
    spec_ha = spec[np.logical_and(spec['wavelength'] > ha_range[
                                  0], spec['wavelength'] < ha_range[1])]

    hb_range = [4750, 5100]
    hb_lines = [4861, 4960, 5008]
    spec_hb = spec[np.logical_and(spec['wavelength'] > hb_range[
                                  0], spec['wavelength'] < hb_range[1])]

    # Clean the lines by removing the continuum
    print('Subtracting continuum from the spectra...')
    spec_ha = remove_continuum(spec_ha, ha_lines)
    spec_hb = remove_continuum(spec_hb, hb_lines)
    # Now they have a new column 'f_lambda_sub' that is continuum subtracted

    # Find the flux in the lines in the photomtery\

    # Method: For each photometric point, figure out which fraction of its
    # flux comes from the desired region of the spectrum, then multiply the
    # flux of that point by that fraciton. Sum all of these fluxes for all
    # points to get a total flux
    # Dont think this works...
    # print('Computing flux fraction for each point...')
    # ha_total_flux = compute_flux_fraction(
    #     phot, phot_filters, phot_filter_keys, ha_range)
    # hb_total_flux = compute_flux_fraction(
    #     phot, phot_filters, phot_filter_keys, hb_range)

    # Second Method:
    # Fit the photometry witha  polynomial and integrate teh flux in the ha_range and hb_range regions
    # Scale the spectra to have the same flux within that range. Doesn't account for how much broader it would be though
    # maybe after replacing, we could lower the flux of the other points using
    # the method above (computing how much flux that point would be boosted
    # from the line)? I think that might work
    # print('Interpolating and computing flux in the photometry')
    # ha_total_flux = compute_phot_flux(phot, ha_range)
    # hb_total_flux = compute_phot_flux(phot, hb_range)

    # Third method
    # Model a line as a Gaussian. Fit the spectrum to get a width of the line. Fix the sigma
    # Take the line over the filter curve of all the points, and try to fit what the best height would be to match the observations
    # Scale the spectrum to match that flux value

    # Read in the continuum from prospector
    
    print(f'Reading Continuum from prospector file')
    model_phot_df  = ascii.read(imd.prospector_fit_csvs_dir + f'/{run_name}_csvs' + f'/{groupID}_cont_phot.csv').to_pandas()
    model_spec_df = ascii.read(imd.prospector_fit_csvs_dir + f'/{run_name}_csvs' + f'/{groupID}_cont_spec.csv').to_pandas()
    redshift_cor = (1+obs['z'])**2

    # Old method with the spectrum is at the end of the .py file, now using
    # photometry from model without emission
    print(f'Integrating flux between observations and model photometry')
    # Interpolate the model and observations for easy comparison
    model_phot_interp = interpolate.interp1d(
        model_phot_df['rest_wavelength'], model_phot_df['phot50_flambda']*redshift_cor, bounds_error=False, fill_value=0)
    obs_interp = interpolate.interp1d(
        phot['rest_wavelength'], phot['f_lambda_scaled'])
    # integrate to find the total difference btween spectum and photometry
    # (one possible method, then scale the liens to contain this flux
    # difference)
    integral_range = (4100, 10000)
    integrated_model_phot = integrate.quad(
        model_phot_interp, integral_range[0], integral_range[1])
    integrated_obs = integrate.quad(
        obs_interp, integral_range[0], integral_range[1])
    flux_difference = integrated_obs[0] - integrated_model_phot[0]

    # Read in the emission line fits 
    fit_df = ascii.read(imd.emission_fit_csvs_dir + f'/{groupID}_emission_fits_scaled.csv').to_pandas()
    total_line_flux = np.sum(fit_df['flux'])
    flux_difference_scale = flux_difference / total_line_flux
    print(f'Flux difference method would give a scale of {flux_difference_scale}')

    # Sum the lines method - assume prospector got the total flux in the lines
    # right, but not the ratio
    # This uses prospector's version of the lines - I should be fitting them myself
    model_lines_df = ascii.read(imd.prospector_fit_csvs_dir + f'/{run_name}_csvs' + f'/{groupID}_lines.csv').to_pandas()
    mosdef_lines = hb_lines + ha_lines
    mosdef_line_fluxes = []
    for line_wave in mosdef_lines:
        line_idx = np.argmin(
            np.abs(model_lines_df['rest_wavelength'] - line_wave))
        mosdef_line_fluxes.append(model_lines_df.iloc[line_idx]['lines50_erg'])
    sum_mosdef_line_fluxes = np.sum(mosdef_line_fluxes)
    sum_lines_scale = sum_mosdef_line_fluxes / total_line_flux
    print(f'Using prospector fits, sum lines gives {sum_lines_scale * np.sqrt(redshift_cor)}')

    # Read in the lines that were fit in the same way as the mosdef spectra
    model_lines_df = ascii.read(imd.prospector_emission_fits_dir + f'/{run_name}_emission_fits/{groupID}_emission_fits.csv').to_pandas()
    total_model_line_flux = np.sum(model_lines_df['flux'])
    sum_lines_scale = total_model_line_flux / total_line_flux

    print(f'Sum lines method would give a scale of {sum_lines_scale}')

    # print('Finding optimal scale for the spectrum')
    # scale, target_fluxes, opt_point_fluxes, phot_waves = find_line_flux(fit_df,
    #                                                                     phot, phot_filters, phot_filter_keys, groupID, model_phot_df)

    # Now we prepare a new dataframe that contains only the lines that will be
    # merged with the photometry dataframe

    all_lines = hb_lines + ha_lines
    keep_idxs = []
    for line in all_lines:
        center_spec_idx = np.argmin(np.abs(line - spec['wavelength']))
        keep_idxs = keep_idxs + \
            list(np.arange(center_spec_idx - keep_range,
                           center_spec_idx + keep_range + 1, 1))
    keep_idxs = np.sort(list(set(keep_idxs)))

    # This contains just the rows around the lines
    merge_spec_df = spec.iloc[keep_idxs][
        ['wavelength', 'f_lambda_scaled', 'err_f_lambda_scaled']]
    merge_spec_df['f_lambda'] = flux_difference_scale * merge_spec_df['f_lambda_scaled']
    merge_spec_df = merge_spec_df.rename(
        columns={'wavelength': 'rest_wavelength', 'err_f_lambda_scaled': 'err_f_lambda_d'})
    merge_spec_df['err_f_lambda_u'] = merge_spec_df['err_f_lambda_d']

    merge_phot_df = phot[['rest_wavelength', 'f_lambda_scaled', 'err_f_lambda_u_scaled', 'err_f_lambda_d_scaled']]
    merge_phot_df = merge_phot_df.rename(columns={'f_lambda_scaled': 'f_lambda', 'err_f_lambda_u_scaled': 'err_f_lambda_u', 'err_f_lambda_d_scaled': 'err_f_lambda_d'})

    # Merge the dataframes
    phot_merged = pd.concat([merge_phot_df, merge_spec_df]) 

    return phot_merged, model_phot_df, flux_difference_scale, sum_lines_scale, integral_range, redshift_cor


def prepare_data_for_merge(groupID, run_name, norm_method='cluster_norm'):
    """Prepares and reads the data for the main function

    Parameters:
    groupID (int): ID of the group to merge
    norm_method (str): Method used to normalize the composite spectra



    Returns:
    spec (pd.DataFrame): DataFrame with the spectroscopy
    phot (pd.DataFrame): Dataframe with the photometry
    phot_filters (dict): Dict that contains dataframes for the filters, accessed by calling the nearest integer point
    phot_filter_keys (list): List of points as strings that can be used to call the phot_filters dict
    """
    obs = load_obj(f'{groupID}_obs', run_name)

    # Read in spectrum and photometry from their corresponding folders
    print(f'Reading spectrum {groupID}...')
    spec = ascii.read(imd.composite_spec_dir + f'/{norm_method}_csvs/{groupID}_spectrum_scaled.csv').to_pandas()
    print(f'Reading photometry {groupID}...')
    phot = ascii.read(imd.composite_sed_csvs_dir + f'/{groupID}_sed.csv').to_pandas()
    # filt_folder = imd.composite_filter_csvs_dir + f'/{groupID}_filter_csvs/'
    # filt_files = [file for file in os.listdir(filt_folder) if '.csv' in file]
    # phot_filters = {}
    # phot_filter_keys = []
    # for i in range(len(filt_files)):
    #     filt_file = filt_files[i]
    #     point = filt_file.split('.')[0].split('_')[1]
    #     print(f'Reading in filter for point {point}...')
    #     phot_filters[point] = ascii.read(filt_folder + filt_file).to_pandas()
    #     phot_filter_keys.append(point)
    

    return spec, phot, obs


# -------------------
# Computing Scale
# -------------------


def find_line_flux(fit_df, phot, phot_filters, phot_filter_keys, groupID, model_phot_df):
    """For each point, compute what fraction of the flux is within the line range

    Parameters:
    fit_df (pd.DataFrame): Dataframe containing data from line fitting code
    phot (pd.DataFrame): Photometry for the composite
    phot_filters (dict): Dictionary of the read-in filter curves, keyed by the central point
    phot_filter_keys (list): List of central points for the filters
    groupID (int): ID of the group
    model_phot_df (func): Dataframe of model photometric values, just the continuum


    Returns:
    opt_scale (float): Optimal scale for the spectum
    target_fluxes (array): Total flux in each photometric point (found by flux*integral(transmission))
    opt_point_fluxes (array): What the final fluxes were when applying the correct scale
    """
    # Set a threshold - we will only consider points for which the
    # transmission in the line region is greater than 0.1
    transmission_thresh = 0.6

    phot_filter_keys_int = [int(key) for key in phot_filter_keys]

    line_centers = fit_df['line_center_rest']

    def line_fit_func(phot_waves, scale):
        point_fluxes = []
        for phot_wave in phot_waves:
            transmissions = []
            transmitted_fluxes = []

            # Get teh filter curve for the given point
            key = np.argmin(np.abs(phot_wave - np.array(phot_filter_keys_int)))
            filter_curve = phot_filters[phot_filter_keys[key]]

            # Store the transmission of that point for each line
            for i in range(len(line_centers)):
                line_idx = np.argmin(
                    np.abs(filter_curve['rest_wavelength'] - line_centers.iloc[i]))
                line_transmission = filter_curve.iloc[line_idx]['transmission']
                transmissions.append(line_transmission)

            # Compute what the flux would be in that line scaled through the
            # transmission
            for i in range(len(transmissions)):
                if transmissions[i] > transmission_thresh:
                    line_flux = (scale * fit_df['amplitude'].iloc[i]) * \
                        fit_df['sigma'].iloc[i] * np.sqrt(2 * np.pi)
                    transmitted_flux = line_flux * transmissions[i]
                    transmitted_fluxes.append(transmitted_flux)
                else:
                    transmitted_fluxes.append(0)

            # This is the flux expected to be observed, given the scale of the
            # spectrum
            point_flux = np.sum(transmitted_fluxes)
            point_fluxes.append(point_flux)

        return np.array(point_fluxes)

    def get_target_flux(phot_wave, model_phot_df):
        # This is the flux in the point - integrating the transmission curve and
        # multiplying by the photometric flux
        transmissions = []
        transmitted_fluxes = []

        # Get the filter curve for the given point
        key = np.argmin(np.abs(phot_wave - np.array(phot_filter_keys_int)))
        filter_curve = phot_filters[phot_filter_keys[key]]
        curve_interp = interpolate.interp1d(
            filter_curve['rest_wavelength'], filter_curve['transmission'])
        integrated_transmission = integrate.quad(curve_interp, 801, 39999)

        # We also want to include the contribution from the continuum - this
        # needs to use the model continuum, and subtract it off from the point
        phot_idx = np.argmin(
            np.abs(phot_wave - np.array(phot['rest_wavelength'])))
        target_flux = integrated_transmission[
            0] * (phot.iloc[phot_idx]['f_lambda_scaled'] - model_phot_df.iloc[phot_idx]['phot50_flambda'])
        target_flux = (phot.iloc[phot_idx]['f_lambda_scaled'] -
                       model_phot_df.iloc[phot_idx]['phot50_flambda']) * integrated_transmission[0]

        return target_flux

    # First, check which points willl have too low transmission from the
    # lines, and therefore will have a point_flux of zero
    all_phot_waves = np.array(phot['rest_wavelength'])
    all_point_fluxes = line_fit_func(all_phot_waves, 1)
    good_idxs = [i for i, flux in enumerate(all_point_fluxes) if flux != 0]

    phot_waves = np.array(phot.iloc[good_idxs]['rest_wavelength'])

    # For each point, compute what the flux in the line would have to be
    target_fluxes = np.array([get_target_flux(phot_wave, model_phot_df)
                              for phot_wave in phot_waves])

    # Model the line as a gaussian with width sigma  and center center
    opt_scale, pcov = curve_fit(line_fit_func, phot_waves, target_fluxes)

    opt_point_fluxes = line_fit_func(phot_waves, opt_scale)

    return opt_scale, target_fluxes, opt_point_fluxes, phot_waves


# -------------------
# Plotting
# -------------------

def merge_plot(groupID, run_name, phot_merged, phot, spec, model_phot_df, flux_difference_scale, sum_lines_scale, integral_range, redshift_cor):
    """Plots merging process

    Parameters:

    Returns:
    Saves a pdf of the fits for all of the lines
    """
    axisfont = 14
    ticksize = 12
    ticks = 8
    titlefont = 24
    legendfont = 14
    textfont = 16

    fig = plt.figure(figsize=(12, 8))
    ax_final = fig.add_axes([0.08, 0.08, 0.6, 0.9])
    # ax_scale_offsets = fig.add_axes([0.75, 0.08, 0.23, 0.23])
    ax_ha = fig.add_axes([0.75, 0.40, 0.23, 0.23])
    ax_hb = fig.add_axes([0.75, 0.72, 0.23, 0.23])
    ax_arr = [ax_final, ax_ha, ax_hb]

    for ax in ax_arr:
        ax.plot(phot_merged['rest_wavelength'], phot_merged['rest_wavelength'] * phot_merged[
            'f_lambda'], ls='', marker='o', color='black', label='Phot + Spec')
        ax.plot(phot['rest_wavelength'], phot['rest_wavelength'] * phot['f_lambda_scaled'],
                ls='', marker='o', color='orange', label='Photometry')
        ax.set_ylabel("$\lambda$ F$_\lambda$")
        ax.set_xlabel("Wavelength ($\AA$)")
        ax.plot(model_phot_df['rest_wavelength'], model_phot_df['rest_wavelength'] * model_phot_df['phot50_flambda']*redshift_cor,
                ls='', marker='o', color='blue', label='Continuum Photometry')

    ax_final.legend()

    ax_final.set_ylim(0.8 * np.percentile(phot_merged['rest_wavelength'] * phot_merged['f_lambda'], 1),
                      1.1 * np.percentile(phot_merged['rest_wavelength'] * phot_merged['f_lambda'], 80))

    ax_ha.set_xlim(6540, 6590)
    ax_hb.set_xlim(4800, 5050)
    fig.text(0.68, 0.9, 'H_alpha', transform=ax_ha.transAxes)
    fig.text(0.1, 0.9, 'H_beta & OIII', transform=ax_hb.transAxes)
    ax_final.set_xlim(
        np.min(phot_merged['rest_wavelength']) - 30, np.max(phot_merged['rest_wavelength']) + 3000)
    ax_final.set_xscale('log')
    # print(f'Scale = {round(scale[0], 3)}')

    # Fitting axis
    # opt_point_fluxes = np.array([opt_point_fluxes[i][0]
    #                              for i in range(len(opt_point_fluxes))])
    # ax_scale_offsets.plot(
    #     phot_waves, (target_fluxes / opt_point_fluxes), color='black', ls='', marker='o')
    # ax_scale_offsets.set_xlabel("Wavelength ($\AA$)")
    # ax_scale_offsets.set_ylabel("Target flux / Transmitted Line flux")

    # fig.text(0.1, 0.9, f'Scale = {round(scale[0], 3)}', transform=ax_final.transAxes)
    fig.text(0.1, 0.15, f'Flux Diff Scale = {round(flux_difference_scale, 3)}', transform=ax_final.transAxes)
    fig.text(0.1, 0.10, f'Sum Lines Scale = {round(sum_lines_scale, 3)}', transform=ax_final.transAxes)

    # Color in the region we are integrating over for clarity
    # Need the indices of wavelength that are between the integral values
    model_phot_interp = interpolate.interp1d(
        model_phot_df['rest_wavelength'], model_phot_df['rest_wavelength'] * model_phot_df['phot50_flambda']*redshift_cor, bounds_error=False, fill_value=0)
    obs_interp = interpolate.interp1d(
        phot['rest_wavelength'], phot['rest_wavelength'] * phot[
            'f_lambda_scaled'])
    waves = np.arange(integral_range[0], integral_range[1], 1)
    ax_final.fill_between(waves, obs_interp(waves), model_phot_interp(waves), color='r', alpha=0.5)

    imd.check_and_make_dir(imd.composite_seds_spec_images_dir + f'/{run_name}_images')
    fig.savefig(imd.composite_seds_spec_images_dir + f'/{run_name}_images' + f'/{groupID}_sed_spec.pdf')


# -------------------
# Modifying Spectra
# -------------------

def remove_continuum(spec, line_locs, width=10):
    """Removes the continuum from the spectrum, leaving just the lines

    Parameters:
    spec (pd.DataFrame): Spectrum for which to remove the continuum
    line_locs (list): Locations of the lines in Angstroms
    width (int): One-driectional width to mask when finding the continuum


    Returns:
    spec (pd.DataFrame): Same dataframe, but with new column 'f_lambda_sub' for continuum subtracted f_lambda values
    """

    # Currently assuming that the continuum is flat in the line region
    mask = np.ones(len(spec), dtype=bool)
    # First, make a mask to filter out the lines
    for line_loc in line_locs:
        mask_line = np.logical_or(spec['wavelength'] < (
            line_loc - width), spec['wavelength'] > (line_loc + width))
        mask = mask & mask_line

    spec_cont = spec[mask]
    median_cont = np.median(spec_cont['f_lambda_scaled'])
    spec['f_lambda_sub'] = spec['f_lambda_scaled'] - median_cont
    return spec


# -------------------
# Old functions
# -------------------

def compute_flux_fraction(phot, phot_filters, phot_filter_keys, line_range):
    """For each point, compute what fraction of the flux is within the line range

    Parameters:
    phot (pd.DataFrame): Photometry for the composite
    phot_filters (dict): Dictionary of the read-in filter curves, keyed by the central point
    phot_filter_keys (list): List of central points for the filters
    line_range (tuple): Rnage of [low, high] for the region to compute the flux fraction for


    Returns:
    total_flux (float): Integrated flux value over the line region
    """
    phot_filter_keys_int = [int(key) for key in phot_filter_keys]
    flux_fractions = []

    # Loop over all points
    for i in range(len(phot)):
        # Find the filter curve corresponding to that point
        key = np.argmin(
            np.abs(phot.iloc[i]['rest_wavelength'] - np.array(phot_filter_keys_int)))
        filter_curve = phot_filters[phot_filter_keys[key]]
        # Find the fraction of transmission that is within the line_range
        total_transmission = np.sum(filter_curve['transmission'])
        line_idxs = np.logical_and(filter_curve['rest_wavelength'] > line_range[
            0], filter_curve['rest_wavelength'] < line_range[1])
        line_transmission = np.sum(filter_curve[line_idxs]['transmission'])
        transmission_fraction = line_transmission / total_transmission
        # Compute the fraction of flux that is from this range
        flux_fraction = transmission_fraction * phot.iloc[i]['f_lambda_scaled']
        flux_fractions.append(flux_fraction)
        print(f'{phot_filter_keys[i]} fraction: {transmission_fraction}')

    total_phot_flux = np.sum(flux_fractions)

    # This is just the flux assuming that the filter curve is flat in the
    # region (right?), so now we multiply by the wavelength range length
    total_flux = total_phot_flux * (line_range[1] - line_range[0])
    return total_flux


def compute_phot_flux(phot, line_range):
    """Compute the total flux wihtin the line_range region

    Parameters:
    phot (pd.DataFrame): Photometry for the composite
    line_range (tuple): Rnage of [low, high] for the region to compute the flux fraction for


    Returns:
    total_flux (float): Integrated flux value over the line region
    """
    phot_interp = interpolate.interp1d(
        phot['rest_wavelength'], phot['f_lambda_scaled'])
    total_flux = integrate.quad(phot_interp, line_range[0], line_range[1])
    return total_flux


# --------------------
# Old Code
# --------------------
'''
print(f'Integrating flux between observations and model spectrum')
    # Going to try to use the spectra here - mask the lines, interpolate on both the remaining spectrum and photometry, find area under the curve
    # First, we need to mask the spectra lines - I'll take any line that is
    # significantly abve the median of the spectrum and remove a small area
    # around it:
    spec_range = np.logical_and(model_spec_df['rest_wavelength'] > 2800, model_spec_df[
                                'rest_wavelength'] < 11000)
    # Identifies where the lines are by finding where points are at least
    # 1.5xx greater than the median of points around them
    line_peaks = [model_spec_df.iloc[i]['spec50_flambda'] > 1.5 *
                  np.median(model_spec_df.iloc[i - 25:i + 25]['spec50_flambda']) for i in model_spec_df[spec_range].index]
    mask_indices = model_spec_df[spec_range][line_peaks].index
    expanded_mask_indices = mask_indices
    # Expands the mask to include 10 points on either side of any masked point
    for i in range(len(mask_indices)):
        expanded = np.arange(mask_indices[i] - 10, mask_indices[i] + 11)
        expanded_mask_indices = np.array(
            list((set(np.hstack([expanded_mask_indices, expanded])))))
    expanded_mask_indices = np.sort(expanded_mask_indices)
    masked_model_spec_df = model_spec_df[
        spec_range].drop(expanded_mask_indices)
    # Interpolate the spectra nad observations for easy comparison
    model_spec_interp = interpolate.interp1d(
        masked_model_spec_df['rest_wavelength'], masked_model_spec_df['spec50_flambda'], bounds_error=False, fill_value=0)
    obs_interp = interpolate.interp1d(
        phot['rest_wavelength'], phot['f_lambda'])
    # integrate to find the total difference btween spectum and photometry
    # (one possible method, then scale the liens to contain this flux
    # difference)
    interal_range = (3000, 10000)
    integrated_model_spec = integrate.quad(
        model_spec_interp, interal_range[0], interal_range[1])
    integrated_obs = integrate.quad(
        obs_interp, interal_range[0], interal_range[1])
    flux_difference = integrated_obs[0] - integrated_model_spec[0]
    '''

for groupID in range(29):
    try:
        main(groupID, 'redshift_maggies')
    except:
        pass