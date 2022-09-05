# Codes for stacking MOSDEF spectra
# stack_spectra(groupID, norm_method) to stack within clusters

import sys
import time
import numpy as np
import pandas as pd
from astropy.io import ascii
from mosdef_obj_data_funcs import read_sed, get_mosdef_obj, read_composite_sed, read_fast_continuum, setup_get_ssfr, merge_ar_ssfr, merge_emission
import matplotlib.pyplot as plt
from scipy import interpolate
import initialize_mosdef_dirs as imd
import cluster_data_funcs as cdf
from spectra_funcs import read_axis_ratio_spectrum, clip_skylines, check_line_coverage, get_spectra_files, median_bin_spec, read_spectrum, get_too_low_gals, norm_spec_sed, read_composite_spectrum, prepare_mock_observe_spectrum, mock_observe_spectrum
import matplotlib.patches as patches
from axis_ratio_funcs import read_interp_axis_ratio, filter_ar_df, read_filtered_ar_df
from operator import itemgetter
from itertools import *
from matplotlib import patches
from read_FAST_spec import read_FAST_file
from cosmology_calcs import flux_to_luminosity
from sfms_bins import *

axis_ratio_catalog = ascii.read(imd.loc_axis_ratio_cat).to_pandas()


def stack_spectra(groupID, norm_method, re_observe=False, mask_negatives=False, ignore_low_spectra=False, axis_ratio_df=[], axis_group=0, save_name='', scale_factors=0, stack_type='median', bootstrap=0):
    """Stack all the spectra for every object in a given group

    Parameters:
    groupID (int): ID of the cluster to perform the stacking
    norm_method (str): Method to normalize - 'cluster_norm' for same norms as sed stacking, 'composite_sed_norm' to normalize to the composite, 'composite_filter' to observe the spectrum in a filter and use one point to normalize, 'positive_median' to take the median of all poitive values and scale that
    re_observe (boolean): Set to True if using 'composite_filter' and you want to re-observe all of the spectra
    mask_negatives (boolean): Set to True to mask all negative values in the spectra
    ignore_low_spectra (boolean): Set to True to ignore all spectra with negative median values
    axis_ratio_df (pd.DataFrame): Set to a dataframe of axis ratios and it will stack all spectra within that dataframe. Also set axis_group
    axis_group (int): Number of the axis ratio group
    save_name (str): location to save the files
    scale_factors (list): Set to a list to manually choose the normalization for each object - requires norm_method 'manual'
    stack_type (str): either 'mean' or 'median' to choose whether to use the mean of each value or median for the stack
    bootstrap (int): Number of times to bootstrap the stacking. Set to 0 to not bootstrap. 

    Returns:
    """
    # Used to check if this is the first time through the loop
    if re_observe == True:
        first_loop = 1
        spectrum_files_list = []
        scaled_observed_fluxes = []
        fraction_in_ranges = []
        composite_waves = []
        composite_fluxes = []

    # Check if we are using axis ratio stacking or not
    axis_stack = False
    if len(axis_ratio_df) > 0:
        axis_stack = True

    if axis_stack:
        mosdef_objs = [get_mosdef_obj(axis_ratio_df.iloc[i]['field'], axis_ratio_df.iloc[
            i]['v4id']) for i in range(len(axis_ratio_df))]

    else:
        composite_sed = read_composite_sed(groupID)

        cluster_names, fields_ids = cdf.get_cluster_fields_ids(groupID)

        mosdef_objs = [get_mosdef_obj(field, int(v4id))
                       for field, v4id in fields_ids]

    # min, max, step-size
    spectrum_wavelength = np.arange(3000, 10000, 1)
    if axis_stack:
        spectrum_wavelength = np.arange(3000, 10000, 0.5)

    # Now that we have the mosdef objs for each galaxy in the cluster, we need
    # to loop over each one
    interp_cluster_spectra_dfs = []
    norm_factors = []
    loop_count = -1
    for mosdef_obj in mosdef_objs:
        loop_count += 1
        # Get the redshift and normalization
        z_spec = mosdef_obj['Z_MOSFIRE']
        field = mosdef_obj['FIELD_STR']
        v4id = mosdef_obj['V4ID']

        norm_sed = read_sed(field, v4id, norm=True)
        
        # print(f'Reading Spectra for {field} {v4id}, z={z_spec:.3f}')
        # Check to see if the galaxy includes all emission lines

        coverage_list = [
            ('Halpha', 6564.61),
            ('Hbeta', 4862.68),
            ('O3_5008', 5008.24),
            ('O3_4960', 4960.295),
            ('N2_6585', 6585.27)
        ]

        # Only check the coverage if we are not an axis stack mode - it is pre-checked there
        if not axis_stack:
            covered = check_line_coverage(mosdef_obj, coverage_list)
            if covered == False:
                continue
        # Find all the spectra files corresponding to this object
        spectra_files = get_spectra_files(mosdef_obj)
        for spectrum_file in spectra_files:
            spectrum_df = read_spectrum(mosdef_obj, spectrum_file)

            # Clip the skylines:
            spectrum_df['f_lambda_clip'], spectrum_df['mask'], spectrum_df['err_f_lambda_clip'] = clip_skylines(
                spectrum_df['obs_wavelength'], spectrum_df['f_lambda'], spectrum_df['err_f_lambda'], mask_negatives=mask_negatives)

            if ignore_low_spectra:
                # Find the matching wavelength
                med_spec_wave = np.median(spectrum_df['rest_wavelength'])
                sed_idx = np.argmin(
                    np.abs(norm_sed['rest_wavelength'] - med_spec_wave))
                sed_val = norm_sed['f_lambda'].iloc[sed_idx]
                spec_median = np.median(spectrum_df[spectrum_df['f_lambda_clip'] != 0][
                    'f_lambda_clip'])
                ratio = sed_val / spec_median
                # print(f'Ratio = {ratio}')
                if ratio > 2 or ratio < 0:
                    print('Skipping')
                    continue

            # NORMALZE - HOW BEST TO DO THIS?
            # Original Method - using the computed norm_factors form composite sed formation
            if norm_method == 'cluster_norm':
                norm_factor = np.median(norm_sed['norm_factor'])
            elif norm_method == 'composite_sed_norm':
                try:
                    norm_factor, spec_correlate, used_points = norm_spec_sed(
                        composite_sed, spectrum_df)
                except Exception as expt:
                    print('Could not normalize')
                    print(expt)
            elif norm_method == 'composite_filter':
                if re_observe == True:
                    if first_loop == 1:
                        filter_dfs, bounds, points = prepare_mock_observe_spectrum(
                            groupID)
                        print(f'Done reading filters for group {groupID}')
                        first_loop = 0
                    scaled_flux_filter_nu, fraction_in_range, composite_wave, composite_flux = mock_observe_spectrum(
                        composite_sed, spectrum_df, filter_dfs, bounds, points)
                    print(f'Observed spectrum {spectrum_file}')
                    spectrum_files_list.append(spectrum_file)
                    scaled_observed_fluxes.append(scaled_flux_filter_nu)
                    fraction_in_ranges.append(fraction_in_range)
                    composite_waves.append(composite_wave)
                    composite_fluxes.append(composite_flux)
                    continue
                else:
                    observed_spec_df = ascii.read(imd.cluster_dir + f'/composite_spectra/composite_filter/{groupID}_observed_specs.csv').to_pandas()
                    file_idx = observed_spec_df['filename'] == spectrum_file
                    spectrum_flux_filter = observed_spec_df[
                        file_idx]['observed_flux']
                    composite_flux_filter = observed_spec_df[
                        file_idx]['composite_flux']
                    norm_factor = (composite_flux_filter /
                                   spectrum_flux_filter).iloc[0]
                    if norm_factor < 0 or norm_factor > 100:
                        norm_factor = 0
            elif norm_method == 'positive_median':
                med_wave = np.median(spectrum_df['rest_wavelength'])
                med_pos_flux = np.median(
                    spectrum_df[spectrum_df['f_lambda_clip'] > 0]['f_lambda_clip'])
                interp_sed = interpolate.interp1d(
                    composite_sed['rest_wavelength'], composite_sed['f_lambda'])
                scale_sed_flux = interp_sed(med_wave)
                norm_factor = scale_sed_flux / med_pos_flux
            elif norm_method == 'manual':
                pass
            else:
                sys.exit(
                    'Select norm_method: "cluster_norm", "composite_sed_norm", ')

            # If stacking in axis ratio groups, do NOT normalize
            # if axis_stack:
            #     norm_factor = 1.

            # New method - when staking axis ratio groups, normalize by halpha
            if axis_stack:
                axis_idx = np.logical_and(axis_ratio_catalog['field'] == field, axis_ratio_catalog['v4id'] == v4id)
                ha_flux = axis_ratio_catalog[axis_idx].iloc[0]['ha_flux']
                norm_factor = norm_axis_stack(ha_flux, z_spec)  

                # Also, grab the FAST continuum and normalize it in the same way, will be used later
                fast_file_df = read_FAST_file(field, v4id)
                # Convert to rest wavelength
                fast_file_df['rest_wavelength'] = fast_file_df['wavelength'] / (1+z_spec)
                fast_file_df['f_lambda_norm'] = fast_file_df['f_lambda']*norm_factor
                imd.check_and_make_dir(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_conts/{axis_group}_conts/')
                fast_file_df.to_csv(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_conts/{axis_group}_conts/{field}_{v4id}_cont.csv', index=False)
                


            # Override all else if it's manual
            if norm_method == 'manual':
                norm_factor = scale_factors.iloc[loop_count]

            print(f'    Norm factor: {norm_factor}')
            # Read in the continuum and normalize that
            continuum_df = read_fast_continuum(mosdef_obj)
            continuum_df['f_lambda_norm'] = continuum_df[
                'f_lambda'] * norm_factor

            # Normalize the spectra
            spectrum_df['f_lambda_norm'] = spectrum_df[
                'f_lambda_clip'] * norm_factor

            spectrum_df['err_f_lambda_norm'] = spectrum_df[
                'err_f_lambda_clip'] * norm_factor

            # These create a padded_mask which is True for each of the pixesl
            # AROUND the masked lines
            mask = spectrum_df['f_lambda_clip'] == 0
            padded_mask = add_trues(mask)

            # Interpolate using the mask, then this will not drag down points
            # near masked values
            norm_interp = interpolate.interp1d(
                spectrum_df[np.logical_not(mask)]['rest_wavelength'], spectrum_df[np.logical_not(mask)]['f_lambda_norm'], fill_value=0, bounds_error=False)
            err_interp = interpolate.interp1d(
                spectrum_df[np.logical_not(mask)]['rest_wavelength'], spectrum_df[np.logical_not(mask)]['err_f_lambda_norm'], fill_value=0, bounds_error=False)

            # Interpolating the continuum is easier, no need to consider mask
            cont_interp = interpolate.interp1d(
                continuum_df['rest_wavelength'], continuum_df['f_lambda_norm'], fill_value=0, bounds_error=False)

            spectrum_flux_norm = norm_interp(spectrum_wavelength)
            spectrum_err_norm = err_interp(spectrum_wavelength)
            cont_norm = cont_interp(spectrum_wavelength)

            # After interpolation, we need to set points to zero that fall in
            # the ranges
            idx_clips = clip_spectrum(
                spectrum_df, padded_mask, spectrum_wavelength, spectrum_flux_norm)
            for idx in idx_clips:
                spectrum_flux_norm[idx] = 0
            idx_zeros = spectrum_flux_norm == 0
            spectrum_err_norm[idx_zeros] = 0
            cont_norm[idx_zeros] = 0

            # Store the interpolated spectrum
            interp_spectrum_df = pd.DataFrame(zip(spectrum_wavelength, spectrum_flux_norm, spectrum_err_norm, cont_norm),
                                              columns=['rest_wavelength', 'f_lambda_norm', 'err_f_lambda_norm', 'cont_norm'])
            if axis_stack:
                imd.check_and_make_dir(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_indiv_spectra')
                imd.check_and_make_dir(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_indiv_spectra/{axis_group}')
                interp_spectrum_df.to_csv(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_indiv_spectra/{axis_group}/{field}_{v4id}_interp_spec_df.csv', index=False)
            interp_cluster_spectra_dfs.append(interp_spectrum_df)
            norm_factors.append(norm_factor)

    if re_observe == True:
        observed_spec_df = pd.DataFrame(zip(spectrum_files_list, scaled_observed_fluxes, fraction_in_ranges, composite_waves, composite_fluxes), columns=[
            'filename', 'observed_flux', 'fraction_in_range', 'composite_wavelength', 'composite_flux'])
        observed_spec_df.to_csv(imd.cluster_dir + f'/composite_spectra/composite_filter/{groupID}_observed_specs.csv', index=False)
        print(f'Saved {groupID}_observed_specs.csv')
        return



    total_spec, total_cont, total_errs, number_specs_by_wave, norm_value_specs_by_wave = perform_stack(stack_type, interp_cluster_spectra_dfs, norm_factors)
    print(f'bootstrap {bootstrap}')
    #If boostrrapping is turned on, perform multiple stacks with different subsets of the data, and save each of these
    if bootstrap>0:
        start_strap = time.time()
        print('Starting bootstrap')
        bootstrap_count = 0
        imd.check_and_make_dir(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_spectra_boots/')
        while bootstrap_count<bootstrap:
            # Resample the interp_spectrum_dfs and corresponding norm factors to bootstrap with:
            n_gals = len(interp_cluster_spectra_dfs)
            # Grab the indices of which galaxies to use
            print('Resampling...')
            keys = np.random.choice(range(n_gals), size=n_gals) 
            boot_interp_cluster_spectra_dfs = [interp_cluster_spectra_dfs[keys[k]] for k in range(n_gals)]
            boot_norm_factors = [norm_factors[keys[k]] for k in range(n_gals)]
            print('Stacking...')
            boot_total_spec, boot_total_cont, boot_total_errs, boot_number_specs_by_wave, boot_norm_value_specs_by_wave = perform_stack(stack_type, boot_interp_cluster_spectra_dfs, boot_norm_factors)
            print('Making dataframe...')
            boot_total_spec_df = pd.DataFrame(zip(spectrum_wavelength, boot_total_spec, boot_total_errs, boot_total_cont, boot_number_specs_by_wave, boot_norm_value_specs_by_wave), columns=['wavelength', 'f_lambda', 'err_f_lambda', 'cont_f_lambda', 'n_galaxies', 'norm_value_summed'])
            print('Saving...')
            boot_total_spec_df.to_csv(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_spectra_boots/{axis_group}_spectrum_{bootstrap_count}.csv', index=False)
            bootstrap_count=bootstrap_count+1
        end_strap = time.time()
        print(f'Bootstrap took {end_strap-start_strap} seconds')


    # Now we have divided each point by the sum of the normalizations that
    # contributed to it.
    total_spec_df = pd.DataFrame(zip(spectrum_wavelength, total_spec, total_errs, total_cont, number_specs_by_wave, norm_value_specs_by_wave),
                                 columns=['wavelength', 'f_lambda', 'err_f_lambda', 'cont_f_lambda', 'n_galaxies', 'norm_value_summed'])

    if axis_stack:
        median_ratio = np.median(axis_ratio_df['use_ratio'])
        total_spec_df.to_csv(
            imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_spectra/{axis_group}_spectrum.csv', index=False)
        plot_spec(groupID, norm_method,
                  axis_group=axis_group, save_name=save_name)

    else:
        save_dir = imd.composite_spec_dir + f'/{norm_method}_csvs'
        imd.check_and_make_dir(save_dir)
        total_spec_df.to_csv(save_dir + f'/{groupID}_spectrum.csv', index=False)

        plot_spec(groupID, norm_method)
    return




def plot_spec(groupID, norm_method, mask_negatives=False, thresh=0.1, axis_group=-1, save_name='', axis_obj = 'False'):
    """Plots the spectrum

    Parameters:
    groupID (int): id of the group that you are working with
    thresh (float): From 0 to 1, fraction where less than this percentage of the group will be marked as a bad part of the spectrum
    axis_group(int): Set the the group number to instead plot for the aixs_ratio group

    Returns:
    """
    axisfont = 14
    ticksize = 12
    ticks = 8

    if axis_group > -1:
        total_spec_df = read_axis_ratio_spectrum(
            axis_group, save_name=save_name)
    else:
        total_spec_df = read_composite_spectrum(groupID, norm_method)

    if axis_obj == 'False':
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_axes([0.09, 0.35, 0.88, 0.60])
        ax_Ha = fig.add_axes([0.69, 0.70, 0.25, 0.21])
        ax_contribute = fig.add_axes([0.09, 0.08, 0.88, 0.22])

        zoom_box_color = 'mediumseagreen'
        ax_Ha.spines['bottom'].set_color(zoom_box_color)
        ax_Ha.spines['top'].set_color(zoom_box_color)
        ax_Ha.spines['right'].set_color(zoom_box_color)
        ax_Ha.spines['left'].set_color(zoom_box_color)

    spectrum = total_spec_df['f_lambda']
    continuum = total_spec_df['cont_f_lambda']
    err_spectrum = total_spec_df['err_f_lambda']
    wavelength = total_spec_df['wavelength']
    n_galaxies = total_spec_df['n_galaxies']
    norm_value_summed = total_spec_df['norm_value_summed']

    # Median bin the spectrum
    wave_bin, spec_bin = median_bin_spec(wavelength, spectrum)

    too_low_gals, plot_cut, not_plot_cut, n_gals_in_group, cutoff, cutoff_low, cutoff_high = get_too_low_gals(
        groupID, norm_method, save_name, thresh, axis_group=axis_group)

    if axis_obj != 'False':
        ax = axis_obj

    ax.plot(wavelength, spectrum, color='black', lw=1, label='Spectrum')
    ax.fill_between(wavelength, spectrum - err_spectrum, spectrum + err_spectrum,
                    color='gray', alpha=0.5)
    ax.plot(wavelength[too_low_gals][plot_cut], spectrum[too_low_gals][plot_cut],
            color='red', lw=1, label=f'Too Few Galaxies ({cutoff})')
    ax.plot(wavelength[too_low_gals][not_plot_cut], spectrum[too_low_gals][not_plot_cut],
            color='red', lw=1)
    ax.plot(wave_bin, spec_bin, color='orange', lw=1, label='Median Binned')
    ax.plot(wavelength, continuum, color='blue', lw=1, label='continuum')
    if mask_negatives:
        ax.set_ylim(-1 * 10**-20, 1.01 * np.max(spectrum))
    else:
        ax.set_ylim(-1 * 10**-18, 1.01 * np.max(spectrum))
    ax.legend(loc=2, fontsize=axisfont - 3)
    ax.set_ylabel('F$_\lambda$', fontsize=axisfont)
    ax.tick_params(labelsize=ticksize, size=ticks)

    if axis_obj != 'False':
        return


    ax_Ha.fill_between(wavelength, spectrum - err_spectrum, spectrum + err_spectrum,
                       color='gray', alpha=0.5)
    ax_Ha.plot(wavelength, spectrum, color='black', lw=1)
    ax_contribute.plot(wavelength, n_galaxies, color='black',
                       lw=1, label='Number of Galaxies')
    ax_contribute.plot(wavelength[too_low_gals][plot_cut], n_galaxies[too_low_gals][plot_cut],
                       color='red', lw=1, label=f'Too Few Galaxies ({cutoff})')
    ax_contribute.plot(wavelength[too_low_gals][not_plot_cut], n_galaxies[too_low_gals][not_plot_cut],
                       color='red', lw=1)
    # ax_contribute.plot(wavelength, norm_value_summed, color='orange',
    #                    lw = 1, label = 'Normalized Value of Galaxies'
    ax_Ha.plot(wavelength, continuum, color='blue', lw=1)
    

    
    y_Ha_lim_max = np.max(spectrum[np.logical_and(
        wavelength > 6570, wavelength < 6800)])
    if mask_negatives:
        y_Ha_lim_min = 0.9 * np.min(spectrum[np.logical_and(
            wavelength > 6570, wavelength < 6800)])
    else:
        y_Ha_lim_min = np.min(spectrum[np.logical_and(
            wavelength > 6570, wavelength < 6800)])
    ax_Ha.set_ylim(y_Ha_lim_min, y_Ha_lim_max * 1.1)
    ax_Ha.set_xlim(6500, 6800)
    
    ax_contribute.legend(fontsize=axisfont - 3)

    rect = patches.Rectangle((6500, y_Ha_lim_min), 300, (y_Ha_lim_max -
                                                         y_Ha_lim_min), linewidth=1.5, edgecolor=zoom_box_color, facecolor='None')

    ax.add_patch(rect)
    # ax.set_xlim()
    ax_contribute.set_xlabel('Wavelength ($\\rm{\AA}$)', fontsize=axisfont)
    ax_contribute.set_ylabel('N', fontsize=axisfont)
    
    ax_contribute.tick_params(labelsize=ticksize, size=ticks)
    if axis_group > -1:
        # ax.text
        fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_spectra_images/{axis_group}_spectrum.pdf')
    else:
        save_dir = imd.composite_spec_dir + f'/{norm_method}_images'
        imd.check_and_make_dir(save_dir)
        fig.savefig(save_dir + f'/{groupID}_spectrum.pdf')
    plt.close()


def divz(X, Y):
    return X / np.where(Y, Y, Y + 1) * np.not_equal(Y, 0)


def stack_all_spectra(n_clusters, norm_method, re_observe=False, mask_negatives=False, ignore_low_spectra=False):
    """Runs the stack_spectra() function on every cluster

    Parameters:
    n_clusters (int): Number of clusters

    Returns:
    """
    for i in range(n_clusters):
        print(f'Stacking spectrum {i}...')
        stack_spectra(i, norm_method, re_observe=re_observe,
                      mask_negatives=mask_negatives, ignore_low_spectra=ignore_low_spectra)


def plot_all_spectra(n_clusters, norm_method, mask_negatives=False):
    """Runs the plot_spec() function on every cluster

    Parameters:
    n_clusters (int): Number of clusters
    norm_method (str): "cluster_norm" or "composite_sed_norm"


    Returns:
    """
    for i in range(n_clusters):
        plot_spec(i, norm_method, mask_negatives=mask_negatives)

def add_trues(df):
    """Given a dataframe of True and False, pad a True onto each group of Trues

    Parameters:
    df (pd.DataFrame): dataframe containing trues and falses

    Returns:
    df(pd.DataFrame): modified dataframe where every value adjacent to a group of Trues is now True
    """
    i = 0
    # Loop through the dataframe until second to last value
    while i < (len(df) - 1):
        current_bool = df.iloc[i]
        next_bool = df.iloc[i + 1]
        # If the values match, ignore it, if they don't, we need to change one
        # of them
        if current_bool != next_bool:
            # If the current value is True, then we want to add another True to
            # the next value, then skip over evaluating that one
            if current_bool == True:
                df.iloc[i + 1] = True
                # Skip the next value, since we just set it to true
                i = i + 1
            # If the current value is False, make it true
            if current_bool == False:
                df.iloc[i] = True
        # Move on to the next element
        i = i + 1
    return df


def clip_spectrum(spectrum_df, padded_mask, spectrum_wavelength, spectrum_flux_norm):
    """Sets the spectrum to zero in the regions where we need the spectrum clipped (between where padded_mask == True)

    Parameters:
    spectrum_df (pd.DataFrame): dataframe containing spectrum info from code
    padded_mask (pd.DataFrame): dataframe of same size that contains trues in clusters that should be masked out

    Returns:
    idx_clips (list): List of indices where the spectrum should be set to zero
    """
    indices = spectrum_df['rest_wavelength'][padded_mask].index
    idx_array = np.array(indices)
    idx_lists = [list(map(itemgetter(1), g)) for k, g in groupby(
        enumerate(idx_array), lambda x: x[0] - x[1])]
    mins_maxs = [(np.min(spectrum_df['rest_wavelength'].iloc[idx_list]), np.max(
        spectrum_df['rest_wavelength'].iloc[idx_list])) for idx_list in idx_lists]
    idx_clips = [np.logical_and(spectrum_wavelength > waves[
        0], spectrum_wavelength < waves[1]) for waves in mins_maxs]
    return idx_clips

def norm_axis_stack(ha_flux, z_spec):
    """To keep them all at the same area under the curve of the halpha line, we should normalize by flux and NOT luminosity
    """
    ha_luminosity = flux_to_luminosity(ha_flux, z_spec)
    norm_factor = 3e41 / ha_luminosity
    # norm_factor = 1e-17/ha_flux
    return norm_factor




def perform_stack(stack_type, interp_cluster_spectra_dfs, norm_factors):
    """ Does the final step of stacking the spectra
    
    Parameters:
    stack_type (str): Either "mean" or "median" to determine how to stack
    norm_interp_specs (list of dataframes): Spectra, errors, continuum, interpolated to the same wavelength range and normalized
    number_spects_by_wave (list): Number of galaixes that contribute to each point on the spectrum

    Returns:
    total_spec (array): Summed spectrum 
    total_cont (array): Summed continuum 
    total_errs (array): summed uncertainties 
    number_specs_by_wave (array): At each pixel, how manay galaxies contributed to it 
    norm_value_specs_by_wave (array): Same as above, but each galaxies is added as its normalization value rather than 1
    """
    # Pulls out just the flux values of each spectrum
    norm_interp_specs = [interp_cluster_spectra_dfs[i]['f_lambda_norm']
                         for i in range(len(interp_cluster_spectra_dfs))]
    # Pull out the errors
    norm_interp_errs = [interp_cluster_spectra_dfs[i]['err_f_lambda_norm']
                        for i in range(len(interp_cluster_spectra_dfs))]

    # Pulls out just the continuum values of each spectrum
    norm_interp_conts = [interp_cluster_spectra_dfs[i]['cont_norm']
                         for i in range(len(interp_cluster_spectra_dfs))]

    nonzero_idxs = [np.nonzero(np.array(norm_interp_specs[i]))
                    for i in range(len(norm_interp_specs))]

    sum_norms = [np.zeros(len(norm_interp_specs[i]))
                 for i in range(len(norm_interp_specs))]

    for i in range(len(norm_interp_specs)):
        sum_norms[i][nonzero_idxs[i]] = norm_factors[i]
    # This sum_norms variable is a list of arrays. Each array correponds to
    # one spectrum from one galaxy. In that array, every point for which this
    # galaxy has a non-masked flux, there is its normalization value. All
    # other points are zero

    # For each wavelength, counts the number of spectra that are nonzero
    number_specs_by_wave = np.count_nonzero(norm_interp_specs, axis=0)

    # This computes the value of the normalizations at each wavelength - NOT
    # USING
    norm_value_specs_by_wave = np.sum(sum_norms, axis=0)

    if stack_type == 'mean':
        # Add the spectrum
        summed_spec = np.sum(norm_interp_specs, axis=0)

        # Add the continuum
        summed_cont = np.sum(norm_interp_conts, axis=0)

        # Add the errors in quadrature:
        variances = [norm_interp_err**2 for norm_interp_err in norm_interp_errs]
        err_in_sum = np.sqrt(np.sum(variances, axis=0))

        # Have to use divz since lots of zeros
        # total_spec = divz(summed_spec, norm_value_specs_by_wave)
        total_spec = divz(summed_spec, number_specs_by_wave)
        total_cont = divz(summed_cont, number_specs_by_wave)
        total_errs = divz(err_in_sum, number_specs_by_wave)
    
    # Older slow code
    # if stack_type == 'median':
    #     # Set up the spectra to fill
    #     total_spec = []
    #     total_cont = []
    #     total_errs = []

    #     # Loop through the number of spectra that contribute to each point
    #     for i in range(len(number_specs_by_wave)):
            
    #         num_specs = number_specs_by_wave[i]
            
    #         # If there are no spectra contributing to that point, set it to zero
    #         if num_specs == 0:
    #             total_spec.append(0)
    #             total_cont.append(0)
    #             total_errs.append(0)
    #             continue
            
    #         # Otherwise, find the nonzero spectra and median them
    #         else:
    #             nonzero_spec_values = [norm_interp_specs[j][i] for j in range(len(norm_interp_specs)) if norm_interp_specs[j][i] != 0]
    #             nonzero_cont_values = [norm_interp_conts[j][i] for j in range(len(norm_interp_conts)) if norm_interp_conts[j][i] != 0]
    #             nonzero_err_values = [norm_interp_errs[j][i] for j in range(len(norm_interp_errs)) if norm_interp_errs[j][i] != 0]
    #             median_point = np.median(nonzero_spec_values)
    #             median_cont = np.median(nonzero_cont_values)
    #             variances = [nonzero_err_value**2 for nonzero_err_value in nonzero_err_values]
    #             err_in_sum = np.sqrt(np.sum(variances))
    #             err_in_mean = divz(err_in_sum, num_specs)

    #             ### Using 1.25*mean error, not sure if this is valid
    #             median_cont = np.median(nonzero_err_values)
    #             total_spec.append(median_point)
    #             total_cont.append(median_cont)
    #             total_errs.append(1.25*err_in_mean)


    if stack_type == 'median':
        # Turn the lists of norm_interp_specs into an array
        interp_spec_arr = np.array(norm_interp_specs)
        spec_zeros_mask = np.ma.masked_where(interp_spec_arr == 0, interp_spec_arr)
        total_spec = np.ma.median(spec_zeros_mask, axis=0).filled(0)

        # Repeat for cont:
        interp_cont_arr = np.array(norm_interp_conts)
        cont_zeros_mask = np.ma.masked_where(interp_cont_arr == 0, interp_cont_arr)
        total_cont = np.ma.median(cont_zeros_mask, axis=0).filled(0)

        # Find errs:
        interp_err_arr = np.array(norm_interp_errs)
        interp_err_arr_sums = np.sqrt(np.sum(interp_err_arr**2, axis=0))
        number_specs_by_wave_arr = np.array(number_specs_by_wave)
        # MAD - 1.25 * error in the mean
        total_errs = 1.25*(interp_err_arr_sums/number_specs_by_wave_arr)

    return total_spec, total_cont, total_errs, number_specs_by_wave, norm_value_specs_by_wave





def stack_axis_ratio(mass_width, split_width, starting_points, ratio_bins, save_name, split_by, stack_type, sfms_bins, use_whitaker_sfms, use_z_dependent_sfms, re_filter=False, bootstrap=0):
    """Stacks galaxies in groups by axis ratio

    old params: , l_mass_cutoff=0, l_ssfr_cutoff=0, l_mass_bins=0, l_ssfr_bins=0, scale_ha=0

    Parameters:
    mass_width (float): How big the mass bins are
    split_width (float): How wide the y-axis bins are
    starting_points (list of tuples): Where to start the ssfr bins in (mass, ssfr) coordinates
    ratio_bins (tuple): Where to cut the axis ratio bins e.g. (0.4, 0.7)
    split_by (str): What column to use for the pslitting - ssfr, or eq_width_ha
    use_ha_ssfr (int): Set to 1 to use the halpha calculated ssfrs
    stack_type (str): Either mean or median, what to use when stacking the galaxies
    sfms_bins (boolean): Set to True to overwrite the other bins and split along the sfms
    bootstrap (int): Set to a value for how many times to bootstrap

    Returns:
    """
    if re_filter == True:
        ar_df = read_interp_axis_ratio()
        # Filters the ar_df, see filers in the code
        ar_df = filter_ar_df(ar_df)
    else:
        ar_df = read_filtered_ar_df()

    ### REMOVE - JUST FOR LOW/HIGH REDSHIFT SEPARATION
    # ar_df = ar_df[ar_df['Z_MOSFIRE']>1.8]


    # Add a column for ssfr
    ar_df['log_ssfr'] = np.log10((ar_df['sfr'])/(10**ar_df['log_mass']))
    ar_df['log_halpha_ssfr'] = np.log10((ar_df['halpha_sfrs'])/(10**ar_df['log_mass']))
    ar_df['log_use_ssfr'] = np.log10((ar_df['use_sfr'])/(10**ar_df['log_mass']))
    ar_df['log_use_sfr'] = np.log10(ar_df['use_sfr'])


    # Save which method of ssfr is used
    if split_by=='log_ssfr':
        ar_df['split_for_stack'] = ['log_ssfr']*len(ar_df)
    if split_by=='log_use_ssfr':
        ar_df['split_for_stack'] = ['log_use_ssfr']*len(ar_df)
    if split_by=='log_use_sfr':
        ar_df['split_for_stack'] = ['log_use_sfr']*len(ar_df)
    if split_by=='log_halpha_ssfr':
        ar_df['split_for_stack'] = ['log_halpha_ssfr']*len(ar_df)
    if split_by=='eq_width_ha':
        ar_df['split_for_stack'] = ['eq_width_ha']*len(ar_df)

    split_name = ar_df.iloc[0]['split_for_stack']

    # Split into n_bins groups
    axis_group = 0
    cluster_name = save_name
    
    # Figure out what axis ratio bin (+1 is since the bins are dividers)
    if len(ratio_bins)+1 == 3:
        ar_df_low = ar_df[ar_df['use_ratio']<ratio_bins[0]]
        ar_df_mid = ar_df[np.logical_and(ar_df['use_ratio']>=ratio_bins[0],ar_df['use_ratio']<=ratio_bins[1])]
        ar_df_high = ar_df[ar_df['use_ratio']>ratio_bins[1]]
        
        ar_dfs = [ar_df_low, ar_df_mid, ar_df_high]
    
    if len(ratio_bins)+1 == 2:
        ar_df_low = ar_df[ar_df['use_ratio']<ratio_bins[0]]
        ar_df_high = ar_df[ar_df['use_ratio']>=ratio_bins[0]]
        
        ar_dfs = [ar_df_low, ar_df_high]

    if len(ratio_bins)+1 == 1:
        ar_dfs = [ar_df]


    for i in range(len(ar_dfs)):
        df = ar_dfs[i]
        dfs = []

        # If we're not using the sfms bins, use starting points to split
        if sfms_bins==False:
            
            for j in range(len(starting_points)):
                mass_start = starting_points[j][0]
                split_start = starting_points[j][1]
                mass_idx = np.logical_and(df['log_mass']>mass_start, df['log_mass']<=mass_start+mass_width)
                split_idx = np.logical_and(df[split_name]>split_start, df[split_name]<=split_start+split_width)                
                bin_idx = np.logical_and(mass_idx, split_idx)
                dfs.append(df[bin_idx])
        
        # If we are using sfms bins, see if the data falls below, above, or between the cuts
        if sfms_bins==True:
            # low_idx = df[split_by] < 1.07*df['log_mass']-9.83
            # high_idx = df[split_by] > 1.07*df['log_mass']-9.15
            # mid_idx = np.logical_and(np.logical_not(low_idx), np.logical_not(high_idx))
            if use_whitaker_sfms==True and use_z_dependent_sfms==True:
                low_z = df['Z_MOSFIRE'] < 1.8
                high_z = df['Z_MOSFIRE'] > 1.8
                low_idx = pd.concat([df[low_z][split_by] < whitaker_sfms(df[low_z]['log_mass'], a_lowz_fit), df[high_z][split_by] < whitaker_sfms(df[high_z]['log_mass'], a_highz_fit)])
                high_idx = pd.concat([df[low_z][split_by] >= whitaker_sfms(df[low_z]['log_mass'], a_lowz_fit), df[high_z][split_by] >= whitaker_sfms(df[high_z]['log_mass'], a_highz_fit)])
            elif use_whitaker_sfms==True:
                a = -24.0415
                b = 4.1693
                c = -0.1638
                low_idx = df[split_by] < a + b*df['log_mass'] + c*df['log_mass']**2
                high_idx = df[split_by] >= a + b*df['log_mass'] + c*df['log_mass']**2
            elif use_z_dependent_sfms==True:
                low_z = df['Z_MOSFIRE'] < 1.8
                high_z = df['Z_MOSFIRE'] > 1.8
                low_idx = pd.concat([df[low_z][split_by] < sfms_lowz_slope*df[low_z]['log_mass']+sfms_lowz_yint, df[high_z][split_by] < sfms_highz_slope*df[high_z]['log_mass']+sfms_highz_yint])
                high_idx = pd.concat([df[low_z][split_by] >= sfms_lowz_slope*df[low_z]['log_mass']+sfms_lowz_yint, df[high_z][split_by] >= sfms_highz_slope*df[high_z]['log_mass']+sfms_highz_yint])
            else:
                low_idx = df[split_by] < sfms_slope*df['log_mass']+sfms_yint
                high_idx = df[split_by] >= sfms_slope*df['log_mass']+sfms_yint

            low_mass = df['log_mass']<=10
            high_mass = df['log_mass']>10


            # REMOVE LATER ----- FOR METALLICITY TESTS
            # def fixed_metal_df(in_df, thresh_low, thresh_high):
            #     in_df = in_df[in_df['logoh_pp_n2']>thresh_low]
            #     in_df = in_df[in_df['logoh_pp_n2']<thresh_high]
            #     return in_df
            # df_low_low = df[np.logical_and(low_mass, low_idx)]
            # df_low_low = fixed_metal_df(df_low_low, 8.4, 8.6)
            # dfs.append(df_low_low)
            # df_low_high = df[np.logical_and(low_mass, high_idx)]
            # df_low_high = fixed_metal_df(df_low_high, 8.2, 8.4)
            # dfs.append(df_low_high)
            # df_high_low = df[np.logical_and(high_mass, low_idx)]
            # df_high_low = fixed_metal_df(df_high_low, 8.4, 8.6)
            # dfs.append(df_high_low)
            # df_high_high = df[np.logical_and(high_mass, high_idx)]
            # df_high_high = fixed_metal_df(df_high_high, 8.4, 8.6)
            # dfs.append(df_high_high)

            # REMOVE LATER ----- FOR HALF_LIGHT TESTS
            # def fixed_half_light_df(in_df, thresh_low, thresh_high):
            #     in_df = in_df[in_df['half_light']>thresh_low]
            #     in_df = in_df[in_df['half_light']<thresh_high]
            #     return in_df
            # df_low_low = df[np.logical_and(low_mass, low_idx)]
            # df_low_low = fixed_half_light_df(df_low_low, 0.1, 0.5)
            # dfs.append(df_low_low)
            # df_low_high = df[np.logical_and(low_mass, high_idx)]
            # df_low_high = fixed_half_light_df(df_low_high, 0.1, 0.5)
            # dfs.append(df_low_high)
            # df_high_low = df[np.logical_and(high_mass, low_idx)]
            # df_high_low = fixed_half_light_df(df_high_low, 0.25, 0.6)
            # dfs.append(df_high_low)
            # df_high_high = df[np.logical_and(high_mass, high_idx)]
            # df_high_high = fixed_half_light_df(df_high_high, 0.25, 0.6)
            # dfs.append(df_high_high)

            # ADD BACK ------ Normal method
            dfs.append(df[np.logical_and(low_mass, low_idx)])
            dfs.append(df[np.logical_and(low_mass, high_idx)])
            dfs.append(df[np.logical_and(high_mass, low_idx)])
            dfs.append(df[np.logical_and(high_mass, high_idx)])
            
            # dfs.append(df[np.logical_and(low_mass, mid_idx)])
            # dfs.append(df[np.logical_and(high_mass, mid_idx)])




        for df in dfs:
            # For each group, get a median and scatter of the axis ratios
            median_ratio = np.median(df['use_ratio'])
            scatter_ratio = np.std(df['use_ratio'])
            # print(f'Median: {median_ratio} \nScatter: {scatter_ratio}')
            # Save the dataframe for the group
            df.to_csv(
                (imd.axis_cluster_data_dir + f'/{cluster_name}/{cluster_name}_group_dfs/{axis_group}_df.csv'), index=False)
            # Within each group, start stacking the spectra
            
            stack_spectra(0, 'cluster_norm', axis_ratio_df=df,
                        axis_group=axis_group, save_name=cluster_name, stack_type=stack_type, bootstrap=bootstrap)
            axis_group = axis_group + 1
