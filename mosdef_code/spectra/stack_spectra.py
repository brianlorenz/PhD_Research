# Codes for stacking MOSDEF spectra
# stack_spectra(groupID, norm_method) to stack within clusters

import sys
import os
import string
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
from read_data import mosdef_df
from mosdef_obj_data_funcs import read_sed, read_mock_sed, get_mosdef_obj, read_composite_sed, read_fast_continuum, setup_get_ssfr, merge_ar_ssfr
from filter_response import lines, overview, get_index, get_filter_response
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.integrate as integrate
from query_funcs import get_zobjs
import initialize_mosdef_dirs as imd
import cluster_data_funcs as cdf
from spectra_funcs import read_axis_ratio_spectrum, clip_skylines, check_line_coverage, get_spectra_files, median_bin_spec, read_spectrum, get_too_low_gals, norm_spec_sed, read_composite_spectrum, prepare_mock_observe_spectrum, mock_observe_spectrum
import matplotlib.patches as patches
from axis_ratio_funcs import read_interp_axis_ratio
from operator import itemgetter
from itertools import *


def stack_spectra(groupID, norm_method, re_observe=False, mask_negatives=False, ignore_low_spectra=False, axis_ratio_df=[], axis_group=0, save_name=''):
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

    # Now that we have the mosdef objs for each galaxy in the cluster, we need
    # to loop over each one
    interp_cluster_spectra_dfs = []
    norm_factors = []
    for mosdef_obj in mosdef_objs:
        # Get the redshift and normalization
        z_spec = mosdef_obj['Z_MOSFIRE']
        field = mosdef_obj['FIELD_STR']
        v4id = mosdef_obj['V4ID']
        norm_sed = read_sed(field, v4id, norm=True)
        print(f'Reading Spectra for {field} {v4id}, z={z_spec:.3f}')
        # Check to see if the galaxy includes all emission lines
        covered = check_line_coverage(mosdef_obj)
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
                print(f'Ratio = {ratio}')
                if ratio > 2 or ratio < 0:
                    print('Skipping')
                    continue

            # NORMALZE - HOW BEST TO DO THIS?
            # Original Method - using the computed norm_factors
            if norm_method == 'cluster_norm':
                norm_factor = np.median(norm_sed['norm_factor'])
            elif norm_method == 'composite_sed_norm':
                try:
                    norm_factor, spec_correlate, used_points = norm_spec_sed(
                        composite_sed, spectrum_df)
                except Exception as expt:
                    print('Could nor normalize')
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
            else:
                sys.exit(
                    'Select norm_method: "cluster_norm", "composite_sed_norm", ')

            # If stacking in axis ratio groups, do NOT normalize
            if axis_stack:
                norm_factor = 1.
            print(f'Norm factor: {norm_factor}')

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

            # Save the interpolated spectrum
            interp_spectrum_df = pd.DataFrame(zip(spectrum_wavelength, spectrum_flux_norm, spectrum_err_norm, cont_norm),
                                              columns=['rest_wavelength', 'f_lambda_norm', 'err_f_lambda_norm', 'cont_norm'])
            interp_cluster_spectra_dfs.append(interp_spectrum_df)
            norm_factors.append(norm_factor)

    if re_observe == True:
        observed_spec_df = pd.DataFrame(zip(spectrum_files_list, scaled_observed_fluxes, fraction_in_ranges, composite_waves, composite_fluxes), columns=[
            'filename', 'observed_flux', 'fraction_in_range', 'composite_wavelength', 'composite_flux'])
        observed_spec_df.to_csv(imd.cluster_dir + f'/composite_spectra/composite_filter/{groupID}_observed_specs.csv', index=False)
        print(f'Saved {groupID}_observed_specs.csv')
        return

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
    # galaxy has a non-masked flux, there is it's normalization value. All
    # other points are zero

    # For each wavelength, counts the number of spectra that are nonzero
    number_specs_by_wave = np.count_nonzero(norm_interp_specs, axis=0)

    # This computes the value of the normalizations at each wavelength - NOT
    # USING
    norm_value_specs_by_wave = np.sum(sum_norms, axis=0)

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
    # Now we have divided each point by the sum of the normalizations that
    # contributed to it.
    total_spec_df = pd.DataFrame(zip(spectrum_wavelength, total_spec, total_errs, total_cont, number_specs_by_wave, norm_value_specs_by_wave),
                                 columns=['wavelength', 'f_lambda', 'err_f_lambda', 'cont_f_lambda', 'n_galaxies', 'norm_value_summed'])

    if axis_stack:
        median_ratio = np.median(axis_ratio_df['use_ratio'])
        total_spec_df.to_csv(
            imd.cluster_dir + f'/composite_spectra/axis_stack{save_name}/{axis_group}_spectrum.csv', index=False)
        plot_spec(groupID, norm_method,
                  axis_group=axis_group, save_name=save_name)

    else:
        total_spec_df.to_csv(
            imd.cluster_dir + f'/composite_spectra/{norm_method}/{groupID}_spectrum.csv', index=False)

        plot_spec(groupID, norm_method)
    return


def plot_spec(groupID, norm_method, mask_negatives=False, thresh=0.1, axis_group=-1, save_name=''):
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
    titlefont = 24
    legendfont = 14
    textfont = 16

    if axis_group > -1:
        total_spec_df = read_axis_ratio_spectrum(
            axis_group, save_name=save_name)
    else:
        total_spec_df = read_composite_spectrum(groupID, norm_method)

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

    ax.plot(wavelength, spectrum, color='black', lw=1, label='Spectrum')
    ax.fill_between(wavelength, spectrum - err_spectrum, spectrum + err_spectrum,
                    color='gray', alpha=0.5)
    ax_Ha.fill_between(wavelength, spectrum - err_spectrum, spectrum + err_spectrum,
                       color='gray', alpha=0.5)
    ax.plot(wavelength[too_low_gals][plot_cut], spectrum[too_low_gals][plot_cut],
            color='red', lw=1, label=f'Too Few Galaxies ({cutoff})')
    ax.plot(wavelength[too_low_gals][not_plot_cut], spectrum[too_low_gals][not_plot_cut],
            color='red', lw=1)
    ax.plot(wave_bin, spec_bin, color='orange', lw=1, label='Median Binned')
    ax_Ha.plot(wavelength, spectrum, color='black', lw=1)
    ax_contribute.plot(wavelength, n_galaxies, color='black',
                       lw=1, label='Number of Galaxies')
    ax_contribute.plot(wavelength[too_low_gals][plot_cut], n_galaxies[too_low_gals][plot_cut],
                       color='red', lw=1, label=f'Too Few Galaxies ({cutoff})')
    ax_contribute.plot(wavelength[too_low_gals][not_plot_cut], n_galaxies[too_low_gals][not_plot_cut],
                       color='red', lw=1)
    # ax_contribute.plot(wavelength, norm_value_summed, color='orange',
    #                    lw = 1, label = 'Normalized Value of Galaxies')
    ax.plot(wavelength, continuum, color='blue', lw=1, label='continuum')
    ax_Ha.plot(wavelength, continuum, color='blue', lw=1)

    if mask_negatives:
        ax.set_ylim(-1 * 10**-20, 1.01 * np.max(spectrum))
    else:
        ax.set_ylim(-1 * 10**-18, 1.01 * np.max(spectrum))
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
    ax.legend(loc=2, fontsize=axisfont - 3)
    ax_contribute.legend(fontsize=axisfont - 3)

    rect = patches.Rectangle((6500, y_Ha_lim_min), 300, (y_Ha_lim_max -
                                                         y_Ha_lim_min), linewidth=1.5, edgecolor=zoom_box_color, facecolor='None')

    ax.add_patch(rect)
    # ax.set_xlim()
    ax_contribute.set_xlabel('Wavelength ($\\rm{\AA}$)', fontsize=axisfont)
    ax_contribute.set_ylabel('N', fontsize=axisfont)
    ax.set_ylabel('F$_\lambda$', fontsize=axisfont)
    ax.tick_params(labelsize=ticksize, size=ticks)
    ax_contribute.tick_params(labelsize=ticksize, size=ticks)
    if axis_group > -1:
        ax.text
        fig.savefig(imd.cluster_dir + f'/composite_spectra/axis_stack{save_name}/{axis_group}_spectrum.pdf')
    else:
        fig.savefig(imd.cluster_dir + f'/composite_spectra/{norm_method}/{groupID}_spectrum.pdf')
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
        print(f'Stacking spectrum {i}')
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


def stack_axis_ratio(n_bins=10, l_mass_cutoff=0, l_ssfr_cutoff=0):
    """Stacks galaxies in groups by axis ratio

    Parameters:
    n_bins (int): Number of bins to divide galaxies into
    l_mass_cutoff (int): Splits into 2 mass bins in each group - those above mass of cutoff, and those below
    l_mass_cutoff (int): Splits into 2 ssfr bins in each group - those above ssfr of cutoff, and those below

    Returns:
    """
    ar_df = read_interp_axis_ratio()

    # Remove objects with greater than 0.1 error
    ar_df = ar_df[ar_df['err_use_ratio'] < 0.1]

    # Add ssfrs if needed:
    if l_ssfr_cutoff != 0:
        ssfr_mosdef_merge_no_dups = setup_get_ssfr()
        ar_df = merge_ar_ssfr(ar_df, ssfr_mosdef_merge_no_dups)

    # Sort, so we can easily split by axis ratio
    ar_df_sorted = ar_df.sort_values('use_ratio')

    # Add masses
    l_masses = [get_mosdef_obj(ar_df_sorted.iloc[i]['field'], ar_df_sorted.iloc[i]['v4id'])[
        'LMASS'] for i in range(len(ar_df))]
    ar_df_sorted['LMASS'] = l_masses

    # Split into n_bins groups
    axis_group = 0
    ar_dfs = np.array_split(ar_df_sorted, n_bins)
    for i in range(len(ar_dfs)):
        df = ar_dfs[i]
        if l_mass_cutoff > 0:
            df_low = df[df['LMASS'] < l_mass_cutoff]
            df_high = df[df['LMASS'] >= l_mass_cutoff]
            dfs = [df_low, df_high]
            save_name = '_mass'
        if l_ssfr_cutoff != 0:
            # Remove anything without a measured ssfr
            df = df[df['SSFR'] > 0]
            df['LSSFR'] = np.log10(df['SSFR'])
            df_low = df[df['LSSFR'] < l_ssfr_cutoff]
            df_high = df[df['LSSFR'] >= l_ssfr_cutoff]
            dfs = [df_low, df_high]
            save_name = '_ssfr'
        else:
            dfs = [df]
            save_name = ''
        for df in dfs:
            # For each group, get a median and scatter of the axis ratios
            median_ratio = np.median(df['use_ratio'])
            scatter_ratio = np.std(df['use_ratio'])
            print(f'Median: {median_ratio} \nScatter: {scatter_ratio}')
            # Save the dataframe for the group
            df.to_csv(
                (imd.cluster_dir + f'/composite_spectra/axis_stack{save_name}/{axis_group}_df.csv'), index=False)
            # Within each group, start stacking the spectra
            stack_spectra(0, 'cluster_norm', axis_ratio_df=df,
                          axis_group=axis_group, save_name=save_name)
            axis_group = axis_group + 1


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
