# Creates the composite SED and filter curve for axis ratio
# get_composite_sed(#), or get_all_composite_seds(#)
# If you don't want to plot filters (adds and extra ~2min per SED to do
# so), then add run_fitlers=False as an option

import sys
import os
import numpy as np
from numpy.lib.npyio import save
import pandas as pd
from astropy.io import ascii
from read_data import mosdef_df
from mosdef_obj_data_funcs import read_sed, read_mock_sed
from cross_correlate import get_cross_cor
import matplotlib.pyplot as plt
from filter_response import lines, overview, get_index, get_filter_response
from scipy import interpolate
from mosdef_obj_data_funcs import read_composite_sed
import initialize_mosdef_dirs as imd


def get_all_composite_sed_axis_ratio(num_clusters, save_name, run_filters=False):
    """Generate and save all composite SEDs

    Parameters:
    num_clusters (int): number of clusers
    run_filters (boolean): set to True if you want to compute and plot the filter curves for each composite

    Returns:
    """
    for i in range(num_clusters):
        print(f'Getting composite sed for group {i}...')
        get_composite_sed_axis_group(i, save_name, run_filters=run_filters)



def get_composite_sed_axis_group(axis_group, save_name, run_filters=False):
    """Given the ID assigned to a group, create the composite SED for that group

    Parameters:
    
    Returns:
    """
    ar_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_group_dfs/{axis_group}_df.csv').to_pandas()
    
    count = 0
    for i in range(len(ar_df)):
        field = ar_df.iloc[i]['field']
        v4id = ar_df.iloc[i]['v4id']

        sed = read_sed(field, v4id, norm=False)
        sed['rest_wavelength'] = sed['peak_wavelength'] / (1 + sed['Z_MOSFIRE'])
        # If it's the first one, start the total_sed, otherwise append to it
        if count == 0:
            total_sed = sed
            count = 1
        else:
            total_sed = pd.concat([total_sed, sed])

    # Now we have total_seds, which is a huge dataframe that combines the seds
    # of all of the individual objects

    # 1. Collect particular number of points based on number of stacked
    # galaxies
    number_galaxies = len(ar_df)

    total_sed = total_sed.sort_values('rest_wavelength')
    good_idx = get_good_idx(total_sed)
    composite_sed_points = []
    composite_sed_err_ds = []
    composite_sed_err_us = []
    composite_sed_wavelengths = []
    composite_filters = []
    std_scatter = []

    # Repeat the next few steps for every n points, where n is the number of galaxies.
    # i represents the current point
    i = 0
    step_size = np.max([int(number_galaxies / 3), 1])
    while i < len(total_sed[good_idx]):
        # Check if there's space for 2 more sets of points
        if (i + 2 * step_size) < len(total_sed[good_idx]):
            # If so, collect the next set
            selected_points = total_sed[good_idx].iloc[i:i + step_size]
        else:
            # Otherwise, take all points until the end and end the loop after
            # this cycle
            selected_points = total_sed[good_idx].iloc[i:]
            i = len(total_sed)

        # Get the scatter of the selected points, will be used for plotting
        std_scatter.append(np.percentile(
            selected_points['f_lambda'], [16, 84]))

        # 2. Average the points into a single point, add the errors in quadrature, put at the center of the wavelength range
        # SHOULD THIS POINT BE THE AVERAGE OF THE POINTS OR THE MEDIAN OF THE
        # BOOTSTRAPPED DISTRIBUTION???
        composite_sed_point = np.mean(selected_points['f_lambda'])
        # print(composite_sed_point)
        composite_sed_points.append(composite_sed_point)
        # Bootstrap to get errors:
        composite_sed_err_d, composite_sed_err_u = get_composite_sed_errs(
            selected_points, composite_sed_point)
        composite_sed_err_ds.append(composite_sed_err_d)
        composite_sed_err_us.append(composite_sed_err_u)

        wavelength_min = np.min(
            selected_points['rest_wavelength'])
        wavelength_max = np.max(selected_points['rest_wavelength'])

        composite_sed_wave = np.mean(selected_points['rest_wavelength'])
        composite_sed_wavelengths.append(composite_sed_wave)

        if run_filters:
            print('Check code here before running, dont want to overwrite anything')
            # composite_filter = get_composite_filter(selected_points, wavelength_min,
            #                                         wavelength_max, composite_sed_wave, composite_sed_point, composite_sed_err_d, composite_sed_err_u, groupID)
            # composite_filters.append(composite_filter)
        else:
            composite_filters = 0

        # End of while loop
        # Increment value, make sure to have a failsafe in case it is zero
        i = i + step_size

    composite_sed = pd.DataFrame(
        zip(composite_sed_wavelengths, composite_sed_points, composite_sed_err_ds, composite_sed_err_us), columns=['rest_wavelength', 'f_lambda', 'err_f_lambda_d', 'err_f_lambda_u'])
    vis_composite_sed_axis_ratio(total_sed, composite_sed=composite_sed,
                      composite_filters=composite_filters, axis_group=axis_group, std_scatter=std_scatter, run_filters=run_filters, save_name=save_name)

    # Save the composite SED
    composite_sed.to_csv(
        imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_composite_seds/{axis_group}_composite_sed.csv', index=False)

    # Save the total SED
    total_sed.to_csv(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_composite_seds/{axis_group}_total_sed.csv', index=False)
    

    # Copmosite filters already saved elsewhere
    return

# dataframe format: wavelength, flux, errorflux, redshift, filter_num
# 1. Collect particular number of points based on number of stacked galaxies
# 2. Average the points into a single point, add the errors in quadrature, put at the center of the wavelength range
# 3. Define a filter from the start of the range to the end
# 4. For each point, de-redshift their filter
# 5. For each point, add zeros at the start of filter and start of wavelegnth range so that interpolation outside of the filter gives 0
# 6. Interpolate each filter curve at very high resolution at the same set of points
# 7. At each high-res point, average the interpolated values. Return the averaged fitler curve
# 8. Visualize all of the curves on a plot


def get_composite_sed_errs(selected_points, composite_sed_point):
    """Use bootstrapping to get the errors on a composite SED point

    Parameters:
    selected_points (pd.DataFrame): from get_composite_sed() - these are the points to bootstrap over

    Returns:
    err (array): error in the form (-error, +error)
    """
    means = []
    for i in range(200):
        random_points = np.random.normal(
            selected_points['f_lambda'], selected_points['err_f_lambda'])
        means.append(np.mean(random_points))
    stddevs = np.percentile(means, [15.7, 84.3])
    err_d = np.abs(composite_sed_point - stddevs[0])
    err_u = np.abs(composite_sed_point - stddevs[1])
    return err_d, err_u


def get_composite_filter(selected_points, wavelength_min, wavelength_max, composite_sed_wave, composite_sed_point, composite_sed_err_d, composite_sed_err_u, groupID):
    """Given a point for the composite SED, get teh corresponding filter curve

    Parameters:
    selected_points (pd.DataFrame): from get_composite_sed() - these are the points going intot he composite
    wavelength_min (float): minimum wavelength, start of filter
    wavelength_max (float): maximum wavelength, end of filter
    composite_sed_wave (float): wavelength of composite SED point, used for saving
    composite_sed_point (float): value of composite SED point, just for plotting
    composite_sed_yerr_d (float): lower error on composite SED, just for plotting
    composite_sed_yerr_u (float): upper error on composite SED, just for plotting
    groupID (int): Clustering group, used for saving filter


    Returns:
    filt_response_df (pd.DataFrame): dataframe containing 'rest_wavelength' and 'transmission' of the composite filter
    """
    filt_resolution = 0.5  # Angstrom

    print(f'    Generating filter at point {int(composite_sed_wave)}...')

    # Number of points/filters that go into composite SED
    num_points = len(selected_points)

    wavelength_min = 800
    wavelength_max = 40000

    # 3. Define a filter from the start of the range to the end
    filt_wavelength = np.arange(
        wavelength_min, wavelength_max, filt_resolution)

    # Get the response curve of the filter corresponding to the points
    filt_nums = [selected_points.iloc[i]['filter_num']
                 for i in range(num_points)]
    redshifts = [selected_points.iloc[i]['Z_MOSFIRE']
                 for i in range(num_points)]
    response_curves = [get_filter_response(num)[1] for num in filt_nums]

    # 4. For each point, de-redshift their filter
    for i in range(num_points):
        response_curves[i]['rest_wavelength'] = response_curves[i]['wavelength'] / \
            (1 + redshifts[i])
        # 5. For each point, add zeros at the start of filter and start of
        # wavelegnth range so that interpolation outside of the filter gives 0
        zeros_append = pd.DataFrame([[-99, 0, wavelength_min], [-99, 0, wavelength_max]], columns=[
            'wavelength', 'transmission', 'rest_wavelength'])
        response_curves[i] = response_curves[i].append(zeros_append)
        # Normalize the filter curve
        response_curves[i]['transmission'] = response_curves[i][
            'transmission'] / np.max(response_curves[i]['transmission'])

    # 6. Interpolate each filter curve at very high resolution at the same set
    # of points
    interp_funcs = [interpolate.interp1d(
        response_curves[i]['rest_wavelength'], response_curves[i]['transmission']) for i in range(num_points)]
    # This is a list of all of the interpolated valeus for the curves in the
    # REST frame
    interp_filt_values = [interp_funcs[i](
        filt_wavelength) for i in range(num_points)]

    # 7. At each high-res point, average the interpolated values. Return the
    # averaged fitler curve
    filt_values = np.sum(interp_filt_values, axis=0) / num_points
    # Normalize the averaged filter curve
    filt_values = filt_values / np.max(filt_values)

    filt_response_df = pd.DataFrame(zip(filt_wavelength, filt_values), columns=[
                                    'rest_wavelength', 'transmission'])

    vis_composite_filt(selected_points, filt_wavelength, filt_values,
                       interp_filt_values, composite_sed_wave, composite_sed_point, composite_sed_err_d, composite_sed_err_u, groupID)
    # Save the filter using an int() of the composite sed wavelength
    save_dir = imd.composite_filter_csvs_dir + f'/{groupID}_filter_csvs/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    filt_response_df.to_csv(save_dir + f'point_{int(composite_sed_wave)}.csv', index=False)
    return filt_response_df


def vis_composite_sed_axis_ratio(total_sed, composite_sed=0, composite_filters=0, axis_group=-99, std_scatter=0, run_filters=True, axis_obj='False', save_name='0'):
    """
    If you set an axis obj, it will overwrite the others, and make sure to set a groupID
    """
    axisfont = 14
    ticksize = 12
    ticks = 8
    titlefont = 24
    legendfont = 14
    textfont = 16



    if axis_obj == 'False':
        if run_filters:
            fig, axarr = plt.subplots(2, 1, figsize=(
                8, 9), gridspec_kw={'height_ratios': [6, 1]})
            ax_sed = axarr[0]
            ax_filt = axarr[1]
        else:
            fig, ax_sed = plt.subplots(1, 1, figsize=(
                8, 8))
            axarr = [ax_sed]
    else:
        ax_sed = axis_obj
        axarr = [ax_sed]
        total_sed = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_composite_seds/{axis_group}_total_sed.csv').to_pandas()
        composite_sed = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_composite_seds/{axis_group}_composite_sed.csv').to_pandas()


    good_idx = get_good_idx(total_sed)

    

    plt.set_cmap('plasma')  # coolwarm
    ax_sed.scatter(total_sed[good_idx]['rest_wavelength'], total_sed[good_idx]
                   ['f_lambda'], s=2, c=total_sed[good_idx]['v4id'], zorder=1)

    ax_sed.errorbar(composite_sed['rest_wavelength'], composite_sed['f_lambda'],
                    yerr=[composite_sed['err_f_lambda_d'], composite_sed['err_f_lambda_u']], ls='', marker='o', markersize=4, color='black', zorder=2)

    # Parse the scattter into 16th and 84th percetile arrays
    if axis_obj == 'False':
        scatter_16 = [i for i, j in std_scatter]
        scatter_84 = [j for i, j in std_scatter]
        ax_sed.plot(composite_sed['rest_wavelength'], scatter_16,
                    ls='-', marker='o', markersize=2, color='dimgrey', alpha=0.9, zorder=3)
        ax_sed.plot(composite_sed['rest_wavelength'], scatter_84,
                    ls='-', marker='o', markersize=2, color='dimgrey', alpha=0.9, zorder=3)

    if run_filters:
        for i in range(len(composite_filters)):
            ax_filt.plot(composite_filters[i]['rest_wavelength'],
                         composite_filters[i]['transmission'], color='black')

    for ax in axarr:
        ax.set_xlabel('Rest Wavelength ($\AA$)', fontsize=axisfont)
        ax.set_ylabel('Normalized Flux', fontsize=axisfont)
        ax.set_xscale('log')

    ax_sed.set_ylim(0, 1.2 * np.max(composite_sed['f_lambda']))
    ax_sed.set_xlim(800, 45000)
    # ax_filt.set_ylim(-0.05, 1.05)

    if run_filters:
        filt_dir = ''
        ax_filt.set_xlim(ax_sed.get_xlim())
    else:
        filt_dir = 'composite_seds_nofilt/'

    if axis_obj == 'False':
        plt.tight_layout()
        fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_composite_images/{axis_group}_sed.pdf')
        plt.close()
    else:
        return


def vis_composite_filt(selected_points, filt_wavelength, filt_values, interp_filt_values, composite_sed_wave, composite_sed_point, composite_sed_err_d, composite_sed_err_u, groupID):
    axisfont = 14
    ticksize = 12
    ticks = 8
    titlefont = 24
    legendfont = 14
    textfont = 16

    fig, axarr = plt.subplots(2, 1, figsize=(
        8, 9), gridspec_kw={'height_ratios': [3, 1]})
    ax_sed = axarr[0]
    ax_filt = axarr[1]

    composite_sed_point_err = np.transpose(
        [[composite_sed_err_d, composite_sed_err_u]])

    ax_sed.errorbar(composite_sed_wave, composite_sed_point, yerr=composite_sed_point_err,
                    ls='', marker='o', markersize=6, color='black', label='Composite Point')

    colors = iter(plt.cm.viridis(np.linspace(0, 1, len(interp_filt_values))))
    for i in range(len(interp_filt_values)):
        color = next(colors)
        ax_sed.errorbar(selected_points.iloc[i]['rest_wavelength'], selected_points.iloc[i]['f_lambda'], yerr=selected_points.iloc[i]['err_f_lambda'],
                        ls='', marker='o', markersize=4, color=color)
        ax_filt.plot(filt_wavelength,
                     interp_filt_values[i], color=color)

    # Composite Filter
    ax_filt.plot(filt_wavelength, filt_values, lw=2,
                 color='black')

    nonzero_filts = [i for i, e in enumerate(filt_values) if e != 0]
    filt_minwave = filt_wavelength[nonzero_filts[0]]
    filt_maxwave = filt_wavelength[nonzero_filts[-1]]

    for ax in axarr:
        ax.set_xlabel('Rest Wavelength ($\AA$)', fontsize=axisfont)
        ax.set_ylabel('Normalized Flux', fontsize=axisfont)
        ax.set_xscale('log')
    ax_sed.legend()
    # ax_filt.set_xlim(ax_sed.get_xlim())
    ax_filt.set_xlim(filt_minwave, filt_maxwave)
    # ax_sed.set_ylim(-0.2, 5)
    # ax_filt.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    save_dir = imd.composite_filter_images_dir + f'/{groupID}_filters/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    fig.savefig(save_dir + f'{int(composite_sed_wave)}.pdf')
    plt.close()


def get_good_idx(sed):
    """Given an SED, get the values that are logical data (flux not -99, nonzero error)

    Parameters:
    sed (pd.DataFrame): the sed to get the good indexies for

    Returns:
    good_idx (list): dataframe containing True/False for each element, used to find data points that are usable
    """
    good_idx = np.logical_and(sed['f_lambda'] > -98, sed['err_f_lambda'] > 0)
    return good_idx


get_all_composite_sed_axis_ratio(12, 'mass_ssfr')
