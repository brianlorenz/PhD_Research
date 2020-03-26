# Creates the composite SED and filter curve

import sys
import os
import string
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
from read_data import mosdef_df
from mosdef_obj_data_funcs import read_sed
from clustering import cluster_dir
import matplotlib.pyplot as plt
from filter_response import lines, overview, get_index, get_filter_response
from scipy import interpolate


def get_all_composite_seds(num_clusters):
    """Generate and save all composite SEDs

    Parameters:
    num_clusters (int): number of clusers

    Returns:
    """
    for i in range(num_clusters):
        get_composite_sed(i)


def get_composite_sed(groupID):
    """Given the ID assigned to a group, create the composite SED for that group

    Parameters:
    groupID (int): number/name of the folder that contains the SEDs. Assigned in clustering.py

    Returns:
    """

    cluster_names = os.listdir(cluster_dir+'/'+str(groupID))
    # Splits into list of tuples: [(field, v4id), (field, v4id), (field, v4id), ...]
    fields_ids = [(line.split('_')[0], line.split('_')[1])
                  for line in cluster_names]

    count = 0
    for obj in fields_ids:
        field = obj[0]
        v4id = int(obj[1])
        sed = read_sed(field, v4id)
        # Add column to normalize or scale the SED - WHAT'S THE BEST WAY TO DO THIS?
        norm_factor = np.median(sed['f_lambda'])
        sed['f_lambda_norm'] = sed['f_lambda']/norm_factor
        sed['err_f_lambda_norm'] = sed['err_f_lambda']/norm_factor
        # Add a column to compute the rest_frame wavelength using the redshift
        sed['rest_wavelength'] = sed['peak_wavelength'] / \
            (1+sed['Z_MOSFIRE'])
        # If it's the first one, start the total_sed, otherwise append to it
        if count == 0:
            total_sed = sed
            count = 1
        else:
            total_sed = pd.concat([total_sed, sed])

    # Now we have total_seds, which is a huge dataframe that combines the seds of all of the individual objects

    # 1. Collect particular number of points based on number of stacked galaxies
    number_galaxies = len(cluster_names)

    total_sed = total_sed.sort_values('rest_wavelength')
    good_idx = np.logical_and(
        total_sed['f_lambda'] > -98, total_sed['err_f_lambda'] > 0)

    composite_sed_points = []
    composite_sed_err_ds = []
    composite_sed_err_us = []
    composite_sed_wavelengths = []
    composite_filters = []

    # Repeat the next few steps for every n points, where n is the number of galaxies.
    # i represents the current point
    i = 0
    while i < len(total_sed[good_idx]):
        # Check if there's space for 2 more sets of points
        if (i+2*number_galaxies) < len(total_sed[good_idx]):
            # If so, collect the next set
            selected_points = total_sed[good_idx].iloc[i:i+number_galaxies]
        else:
            # Otherwise, take all points until the end and end the loop after this cycle
            selected_points = total_sed[good_idx].iloc[i:]
            i = len(total_sed)

        # 2. Average the points into a single point, add the errors in quadrature, put at the center of the wavelength range
        # SHOULD THIS POINT BE THE AVERAGE OF THE POINTS OR THE MEDIAN OF THE BOOTSTRAPPED DISTRIBUTION???
        composite_sed_point = np.mean(selected_points['f_lambda_norm'])
        # print(composite_sed_point)
        composite_sed_points.append(composite_sed_point)
        # Bootstrap to get errors:
        composite_sed_err_d, composite_sed_err_u = get_composite_sed_errs(
            selected_points, composite_sed_point)
        composite_sed_err_ds.append(composite_sed_err_d)
        composite_sed_err_us.append(composite_sed_err_u)
        # Get the wavelength as the center of the wavelength range

        wavelength_min = np.min(
            selected_points['rest_wavelength'])
        wavelength_max = np.max(selected_points['rest_wavelength'])

        # HOW TO SET COMPOSITE POINT WAVELENGTH? Probably best not to center, but rather be dragged to average of the consituent points?
        # composite_sed_wave = (wavelength_max-wavelength_min)/2 + wavelength_min
        composite_sed_wave = np.mean(selected_points['rest_wavelength'])
        composite_sed_wavelengths.append(composite_sed_wave)

        composite_filter = get_composite_filter(selected_points, wavelength_min,
                                                wavelength_max, composite_sed_wave, composite_sed_point, composite_sed_err_d, composite_sed_err_u, groupID)
        composite_filters.append(composite_filter)

        # End of while loop
        i = i + number_galaxies

    composite_sed = pd.DataFrame(
        zip(composite_sed_wavelengths, composite_sed_points, composite_sed_err_ds, composite_sed_err_us), columns=['rest_wavelength', 'f_lambda', 'err_f_lambda_d', 'err_f_lambda_u'])
    vis_composite_sed(total_sed, composite_sed=composite_sed,
                      composite_filters=composite_filters, groupID=groupID)

    # Save the composite SED
    composite_sed.to_csv(
        f'/Users/galaxies-air/mosdef/composite_sed_csvs/{groupID}_sed.csv', index=False)

    # Copmosite filters already saved elsewhere
    return composite_sed, composite_filters

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
            selected_points['f_lambda_norm'], selected_points['err_f_lambda_norm'])
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
    filt_resolution = 0.2  # Angstrom

    # Number of points/filters that go into composite SED
    num_points = len(selected_points)

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
            (1+redshifts[i])
        # 5. For each point, add zeros at the start of filter and start of wavelegnth range so that interpolation outside of the filter gives 0
        zeros_append = pd.DataFrame([[-99, 0, wavelength_min], [-99, 0, wavelength_max]], columns=[
            'wavelength', 'transmission', 'rest_wavelength'])
        response_curves[i] = response_curves[i].append(zeros_append)

    # 6. Interpolate each filter curve at very high resolution at the same set of points
    interp_funcs = [interpolate.interp1d(
        response_curves[i]['rest_wavelength'], response_curves[i]['transmission']) for i in range(num_points)]
    # This is a list of all of the interpolated valeus for the curves in the REST frame
    interp_filt_values = [interp_funcs[i](
        filt_wavelength) for i in range(num_points)]

    # 7. At each high-res point, average the interpolated values. Return the averaged fitler curve
    filt_values = np.sum(interp_filt_values, axis=0)/num_points

    filt_response_df = pd.DataFrame(zip(filt_wavelength, filt_values), columns=[
                                    'rest_wavelength', 'transmission'])

    vis_composite_filt(selected_points, filt_wavelength, filt_values,
                       interp_filt_values, composite_sed_wave, composite_sed_point, composite_sed_err_d, composite_sed_err_u, groupID)

    # Save the filter using an int() of the composite sed wavelength
    save_dir = f'/Users/galaxies-air/mosdef/composite_sed_csvs/composite_filter_csvs/{groupID}_filter_csvs/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    filt_response_df.to_csv(save_dir+f'point_{int(composite_sed_wave)}.csv', index=False)

    return filt_response_df


def vis_composite_sed(total_sed, composite_sed=0, composite_filters=0, groupID=-99):
    axisfont = 14
    ticksize = 12
    ticks = 8
    titlefont = 24
    legendfont = 14
    textfont = 16

    good_idx = np.logical_and(
        total_sed['f_lambda'] > -98, total_sed['err_f_lambda'] > 0)

    fig, axarr = plt.subplots(2, 1, figsize=(
        8, 9), gridspec_kw={'height_ratios': [6, 1]})
    ax_sed = axarr[0]
    ax_filt = axarr[1]

    ax_sed.errorbar(total_sed[good_idx]['rest_wavelength'], total_sed[good_idx]
                    ['f_lambda_norm'], yerr=total_sed[good_idx]['err_f_lambda_norm'], ls='', marker='o', markersize=2, color='grey', zorder=1)

    ax_sed.errorbar(composite_sed['rest_wavelength'], composite_sed['f_lambda'],
                    yerr=[composite_sed['err_f_lambda_d'], composite_sed['err_f_lambda_u']], ls='', marker='o', markersize=4, color='black', zorder=2)
    for i in range(len(composite_filters)):
        ax_filt.plot(composite_filters[i]['rest_wavelength'],
                     composite_filters[i]['transmission'], color='black')

    for ax in axarr:
        ax.set_xlabel('Rest Wavelength ($\AA$)', fontsize=axisfont)
        ax.set_ylabel('Normalized Flux', fontsize=axisfont)
        ax.set_xscale('log')
    ax_sed.set_ylim(-0.2, 5)
    ax_filt.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    fig.savefig(f'/Users/galaxies-air/mosdef/Clustering/composite_seds/{groupID}_sed.pdf')
    plt.close()


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
        ax_sed.errorbar(selected_points.iloc[i]['rest_wavelength'], selected_points.iloc[i]['f_lambda_norm'], yerr=selected_points.iloc[i]['err_f_lambda_norm'],
                        ls='', marker='o', markersize=4, color=color)
        ax_filt.plot(filt_wavelength,
                     interp_filt_values[i], color=color)

    # Composite Filter
    ax_filt.plot(filt_wavelength, filt_values, lw=2,
                 color='black')

    for ax in axarr:
        ax.set_xlabel('Rest Wavelength ($\AA$)', fontsize=axisfont)
        ax.set_ylabel('Normalized Flux', fontsize=axisfont)
        ax.set_xscale('log')
    ax_sed.legend()
    ax_sed.set_ylim(-0.2, 5)
    ax_filt.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    save_dir = f'/Users/galaxies-air/mosdef/Clustering/composite_filter/{groupID}_filters/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    fig.savefig(save_dir+f'{int(composite_sed_wave)}.pdf')
    plt.close()
