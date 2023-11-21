# Deals with the UVJ of the seds and composites. Make sure to run
# observe_all_uvj to save a dataframe of all uvj values
# plot_all_uvj_clusters(n_clusters)
# plot_full_uvj()

import sys
import os
import string
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
from read_data import mosdef_df
from mosdef_obj_data_funcs import read_sed, read_mock_sed, get_mosdef_obj, read_composite_sed
from filter_response import lines, overview, get_index, get_filter_response
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.integrate as integrate
from query_funcs import get_zobjs
import initialize_mosdef_dirs as imd
import cluster_data_funcs as cdf
import matplotlib as mpl
from plot_vals import *
from generate_clusters import read_filtered_gal_df, read_removed_gal_df


def get_uvj(field, v4id):
    """Get the U-V and V-J for a given galaxy

    Parameters:
    field (str): field of the galaxy
    v4id (int): v4id from 3DHST

    Returns:
    uvj_tuple (tuple): tuple of the form (U-V, V-J) for the input object from mosdef
    """

    # Read the file
    uvj_df = ascii.read(imd.loc_uvj).to_pandas()

    # Get the object from mosdef_df, since we need id and not v4id
    mosdef_obj = get_mosdef_obj(field, v4id)

    # Get the input object
    obj = uvj_df[np.logical_and(
        uvj_df['field'] == field, uvj_df['id'] == mosdef_obj['ID'])]

    # Get the U-V and V-J for that object
    try:
        u_v = obj['u_v'].iloc[0]
        v_j = obj['v_j'].iloc[0]
        uvj_tuple = (u_v, v_j)
    except IndexError:
        sys.exit(f'Could not find object ({field}, {v4id}) in uvj_df')
    return uvj_tuple


def observe_uvj(sed, composite=True):
    """Measure the U-V and V-J flux of a composite SED

    Parameters:
    sed (pd.DataFrame): sed to observe, needs columns 'rest_wavelength' and 'f_lambda'
    composite (boolean): set to True if using composite SED, False if using one of the other seds. Need to compute rest wavelength in that case

    Returns:
    uvj_tuple (tuple): tuple of the form (U-V, V-J) for the input composite SED
    """

    if composite == False:
        sed['rest_wavelength'] = sed[
            'peak_wavelength'] / (1 + sed['Z_MOSFIRE'])
        good_idx = np.logical_and(
            sed['f_lambda'] > -98, sed['err_f_lambda'] > 0)
        sed = sed[good_idx]

    # Create an interpolation object. Use this with interp_sed(wavelength) to
    # get f_lambda at that wavelength
    interp_sed = interpolate.interp1d(sed['rest_wavelength'], sed['f_lambda'])
    max_interp = np.max(sed['rest_wavelength'])

    # Filters are U=153, V=155, J=161
    U_filt_num = 153
    V_filt_num = 155
    J_filt_num = 161

    # Fluxes are in f_nu NOT f_lambda
    U_flux_nu = observe_filt(interp_sed, U_filt_num, max_interp)
    V_flux_nu = observe_filt(interp_sed, V_filt_num, max_interp)
    J_flux_nu = observe_filt(interp_sed, J_filt_num, max_interp)
    print(U_flux_nu, V_flux_nu, J_flux_nu)

    U_V = -2.5 * np.log10(U_flux_nu / V_flux_nu)
    if U_flux_nu < 0 or V_flux_nu < 0:
        U_V = -99
    V_J = -2.5 * np.log10(V_flux_nu / J_flux_nu)
    if V_flux_nu < 0 or J_flux_nu < 0:
        V_J = -99

    uvj_tuple = (U_V, V_J)
    return uvj_tuple


def observe_all_uvj(n_clusters, individual_gals=False, composite_uvjs=True):
    '''Observes the UVJ flux for each galaxy SED and for all cluster SEDs, then saves them to a dataframe

    Parameters:
    n_groups (int): number of clusters
    individual_gals (boolean): set to True if you want to recalculate UVJ for all galaxies
    composite_uvjs (boolean): set to True if you want to recalcualte UVJ for all composite SEDs

    Returns:
    '''
    if individual_gals:
        zobjs = get_zobjs()
        uvs = []
        vjs = []
        fields = [field for field, v4id in zobjs]
        v4ids = [v4id for field, v4id in zobjs]
        for obj in zobjs:
            print(f'Measuring UVJ for {obj[0]}, {obj[1]}')
            # catches objects with v4id = -9999
            if obj[1] < 0:
                uvs.append(-99)
                vjs.append(-99)
                continue
            sed = read_sed(obj[0], obj[1])
            uvj = observe_uvj(sed, composite=False)
            uvs.append(uvj[0])
            vjs.append(uvj[1])
        galaxy_uvj_df = pd.DataFrame(zip(fields, v4ids, uvs, vjs), columns=[
            'field', 'v4id', 'U_V', 'V_J'])
        galaxy_uvj_df.to_csv(
            imd.uvj_dir + '/galaxy_uvjs.csv', index=False)

    if composite_uvjs:
        uvs = []
        vjs = []
        groupIDs = []
        for groupID in range(0, n_clusters):
            composite_sed = read_composite_sed(groupID)
            uvj = observe_uvj(composite_sed)
            uvs.append(uvj[0])
            vjs.append(uvj[1])
            groupIDs.append(groupID)
        composite_uvj_df = pd.DataFrame(zip(groupIDs, uvs, vjs), columns=[
            'groupID', 'U_V', 'V_J'])
        imd.check_and_make_dir(imd.composite_uvj_dir)
        composite_uvj_df.to_csv(
            imd.composite_uvj_dir + '/composite_uvjs.csv', index=False)


def observe_filt(interp_sed, filter_num, max_interp=99999999.):
    """given an SED filter interpolated, measure the value of the SED in that filter

    Parameters:
    interp_sed (scipy.interp1d): interp1d of the SED that you want to measure
    filter_num (int): Number of the filter to observe from in FILTER.RES.latest
    max_interp (float): Maxmum allowed value for interpolation. If below the filte,r returns -99

    Returns:
    flux_filter_nu (int): The photometric SED point for that filter - the observation (in frequency units)
    """
    filter_df = get_filter_response(filter_num)[1]

    interp_filt = interpolate.interp1d(
        filter_df['wavelength'], filter_df['transmission'])

    wavelength_min = np.min(filter_df['wavelength'])
    wavelength_max = np.max(filter_df['wavelength'])
    # Check to see if the value will be above the interpolation range
    if max_interp < wavelength_max:
        print(f'Value above interpolation range for filter {filter_num}')
        return -99
    numerator = integrate.quad(lambda wave: (1 / 3**18) * (wave * interp_sed(wave) *
                                                           interp_filt(wave)), wavelength_min, wavelength_max)[0]
    denominator = integrate.quad(lambda wave: (
        interp_filt(wave) / wave), wavelength_min, wavelength_max)[0]
    flux_filter_nu = numerator / denominator
    return flux_filter_nu


def plot_uvj_cluster(groupID, axis_obj='False'):
    """given a groupID, plot the UVJ diagram of the composite and all galaxies within cluster

    Parameters:
    groupID (int): ID of the cluster to plot
    axis_obj (matplotlib axis): If given an axis, don't make a new figure - just plot on the given axis

    Returns:
    """

    cluster_names, fields_ids = cdf.get_cluster_fields_ids(groupID)

    # UVJs of all galaxies
    galaxy_uvj_df = ascii.read(imd.uvj_dir + '/galaxy_uvjs.csv').to_pandas()
    # UVJs of all composite SEDs
    composite_uvj_df = ascii.read(
        imd.composite_uvj_dir + '/composite_uvjs.csv').to_pandas()

    # Get their uvj values
    u_v = []
    v_j = []
    for obj in fields_ids:
        field = obj[0]
        v4id = int(obj[1])
        idx = np.logical_and(galaxy_uvj_df['field'] ==
                             field, galaxy_uvj_df['v4id'] == v4id)
        u_v.append(galaxy_uvj_df[idx]['U_V'].iloc[0])
        v_j.append(galaxy_uvj_df[idx]['V_J'].iloc[0])
        # Catalog calculation
        #uvjs.append(get_uvj(field, v4id))
    #u_v = [i[0] for i in uvjs]
    #v_j = [i[1] for i in uvjs]

    # Get uvj value of composite sed
    #composite_sed = read_composite_sed(groupID)
    #uvj_composite = observe_uvj(composite_sed)

    uvj_composite = composite_uvj_df[composite_uvj_df['groupID'] == groupID]

    axisfont = 14
    ticksize = 12
    ticks = 8
    legendfont = 14

    if axis_obj == 'False':
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        ax = axis_obj

    setup_uvj_plot(ax, galaxy_uvj_df, composite_uvj_df, axis_obj=axis_obj)

    # Plots the one within the cluster in black
    ax.plot(v_j, u_v,
            ls='', marker='o', markersize=3.5, color='black', label='Cluster Galaxies')

    # Plot the composite SED as a red X
    ax.plot(uvj_composite['V_J'], uvj_composite['U_V'],
            ls='', marker='x', markersize=8, markeredgewidth=2, color='black', label='Composite SED')

    if axis_obj == 'False':
        ax.legend(fontsize=legendfont - 4)
        ax.tick_params(labelsize=ticksize, size=ticks)
        fig.savefig(imd.composite_uvj_dir + f'/cluster_uvjs/{groupID}_UVJ.pdf')
        plt.close()
    else:
        ax.legend()
        return


def plot_uvj_cluster_paper(groupID, axis_obj='False'):
    """Similar to above but cleans up presentation

    Parameters:
    groupID (int): ID of the cluster to plot
    axis_obj (matplotlib axis): If given an axis, don't make a new figure - just plot on the given axis

    Returns:
    """

    cluster_names, fields_ids = cdf.get_cluster_fields_ids(groupID)

    # UVJs of all galaxies
    galaxy_uvj_df = ascii.read(imd.uvj_dir + '/galaxy_uvjs.csv').to_pandas()
    # UVJs of all composite SEDs
    composite_uvj_df = ascii.read(
        imd.composite_uvj_dir + '/composite_uvjs.csv').to_pandas()

    # Get their uvj values
    u_v = []
    v_j = []
    for obj in fields_ids:
        field = obj[0]
        v4id = int(obj[1])
        idx = np.logical_and(galaxy_uvj_df['field'] ==
                             field, galaxy_uvj_df['v4id'] == v4id)
        u_v.append(galaxy_uvj_df[idx]['U_V'].iloc[0])
        v_j.append(galaxy_uvj_df[idx]['V_J'].iloc[0])
        # Catalog calculation
        #uvjs.append(get_uvj(field, v4id))
    #u_v = [i[0] for i in uvjs]
    #v_j = [i[1] for i in uvjs]

    # Get uvj value of composite sed
    #composite_sed = read_composite_sed(groupID)
    #uvj_composite = observe_uvj(composite_sed)

    uvj_composite = composite_uvj_df[composite_uvj_df['groupID'] == groupID]

    axisfont = 14
    ticksize = 12
    ticks = 8
    legendfont = 14

    if axis_obj == 'False':
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        ax = axis_obj

    setup_uvj_plot(ax, galaxy_uvj_df, composite_uvj_df, axis_obj=True)
    ax.set_xlabel('V-J', fontsize=14)
    ax.set_ylabel('U-V', fontsize=14)

    # Plots the one within the cluster in black
    ax.plot(v_j, u_v,
            ls='', marker='o', markersize=3.5, color='black', label='Cluster Galaxies')

    # Plot the composite SED as a red X
    ax.plot(uvj_composite['V_J'], uvj_composite['U_V'],
            ls='', marker='x', markersize=8, markeredgewidth=2, color='black', label='Composite SED')

    if axis_obj == 'False':
        ax.legend(fontsize=legendfont - 4)
        ax.tick_params(labelsize=ticksize, size=ticks)
        imd.check_and_make_dir(imd.composite_uvj_dir + f'/cluster_uvjs/')
        fig.savefig(imd.composite_uvj_dir + f'/cluster_uvjs/{groupID}_UVJ_paper.pdf')
        plt.close()
    else:
        ax.legend()
        return

def plot_all_uvj_clusters(n_clusters):
    """Makes UVJ diagrams for all of the clusters

    Parameters:
    n_clusters (int): Number of clusters

    Returns:
    """
    for i in range(n_clusters):
        print(f'Plotting Cluster {i}')
        plot_uvj_cluster(i)

def plot_all_uvj_clusters_paper(n_clusters):
    """Makes UVJ diagrams for all of the clusters

    Parameters:
    n_clusters (int): Number of clusters

    Returns:
    """
    for i in range(n_clusters):
        print(f'Plotting Cluster {i}')
        plot_uvj_cluster_paper(i)


def plot_full_uvj(n_clusters, color_type='None', include_unused_gals='No'):
    """Generate one overview UVJ diagram, with clusters marked by low-membership and labeled by number

    Parameters:
    n_clusters (int): Number of clusters
    color_type (str): Either set to 'balmer', 'ssfr', or 'metallicity' for what to make the coloarbar
    include_unused_gals (str): Set to 'No', 'Yes', or 'Only', to either exclude, include, or only show unsued galaxies 

    Returns:
    """

    # UVJs of all galaxies
    galaxy_uvj_df = ascii.read(imd.uvj_dir + '/galaxy_uvjs.csv').to_pandas()
    # UVJs of all composite SEDs
    composite_uvj_df = ascii.read(
        imd.composite_uvj_dir + '/composite_uvjs.csv').to_pandas()
    # Cluster summary
    cluster_summary_df = ascii.read(imd.loc_cluster_summary_df).to_pandas()

    ticksize = 12
    ticks = 8
    legendfont = 14

    add_str = ''

    fig, ax = plt.subplots(figsize=(8, 8))

    setup_uvj_plot(ax, galaxy_uvj_df, composite_uvj_df, include_unused_gals=include_unused_gals)

    # bad_clusters = cdf.find_bad_clusters(n_clusters)
    # bad_uvjs = composite_uvj_df.loc[bad_clusters]
    bad_uvjs = composite_uvj_df

    # ax.plot(bad_uvjs['V_J'], bad_uvjs['U_V'],
    #         ls='', marker='x', markersize=5, markeredgewidth=2, color='red', label='Bad Composite SEDs')

    if include_unused_gals!='Only':
        ax.plot(composite_uvj_df['V_J'], composite_uvj_df['U_V'],
                ls='', marker='x', markersize=10, markeredgewidth=2, color='black')

    


    for groupID in range(n_clusters):
        row = composite_uvj_df.iloc[groupID]
        if include_unused_gals!='Only':
            pass
            # ax.text(row['V_J'] - 0.02, row['U_V'] + 0.03, f'{groupID}', size=12, fontweight='bold', color='black')
        if color_type != 'None':
            cmap = mpl.cm.inferno
            cluster_row_idx = cluster_summary_df['groupID'] == groupID
            if color_type == 'balmer':
                norm = mpl.colors.Normalize(vmin=2, vmax=10) 
                cluster_colorval = cluster_summary_df[cluster_row_idx]['balmer_dec']
                cbar_label = 'Balmer Dec'
                add_str = '_balmer_color'
            elif color_type == 'ssfr':
                norm = mpl.colors.Normalize(vmin=-9.5, vmax=-8.25)
                cluster_colorval = cluster_summary_df[cluster_row_idx]['median_log_ssfr'] 
                cbar_label = 'log(sSFR)'
                add_str = '_ssfr_color'
            elif color_type == 'metallicity':
                norm = mpl.colors.Normalize(vmin=8.2, vmax=8.9)
                cluster_colorval = cluster_summary_df[cluster_row_idx]['O3N2_metallicity'] 
                cbar_label = 'O3N2 Metallicity'
                add_str = '_metal_color'
            else:
                sys.exit('Unknown color type. Use "balmer", "ssfr", "metallicity", or "None"')

            rgba = cmap(norm(cluster_colorval))
            ax.plot(row['V_J'], row['U_V'],
                ls='', marker='o', markersize=6, markeredgewidth=2, color=rgba)
            
        

    # for groupID in bad_uvjs['groupID']:
    #     ax.text(composite_uvj_df.iloc[groupID]['V_J'] - 0.02, composite_uvj_df.iloc[groupID]
    #             ['U_V'] + 0.03, f'{groupID}', size=12, fontweight='bold', color='red')

    # Plot the bad composite SEDs as a red X
    if color_type != 'None':
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, fontsize=full_page_axisfont)
        cbar.ax.tick_params(labelsize=full_page_axisfont-2)
    
    if include_unused_gals=='Only':
        add_str = '_removed_gals'

    # ax.legend(fontsize=full_page_axisfont - 4)
    ax.tick_params(labelsize=full_page_axisfont)
    ax.set_xlabel('V-J', fontsize=full_page_axisfont)
    ax.set_ylabel('U-V', fontsize=full_page_axisfont)
    plt.subplots_adjust(left=0.13, right=0.87, bottom=0.1, top=0.84)
    fig.savefig(imd.composite_uvj_dir + f'/Full_UVJ{add_str}.pdf')
    plt.close()


def setup_uvj_plot(ax, galaxy_uvj_df, composite_uvj_df, axis_obj='False', include_unused_gals='No', paper_fig=False):
    """Plots all background galaxies and clusters onto the UVJ diagram, as well as the lines

    Parameters:
    ax (matplotlib axis): matplotlib axis to plot on
    galaxy_uvj_df (pd.dataFrame): dataframe containing uvj values for all galaxies
    composite_uvj_df (pd.dataFrame): dataframe containing uvj values for all composites
    include_unused_gals (str): Set to 'No', 'Yes', or 'Only', to either exclude, include, or only show unsued galaxies 
    

    Returns:
    """


    # Filters down to only points used
    if include_unused_gals == 'No':
        used_gal_df = read_filtered_gal_df()
        galaxy_uvj_df = galaxy_uvj_df[galaxy_uvj_df['v4id'].isin(used_gal_df['v4id'])]
        galaxy_uvj_df = galaxy_uvj_df.drop_duplicates(subset='v4id', keep="last")
    if include_unused_gals == 'Only':
        removed_gal_df = read_removed_gal_df()
        galaxy_uvj_df = galaxy_uvj_df[galaxy_uvj_df['v4id'].isin(removed_gal_df['v4id'])]
        galaxy_uvj_df = galaxy_uvj_df.drop_duplicates(subset='v4id', keep="last")


    # Plots all galaxy UVJs in grey
    ax.plot(galaxy_uvj_df['V_J'], galaxy_uvj_df['U_V'],
            ls='', marker='o', markersize=grey_point_size, color=grey_point_color, label='All Galaxies')

    if axis_obj == 'False':
        if include_unused_gals!='Only':
            # Plot all composites as purple X
            if paper_fig == False:
                ax.plot(composite_uvj_df['V_J'], composite_uvj_df['U_V'],
                        ls='', marker='x', markersize=5, markeredgewidth=2, color='purple', label='All Composite SEDs')
    if paper_fig == True:
        for i in range(len(composite_uvj_df)):
            color = get_row_color(i)
            ax.plot(composite_uvj_df.iloc[i]['V_J'], composite_uvj_df.iloc[i]['U_V'],
                            ls='', marker='o', mew=paper_marker_edge_width, mec=paper_mec, markersize=get_row_size(i), color=color, label='All Composite SEDs')

    # UVJ diagram lines
    ax.plot((-100, 0.69), (1.3, 1.3), color='black')
    ax.plot((1.5, 1.5), (2.01, 100), color='black')
    xline = np.arange(0.69, 1.5, 0.001)
    yline = xline * 0.88 + 0.69
    ax.plot(xline, yline, color='black')

    if axis_obj == 'False':
        ax.set_xlabel('V-J', fontsize=14)
        ax.set_ylabel('U-V', fontsize=14)

    ax.set_xlim(-0.5, 2)
    ax.set_ylim(0, 2.5)

# plot_all_uvj_clusters_paper(23)
# plot_all_uvj_clusters(23)
# observe_all_uvj(23, individual_gals=False, composite_uvjs=True)
# plot_full_uvj(20, include_unused_gals='No')
# plot_full_uvj(23, include_unused_gals='Only')
# plot_full_uvj(23, color_type='balmer')
# plot_full_uvj(23, color_type='ssfr')
# plot_full_uvj(23, color_type='metallicity')