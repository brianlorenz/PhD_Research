# Deals with the BPT of the seds and composites.

import numpy as np
from read_data import mosdef_df, read_file
from mosdef_obj_data_funcs import get_mosdef_obj
import matplotlib.pyplot as plt
from query_funcs import get_zobjs
import initialize_mosdef_dirs as imd
import cluster_data_funcs as cdf


def get_sfr(mosdef_objs, sfr_df):
    """Gets a dataframe of the masses and sfrs of galaxies in zobjs

    Parameters:
    zobjs (list): list of tuples of the form (field, v4id)
    sfr_df (pd.DataFrame): Dataframe of sfrs from mosdef_sfrs_latest.fits

    Returns:
    sfrs (list): List of sfrs in the same order as zobjs
    sfr_errs (list of tuples): Errors in the form: lower, upper
    """

    sfr_type = 'SFR_CORR'
    sfrs = []
    sfr_errs = []
    l_sfrs = []
    l_sfr_errs = []

    for mosdef_obj in mosdef_objs:
        field = mosdef_obj['FIELD_STR']
        mosdef_id = mosdef_obj['ID']
        sfr_obj = sfr_df[np.logical_and(
            sfr_df['FIELD_STR'] == field, sfr_df['ID'] == mosdef_id)]

        # Removes any rows where there are negative SFRs
        sfr_obj = sfr_obj[sfr_obj[sfr_type] > 0]

        # If now there are none left, set the SFR to zero
        if len(sfr_obj) < 1:
            sfr = -999
            sfr_err = (-999, -999)
            l_sfr = -999
            l_sfr_err = (-999, -999)
        elif len(sfr_obj) > 1:
            print('Multiple matches, what to do? Currently takes first instance')
            sfr = sfr_obj[sfr_type].iloc[0]
            sfr_l68 = sfr_obj[sfr_type + '_L68'].iloc[0]
            sfr_u68 = sfr_obj[sfr_type + '_U68'].iloc[0]
            sfr_err = (sfr - sfr_l68, sfr_u68 - sfr)

            l_sfr = np.log10(sfr)
            l_sfr_l68 = np.log10(sfr_l68)
            l_sfr_u68 = np.log10(sfr_u68)
            l_sfr_err = (l_sfr - l_sfr_l68, l_sfr_u68 - l_sfr)

            print(sfr_obj)
        else:
            sfr = sfr_obj[sfr_type].iloc[0]
            sfr_l68 = sfr_obj[sfr_type + '_L68'].iloc[0]
            sfr_u68 = sfr_obj[sfr_type + '_U68'].iloc[0]
            sfr_err = (sfr - sfr_l68, sfr_u68 - sfr)

            l_sfr = np.log10(sfr)
            l_sfr_l68 = np.log10(sfr_l68)
            l_sfr_u68 = np.log10(sfr_u68)
            l_sfr_err = (l_sfr - l_sfr_l68, l_sfr_u68 - l_sfr)
        sfrs.append(sfr)
        sfr_errs.append(sfr_err)
        l_sfrs.append(l_sfr)
        l_sfr_errs.append(l_sfr_err)

    return sfrs, sfr_errs, l_sfrs, l_sfr_errs


def plot_mass_sfr(zobjs, savename='None', axis_obj='False', composite_sfr_mass_point=[-47], all_sfrs_res='None'):
    """Plots the mass vs diagram for the objects in zobjs

    Parameters:
    zobjs (list): list of tuples of the form (field, v4id)
    savename (str): location with name ot save the file
    axis_obj (matplotlib_axis): Replace with an axis to plot on an existing axis
    composite_sfr_mass_point (): Set to the point if using a composite sed and you want to plot the point of that

    Returns:
    """

    axisfont = 14
    ticksize = 12
    ticks = 8

    sfr_df = read_sfr_df()
    if all_sfrs_res == 'None':
        all_sfrs_res = get_all_sfrs_masses(sfr_df)

    mosdef_objs = [get_mosdef_obj(zobj[0], zobj[1]) for zobj in zobjs]
    # l_sfrs = [mosdef_obj['LSFR'] for mosdef_obj in mosdef_objs]
    # l_sfr_errs = [(mosdef_obj['LSFR'] - mosdef_obj['L68_LSFR'],
    # mosdef_obj['U68_LSFR'] - mosdef_obj['LSFR']) for mosdef_obj in
    # mosdef_objs]
    l_masses = [mosdef_obj['LMASS'] for mosdef_obj in mosdef_objs]
    l_mass_errs = [(mosdef_obj['LMASS'] - mosdef_obj['L68_LMASS'],
                    mosdef_obj['U68_LMASS'] - mosdef_obj['LMASS']) for mosdef_obj in mosdef_objs]
    l_mass_errs = np.array(l_mass_errs).transpose()

    # Get the sfr values to plot for all object
    sfrs, sfr_errs, l_sfrs, l_sfr_errs = get_sfr(mosdef_objs, sfr_df)
    sfr_errs = np.array(sfr_errs).transpose()
    l_sfr_errs = np.array(l_sfr_errs).transpose()

    if axis_obj == 'False':
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        ax = axis_obj

    # Plot all galaxies iin grey in the background
    all_l_masses, all_l_mass_errs, all_l_sfrs, all_l_sfr_errs = all_sfrs_res
    ax.plot(all_l_masses, all_l_sfrs, marker='o', ms=2,
            color='grey', ls='None')

    # Plot the cluster galaxies in black
    ax.errorbar(l_masses, l_sfrs, xerr=l_mass_errs, yerr=l_sfr_errs,
                marker='o', color='black', ecolor='grey', ls='None')

    # If available, plot the composite in red
    if composite_sfr_mass_point[0] != -47:
        ax.plot(composite_sfr_mass_point[0], composite_sfr_mass_point[
                1], marker='x', color='red')

    # Plot lines of constant ssfr
    ssfrs = [0.1, 1, 10, 100]
    ssfr_l_masses = np.arange(8, 13, 1)
    label_locs = [11.6, 11.6, 10.7, 9.7]
    for i in range(len(ssfrs)):
        ssfr_l_sfrs = np.log10(10**ssfr_l_masses * ssfrs[i] / 10**9)
        label_loc = np.log10(10**label_locs[i] * ssfrs[i] / 10**9) - 0.1
        ax.plot(ssfr_l_masses, ssfr_l_sfrs, ls='--', color='orange')
        ax.text(label_locs[i], label_loc, f'{ssfrs[i]} Gyr$^{-1}$', rotation=50)

    ax.set_xlim(8, 12)
    ax.set_ylim(0, 3)

    if axis_obj == 'False':
        ax.set_xlabel('log(Stellar Mass) (M_sun)', fontsize=axisfont)
        ax.set_ylabel('log(SFR) (M_sun/yr)', fontsize=axisfont)
        ax.tick_params(labelsize=ticksize, size=ticks)

        if savename != 'None':
            fig.savefig(savename)

    plt.close('all')


def plot_mass_sfr_cluster(groupID, all_sfrs_res, axis_obj='False'):
    """Given one clustser, makes the sft/mass plot

    Parameters:
    groupID(int): Number of the cluster to plot
    axis_obj(matplotlib.ax): Axis to plot it on

    Returns:
    """
    cluster_names, fields_ids = cdf.get_cluster_fields_ids(groupID)
    fields_ids = [(obj[0], int(obj[1])) for obj in fields_ids]
    savename = imd.cluster_sfr_plots_dir + f'/{groupID}_mass_sfr.pdf'
    plot_mass_sfr(fields_ids, savename=savename,
                    axis_obj=axis_obj, composite_sfr_mass_point=[-47], all_sfrs_res=all_sfrs_res)


def plot_mass_sfr_clusters(n_clusters):
    """Makes the mass/sfr diagram for all clusters

    Parameters:
    n_clusters(int): Number of clusters

    Returns:
    """

    sfr_df = read_sfr_df()
    all_sfrs_res = get_all_sfrs_masses(sfr_df)
    for groupID in range(n_clusters):
        plot_mass_sfr(groupID, all_sfrs_res, axis_obj='False')



def get_all_sfrs_masses(sfr_df):
    """Gets all the masses and sfrs for all galaxies in our sample

    Parameters:

    Returns:
    all_l_masses (list): mass measurements from FAST
    all_l_mass_errs (2xn array): err on those measurements
    all_l_sfrs (list): sfrs from ha
    all_l_sfr_errs (2xn array): Errs on those

    """
    zobjs = get_zobjs()

    all_mosdef_objs = [get_mosdef_obj(zobj[0], zobj[1]) for zobj in zobjs]
    # l_sfrs = [mosdef_obj['LSFR'] for mosdef_obj in mosdef_objs]
    # l_sfr_errs = [(mosdef_obj['LSFR'] - mosdef_obj['L68_LSFR'],
    # mosdef_obj['U68_LSFR'] - mosdef_obj['LSFR']) for mosdef_obj in
    # mosdef_objs]
    all_l_masses = [mosdef_obj['LMASS'] for mosdef_obj in all_mosdef_objs]
    all_l_mass_errs = [(mosdef_obj['LMASS'] - mosdef_obj['L68_LMASS'],
                        mosdef_obj['U68_LMASS'] - mosdef_obj['LMASS']) for mosdef_obj in all_mosdef_objs]
    all_l_mass_errs = np.array(all_l_mass_errs).transpose()

    # Get the sfr values to plot for all object
    all_sfrs, all_sfr_errs, all_l_sfrs, all_l_sfr_errs = get_sfr(
        all_mosdef_objs, sfr_df)
    all_sfr_errs = np.array(all_sfr_errs).transpose()
    all_l_sfr_errs = np.array(all_l_sfr_errs).transpose()

    return all_l_masses, all_l_mass_errs, all_l_sfrs, all_l_sfr_errs


def read_sfr_df():
    """Reads and prepares the sfr_df

    Parameters:

    Returns:
    sfr_df(pd.DataFrame): dataframe of the fits file mosdef_sfrs_latest.fits

    """

    sfr_df = read_file(imd.loc_sfrs_latest)
    sfr_df['FIELD_STR'] = [sfr_df.iloc[i]['FIELD'].decode(
        "utf-8").rstrip() for i in range(len(sfr_df))]

    return sfr_df
