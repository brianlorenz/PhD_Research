# Deals with the BPT of the seds and composites.

import numpy as np
import pandas as pd
from read_data import mosdef_df
from emission_measurements import read_emission_df, get_emission_measurements
import matplotlib.pyplot as plt
import initialize_mosdef_dirs as imd
import cluster_data_funcs as cdf
from astropy.io import ascii
import matplotlib as mpl
from axis_ratio_funcs import read_filtered_ar_df, read_interp_axis_ratio


def get_bpt_coords(gal_df):
    """Gets the row(s) corresponding to one object

    Parameters:
    gal_df (pd.DataFrame): Dataframe containting all objects

    Returns:
    bpt_df (pd.DataFrame): Dataframe of values to plot on the bpt diagram
    """

    # Will be used to store results
    fields = []
    v4ids = []
    log_NII_Has = []
    log_NII_Ha_errs = []
    log_OIII_Hbs = []
    log_OIII_Hb_errs = []

    # Re-name them
    Ha_flux, Ha_err = gal_df['ha_flux'], gal_df['err_ha_flux']
    NII_flux, NII_err = gal_df['nii_6585_flux'], gal_df['err_nii_6585_flux']
    Hb_flux, Hb_err = gal_df['hb_flux'], gal_df['err_hb_flux']
    OIII_flux, OIII_err = gal_df[
        'oiii_5008_flux'], gal_df['err_oiii_5008_flux']
    NII_Ha, NII_Ha_err = gal_df['nii_ha'], gal_df['err_nii_ha']


    # Calculate ratios and uncertainties
    # log_NII_Ha, log_NII_Ha_err = calc_log_ratio(
    #     NII_flux, NII_err, Ha_flux, Ha_err)
    log_NII_Ha = np.log10(NII_Ha)
    log_NII_Ha_err = 0.434 * (NII_Ha_err / NII_Ha)
    NII_Ha_err = gal_df['nii_ha'], gal_df['err_nii_ha']
    log_OIII_Hb, log_OIII_Hb_err = calc_log_ratio(
        OIII_flux, OIII_err, Hb_flux, Hb_err)

    gal_df['log_NII_Ha'] = log_NII_Ha
    gal_df['log_NII_Ha_err'] = log_NII_Ha_err
    gal_df['log_OIII_Hb'] = log_OIII_Hb
    gal_df['log_OIII_Hb_err'] = log_OIII_Hb_err

    return gal_df


def calc_log_ratio(top_flux, top_err, bot_flux, bot_err):
    """Calculates np.log10(top/bot) and its uncertainty

    Parameters:
    Fluxes an errors for each of the lines

    Returns:
    log_ratio (float): np.log10(top/bot)
    log_ratio_err (float): uncertainty in np.log10(top/bot)
    """
    log_ratio = np.log10(top_flux / bot_flux)
    log_ratio_err = (1 / np.log(10)) * (bot_flux / top_flux) * np.sqrt(
        ((1 / bot_flux) * top_err)**2 + ((-top_flux / (bot_flux**2)) * bot_err)**2)
    return log_ratio, log_ratio_err

def plot_bpt_all_composites():
    fig, ax = plt.subplots(figsize=(8,7))
    clusters_summary_df = ascii.read(imd.loc_cluster_summary_df).to_pandas()
    ### BPT Plot --------------------------------------------
    xrange = (-2, 1)
    yrange = (-1.2, 1.5)
    
   
    plot_bpt(axis_obj=ax, use_other_df=0, add_background=True, skip_gals=True)

    # ax.plot(group_df['log_mass'], group_df['log_use_sfr'], marker='o', color='black', ls='None')

    # Add the median of the cluster
    for i in range(len(clusters_summary_df)):
        groupID = clusters_summary_df['groupID'].iloc[i]
        log_N2_Ha_group = clusters_summary_df['log_N2_Ha'].iloc[i]
        log_O3_Hb_group = clusters_summary_df['log_O3_Hb'].iloc[i]
        
        log_N2_Ha_group_errs = (clusters_summary_df['err_log_N2_Ha_low'].iloc[i], clusters_summary_df['err_log_N2_Ha_high'].iloc[i])
        log_O3_Hb_group_errs = (clusters_summary_df['err_log_O3_Hb_low'].iloc[i], clusters_summary_df['err_log_O3_Hb_high'].iloc[i])
        
        ax.plot(log_N2_Ha_group, log_O3_Hb_group, marker='x', color='red', markersize=10, mew=3, ls='None', zorder=10000, label='Composite')
        ax.text(log_N2_Ha_group - 0.02, log_O3_Hb_group + 0.03, f'{groupID}', size=12, fontweight='bold', color='black')
    ax.set_xlabel('log(N[II] 6583 / H$\\alpha$)', fontsize=14)
    ax.set_ylabel('log(O[III] 5007 / H$\\beta$)', fontsize=14)
    ax.tick_params(labelsize=14, size=14)
    fig.savefig(imd.cluster_dir+'/cluster_stats/all_groups_bpt.pdf')

def plot_bpt(savename='None', axis_obj='False', composite_bpt_point=[-47], composite_bpt_errs=0, use_other_df = 0, use_df='False', add_background=False, color_gals=False, add_prospector='False', groupID=-1, skip_gals=False):
    """Plots the bpt diagram for the objects in zobjs

    Parameters:
    emission_df (pd.DataFrame): Dataframe containing emission line measurements and info
    zobjs (list): list of tuples of the form (field, v4id)
    savename (str): location with name ot save the file
    axis_obj (matplotlib_axis): Replace with an axis to plot on an existing axis
    composite_bpt_point (): Set to the point if using a composite sed and you want to plot the bpt point of that
    use_other_df (boolean): Set to one to use another df, and then specify with use_df
    use_df (pd.DataFrame): Set to a dataframe to plot that instead of gal_df
    small (boolean): Set to true to make the points small and grey
    add_prospector (str): Set to run name to add the point from the recent prospector fit
    groupID (int): groupID when using prospector
    skip_gals (Boolean): Set to true to not plot the galaxies in the cluster

    Returns:
    """

    axisfont = 14
    ticksize = 12
    ticks = 8

    if use_other_df == 0:
        gal_df = read_interp_axis_ratio()
        # gal_df = read_filtered_ar_df()
    else:
        gal_df = use_df

    n_gals = len(gal_df)
    
    # Get the bpt valeus to plot for all objects
    gal_df = get_bpt_coords(gal_df)

    
    if axis_obj == 'False':
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        ax = axis_obj

    # Bpt diagram lines
    xline = np.arange(-3.0, 0.469, 0.001)
    yline = 0.61 / (xline - 0.47) + 1.19  # Kewley (2001)
    xlineemp = np.arange(-3.0, 0.049, 0.001)
    ylineemp = 0.61 / (xlineemp - 0.05) + 1.3  # Kauffman (2003)
    ax.plot(xline, yline, color='dimgrey', lw=2,
            ls='--', label='Kewley+ (2001)')
    ax.plot(xlineemp, ylineemp, color='dimgrey',
            lw=2, ls='-', label='Kauffmann+ (2003)')

    if add_background==True:
        filtered_gal_df = ascii.read(imd.loc_filtered_gal_df).to_pandas()
        filtered_gal_df = get_bpt_coords(filtered_gal_df)
        ax.plot(filtered_gal_df['log_NII_Ha'], filtered_gal_df['log_OIII_Hb'], marker='o', color='grey', ls='None', markersize=1.5, zorder=1)
    
    cmap = mpl.cm.plasma
    norm = mpl.colors.Normalize(vmin=1, vmax=len(gal_df)) 
    print(len(gal_df))
    if skip_gals==False:
        for gal in range(len(gal_df)):
            row = gal_df.iloc[gal]
            if color_gals:
                rgba = cmap(norm(row['group_gal_id']))
            else:
                rgba = 'black'
            ax.errorbar(row['log_NII_Ha'], row['log_OIII_Hb'], xerr=row[
                            'log_NII_Ha_err'], yerr=row['log_OIII_Hb_err'], marker='o', color=rgba, ecolor='grey', ls='None', zorder=1)
    
    # gal_df_2 = gal_df[gal_df['agn_flag']>3]
    # ax.errorbar(gal_df_2['log_NII_Ha'], gal_df_2['log_OIII_Hb'], xerr=gal_df_2[
    #                 'log_NII_Ha_err'], yerr=gal_df_2['log_OIII_Hb_err'], marker='o', color='orange', ecolor='grey', ls='None')

    if composite_bpt_point[0] != -47:
        ax.errorbar(composite_bpt_point[0], composite_bpt_point[
            1], xerr=np.array([composite_bpt_errs[0]]).T, yerr=np.array([composite_bpt_errs[1]]).T, marker='o', color='red', ecolor='red', zorder=1)


    if add_prospector != 'False':
        prospector_fit_df = ascii.read(imd.prospector_emission_fits_dir + f'/{add_prospector}_emission_fits/{groupID}_emission_fits.csv').to_pandas()
        prospector_n2ha = prospector_fit_df['log_N2_Ha'].iloc[0]
        prospector_o3hb = prospector_fit_df['log_O3_Hb'].iloc[0]
        ax.plot(prospector_n2ha, prospector_o3hb, marker='x', color='orange', markersize=10, mew=3, ls='None', zorder=10000, label='Prospector')


    ax.set_xlim(-2, 1)
    ax.set_ylim(-1.2, 1.5)

    if axis_obj == 'False':
        ax.set_xlabel('log(N[II] 6583 / H$\\alpha$)', fontsize=axisfont)
        ax.set_ylabel('log(O[III] 5007 / H$\\beta$)', fontsize=axisfont)
        ax.tick_params(labelsize=ticksize, size=ticks)

        if savename != 'None':
            fig.savefig(savename)
        else:
            plt.show()
    if use_other_df != 0:
        return gal_df



def plot_all_bpt_clusters(n_clusters):
    """Plots the bpt diagram for every cluster

    Parameters:
    n_clusters (int): Number of clusters

    Returns:
    """
    # Read in the emission lines dataframe
    emission_df = read_emission_df()
    for groupID in range(n_clusters):
        plot_bpt_cluster(emission_df, groupID)

def plot_all_clusters_same_bpt(n_clusters):
    """Plots the bpt diagram for every cluster

    Parameters:
    n_clusters (int): Number of clusters

    Returns:
    """
    # Read in the emission lines dataframe
    emission_df = read_emission_df()
    for groupID in range(n_clusters):
        plot_bpt_cluster(emission_df, groupID)
        


def plot_bpt_cluster(emission_df, groupID, axis_obj = 'False'):
    """Plots the bpt diagram for one cluster, given emission df
    
    Parameters:
    emission_df (pd.DataFrame): Use read_emission_df
    groupID (int): ID number of the group to plot for
    """
    # Get the names of all galaxies in the cluster
    cluster_names, fields_ids = cdf.get_cluster_fields_ids(groupID)
    fields_ids = [(obj[0], int(obj[1])) for obj in fields_ids]
    # Location to save the file
    savename = imd.cluster_bpt_plots_dir + f'/{groupID}_BPT.pdf'
    plot_bpt(emission_df, fields_ids, savename=savename, axis_obj=axis_obj)

# plot_bpt()


# ar_df = read_filtered_ar_df()
# ar_df['log_sed_sfr'] = np.log10(ar_df['sed_sfr'])
# ar_path = imd.mosdef_dir + '/axis_ratio_data/Merged_catalogs/filtered_ar_df.csv'
# ar_df.to_csv(ar_path, index=False)
plot_bpt_all_composites()