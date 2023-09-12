import initialize_mosdef_dirs as imd
import matplotlib.pyplot as plt
from astropy.io import ascii
from cosmology_calcs import flux_to_luminosity
import numpy as np
from plot_vals import *
cluster_summary_df = imd.read_cluster_summary_df()

def make_hist(save_dir, groupID, xvar, xlabel, removed_gal_number, bins):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.hist(xvar, color='black', bins=bins)
    ax.axvline(np.median(xvar), ls='--', color='red', label='Median')
    if xlabel == 'Balmer_Dec':
        cluster_balmer = cluster_summary_df.iloc[groupID]['balmer_dec']
        ax.axvline(cluster_balmer, ls='--', color='mediumseagreen', label='Cluster')
        ax.legend(fontsize=14, loc=2)
    ax.set_xlabel(f'Halpha {xlabel}', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.text(0.7, 0.95, f'No detection: {removed_gal_number}', transform=ax.transAxes, fontsize=14)
    ax.set_title(f'Group {groupID}', fontsize=18)
    imd.check_and_make_dir(save_dir + f'/{xlabel}')
    fig.savefig(save_dir + f'/{xlabel}/{groupID}_{xlabel}.pdf')

def plot_group_hists(n_clusters):
    filtered_gal_df = ascii.read(imd.loc_filtered_gal_df).to_pandas()

    filtered_gal_df_hbfilt = filtered_gal_df[filtered_gal_df['hb_flux'] > -98]
    filtered_gal_df_hbfilt = filtered_gal_df_hbfilt[filtered_gal_df_hbfilt['hb_detflag_sfr']==0]
    all_has = filtered_gal_df_hbfilt['ha_flux']
    all_hbs = filtered_gal_df_hbfilt['hb_flux']
    all_balmers = all_has/all_hbs
    all_masses = filtered_gal_df_hbfilt['log_mass']
    avs_nofilt = filtered_gal_df['AV']
    masses_nofilt = filtered_gal_df['log_mass']
    group_balmer_tuples = []
    group_massfilt_tuples = []
    group_av_tuples = []
    group_mass_tuples = []
    groupIDs = []
    

    for groupID in range(n_clusters):
        group_df = ascii.read(imd.cluster_indiv_dfs_dir + f'/{groupID}_cluster_df.csv').to_pandas()
        halpha_fluxes = group_df['ha_flux'][group_df['ha_flux'] > -98]
        removed_gal_number = len(group_df) - len(halpha_fluxes)
        redshifts = group_df['Z_MOSFIRE'][group_df['ha_flux'] > -98]
        halpha_luminosities = flux_to_luminosity(halpha_fluxes, redshifts)
        log_masses = group_df['log_mass']
        log_masses_filt = group_df['log_mass'][group_df['ha_flux'] > -98]
        pseudo_ssfr = halpha_luminosities / (10**log_masses_filt)
        hbfilt = group_df[group_df['hb_flux'] > -98]
        hbfilt = hbfilt[hbfilt['hb_detflag_sfr'] == 0]
        halpha_fluxes_hbfilt = hbfilt['ha_flux']
        hbeta_fluxes_hbfilt = hbfilt['hb_flux']
        balmer_decs = halpha_fluxes_hbfilt/hbeta_fluxes_hbfilt
        log_masses_hbfilt = hbfilt['log_mass']
        removed_gal_number_balmer = len(group_df) - len(balmer_decs)
        group_avs = group_df['AV']

        

        save_dir = imd.mosdef_dir + '/Clustering/cluster_stats/indiv_group_plots'
        imd.check_and_make_dir(save_dir)
        
        flux_bins = np.arange(1e-18, 1e-16, 4e-18)
        lum_bins = np.arange(1e41, 1e43, 4e41)
        mass_bins = np.arange(9, 11, 0.1)
        balmer_bins = np.arange(0, 10, 0.2)
        # Histograms
        make_hist(save_dir, groupID, halpha_fluxes, 'Flux', removed_gal_number, flux_bins)
        make_hist(save_dir, groupID, halpha_luminosities, 'Luminosity', removed_gal_number, lum_bins)
        make_hist(save_dir, groupID, pseudo_ssfr, 'HaLum_Mass', removed_gal_number, lum_bins/1e11)
        make_hist(save_dir, groupID, log_masses, 'log_mass', 0, mass_bins)
        make_hist(save_dir, groupID, balmer_decs, 'Balmer_Dec', removed_gal_number_balmer, balmer_bins)
        
        # Balmer dec vs Mass in each group
        fig, ax = plt.subplots(figsize=(8,8))
        ax.plot(all_masses, all_balmers, marker='o', color='grey', ls='None', ms=2)
        ax.plot(log_masses_hbfilt, balmer_decs, marker='o', color='black', ls='None')
        ax.set_xlim(9,11)
        ax.set_ylim(0,8)
        ax.text(0.7, 0.95, f'No detection: {removed_gal_number_balmer}', transform=ax.transAxes, fontsize=14)
        ax.set_xlabel(stellar_mass_label, fontsize=14)
        ax.set_ylabel(balmer_label, fontsize=14)
        ax.tick_params(labelsize=14)
        imd.check_and_make_dir(save_dir + f'/balmer_mass')
        fig.savefig(save_dir + f'/balmer_mass/{groupID}_balmer_mass.pdf')
        plt.close('all')

        # AV vs Mass in each group
        fig, ax = plt.subplots(figsize=(8,8))
        ax.plot(masses_nofilt, avs_nofilt, marker='o', color='grey', ls='None', ms=2)
        ax.plot(log_masses, group_avs, marker='o', color='black', ls='None')
        ax.set_xlim(9,11)
        ax.set_ylim(0,3)
        ax.set_xlabel(stellar_mass_label, fontsize=14)
        ax.set_ylabel('AV', fontsize=14)
        ax.tick_params(labelsize=14)
        imd.check_and_make_dir(save_dir + f'/av_mass')
        fig.savefig(save_dir + f'/av_mass/{groupID}_av_mass.pdf')
        plt.close('all')

        group_balmer_tuple = np.percentile(balmer_decs, [16,50,84])
        group_massfilt_tuple = np.percentile(log_masses_hbfilt, [16,50,84])
        group_av_tuple = np.percentile(group_avs, [16,50,84])
        group_mass_tuple = np.percentile(log_masses, [16,50,84])
        group_balmer_tuples.append(group_balmer_tuple)
        group_massfilt_tuples.append(group_massfilt_tuple)
        group_av_tuples.append(group_av_tuple)
        group_mass_tuples.append(group_mass_tuple)
        groupIDs.append(groupID)
    
    # Balmer and AV vs mass total
    def plot_all_groups(xvars, yvars, groupIDs, ylabel):
        fig, ax = plt.subplots(figsize=(8,8))
        ignore_groups = imd.ignore_groups
        for groupID in groupIDs:
            if groupID in ignore_groups:
                continue
            group_xvars = xvars[groupID]
            group_yvars = yvars[groupID]
            median_xvar = group_xvars[1]
            err_xvar = [[median_xvar-group_xvars[0]], [group_xvars[2]-median_xvar]]
            median_yvar = group_yvars[1]
            err_yvar = [[median_yvar-group_yvars[0]], [group_yvars[2]-median_yvar]]
            ax.errorbar(median_xvar, median_yvar, xerr=err_xvar, yerr=err_yvar, marker='o', color='black')
            ax.text(median_xvar, median_yvar, f'{int(groupID)}', fontsize=14)
        ax.set_xlim(9,11)
        ax.set_xlabel(stellar_mass_label, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.tick_params(labelsize=14)
        if ylabel == 'Balmer Dec':
            ax.set_ylim(0,8)
            imd.check_and_make_dir(save_dir + f'/balmer_mass')
            fig.savefig(save_dir + f'/balmer_mass/all_balmer_mass.pdf')
        if ylabel == 'AV':
            ax.set_ylim(0,3)
            imd.check_and_make_dir(save_dir + f'/av_mass')
            fig.savefig(save_dir + f'/av_mass/all_av_mass.pdf')
        plt.close('all')
    plot_all_groups(group_massfilt_tuples, group_balmer_tuples, groupIDs, 'Balmer Dec')
    plot_all_groups(group_mass_tuples, group_av_tuples, groupIDs, 'AV')
    
plot_group_hists(19)