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

def plot_halpha_hists(n_clusters):
    filtered_gal_df = ascii.read(imd.loc_filtered_gal_df).to_pandas()

    filtered_gal_df_hbfilt = filtered_gal_df[filtered_gal_df['hb_flux'] > -98]
    filtered_gal_df_hbfilt = filtered_gal_df_hbfilt[filtered_gal_df_hbfilt['hb_detflag_sfr']==0]
    all_has = filtered_gal_df_hbfilt['ha_flux']
    all_hbs = filtered_gal_df_hbfilt['hb_flux']
    all_balmers = all_has/all_hbs
    all_masses = filtered_gal_df_hbfilt['log_mass']

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
plot_halpha_hists(19)