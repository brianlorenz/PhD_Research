import initialize_mosdef_dirs as imd
import matplotlib.pyplot as plt
from astropy.io import ascii
from cosmology_calcs import flux_to_luminosity
import numpy as np

def make_halpha_hist(save_dir, groupID, xvar, xlabel, removed_gal_number, bins):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.hist(xvar, color='black', bins=bins)
    ax.axvline(np.median(xvar), ls='--', color='red')
    ax.set_xlabel(f'Halpha {xlabel}', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.text(0.7, 0.95, f'No detection: {removed_gal_number}', transform=ax.transAxes, fontsize=14)
    ax.set_title(f'Group {groupID}', fontsize=18)
    fig.savefig(save_dir + f'/{groupID}_{xlabel}.pdf')

def plot_halpha_hists(n_clusters):
    for groupID in range(n_clusters):
        group_df = ascii.read(imd.cluster_indiv_dfs_dir + f'/{groupID}_cluster_df.csv').to_pandas()
        halpha_fluxes = group_df['ha_flux'][group_df['ha_flux'] > -98]
        removed_gal_number = len(group_df) - len(halpha_fluxes)
        redshifts = group_df['Z_MOSFIRE'][group_df['ha_flux'] > -98]
        halpha_luminosities = flux_to_luminosity(halpha_fluxes, redshifts)

        save_dir = imd.mosdef_dir + '/Clustering/cluster_stats/indiv_halphas'
        imd.check_and_make_dir(save_dir)
        
        flux_bins = np.arange(1e-18, 1e-16, 4e-18)
        lum_bins = np.arange(1e41, 1e43, 4e41)
        make_halpha_hist(save_dir, groupID, halpha_fluxes, 'Flux', removed_gal_number, flux_bins)
        make_halpha_hist(save_dir, groupID, halpha_luminosities, 'Luminosity', removed_gal_number, lum_bins)
        plt.close('all')
plot_halpha_hists(19)