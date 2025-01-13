import initialize_mosdef_dirs as imd
from read_data import linemeas_df, mosdef_df
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plot_vals import *

def mosdef_nii(linemeas_df, mosdef_df):
    
    # Redshift cut
    z_cut = np.logical_and(mosdef_df['Z_MOSFIRE'] > 1.3, mosdef_df['Z_MOSFIRE'] < 2.3)
    mosdef_df = mosdef_df[z_cut]
    linemeas_df = linemeas_df[linemeas_df['ID'].isin(mosdef_df['ID'])]
    linemeas_df = linemeas_df[linemeas_df['Z_MOSFIRE_INITQUAL']==0]
    
    # Need measured fluxes
    linemeas_df = linemeas_df[linemeas_df['HA6565_FLUX'] > 0]
    linemeas_df = linemeas_df[linemeas_df['NII6550_FLUX'] > 0]
    linemeas_df = linemeas_df[linemeas_df['NII6585_FLUX'] > 0]

    linemeas_df['ha_snr'] = linemeas_df['HA6565_FLUX'] / linemeas_df['HA6565_FLUX_ERR']
    linemeas_df['nii_combined_flux'] = linemeas_df['NII6550_FLUX'] + linemeas_df['NII6585_FLUX']
    linemeas_df['nii_combined_flux_err'] = np.sqrt(linemeas_df['NII6550_FLUX_ERR']**2 + linemeas_df['NII6585_FLUX_ERR']**2)
    linemeas_df['nii_snr'] = linemeas_df['nii_combined_flux'] / linemeas_df['nii_combined_flux_err']
    
    #SNR cut
    linemeas_df = linemeas_df[linemeas_df['ha_snr'] > 2]
    linemeas_df = linemeas_df[linemeas_df['nii_snr'] > 2]

    linemeas_df['nii_ha_ratio'] = linemeas_df['nii_combined_flux'] / linemeas_df['HA6565_FLUX'] 
    linemeas_df['nii6865_ha_ratio'] = linemeas_df['NII6585_FLUX'] / linemeas_df['HA6565_FLUX'] 

    median_nii_ha = np.median(linemeas_df['nii_ha_ratio'])
    std_nii_ha = np.std(linemeas_df['nii_ha_ratio'])
    print(f'median NII/HA (combined) = {round(median_nii_ha,4)}, with std = {round(std_nii_ha,4)}')

    fig, axarr = plt.subplots(1,2,figsize=(10,5))
    ax_6865 = axarr[0]
    ax_combined = axarr[1]

    bins = np.arange(0, 2, 0.05)
    ax_6865.hist(linemeas_df['nii6865_ha_ratio'], bins=bins, color='black')
    ax_combined.hist(linemeas_df['nii_ha_ratio'], bins=bins, color='black')

    ax_6865.axvline(np.median(linemeas_df['nii6865_ha_ratio']), ls='--', color='red')
    ax_combined.axvline(median_nii_ha, ls='--', color='red')
    
    for ax in axarr:
        ax.set_xlim(0,2)
        ax.set_ylim(0,75)
        ax.set_xlabel('N Galaxies')    
        scale_aspect(ax)
    ax_6865.set_xlabel('NII 6865 / H$\\alpha$')    
    ax_combined.set_xlabel('NII (combined) / H$\\alpha$')    

    fig.savefig('/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/nii_analysis.pdf')


mosdef_nii(linemeas_df, mosdef_df)