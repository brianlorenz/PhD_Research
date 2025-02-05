import initialize_mosdef_dirs as imd
from read_data import linemeas_df, mosdef_df, metal_df, sfrs_df
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plot_vals import *
from compute_av import sanders_nii_ratio, sanders_plane

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

    # Combine and filter AGN
    linemeas_df.drop('AGNFLAG', axis=1, inplace=True)
    merged_linemeas_df = pd.merge(linemeas_df, mosdef_df, on='ID')
    merged_linemeas_df = merged_linemeas_df[merged_linemeas_df['AGNFLAG'] == 0]
    merged_linemeas_df = pd.merge(merged_linemeas_df, metal_df, on='ID')

    median_nii_ha = np.median(merged_linemeas_df['nii_ha_ratio'])
    std_nii_ha = np.std(merged_linemeas_df['nii_ha_ratio'])
    print(f'median NII/HA (combined) = {round(median_nii_ha,4)}, with std = {round(std_nii_ha,4)}')

    # Histogram figure
    fig, axarr = plt.subplots(1,2,figsize=(15,5))
    ax_6865 = axarr[0]
    ax_combined = axarr[1]

    bins = np.arange(0, 2, 0.05)
    ax_6865.hist(merged_linemeas_df['nii6865_ha_ratio'], bins=bins, color='black')
    ax_combined.hist(merged_linemeas_df['nii_ha_ratio'], bins=bins, color='black')

    ax_6865.axvline(np.median(merged_linemeas_df['nii6865_ha_ratio']), ls='--', color='red')
    ax_combined.axvline(median_nii_ha, ls='--', color='red')
    
    for ax in axarr:
        ax.set_xlim(0,2)
        ax.set_ylim(0,75)
        ax.set_xlabel('N Galaxies')    
        scale_aspect(ax)
    ax_6865.set_xlabel('NII 6865 / H$\\alpha$')    
    ax_combined.set_xlabel('NII (combined) / H$\\alpha$')    

    fig.savefig('/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/nii_relations/nii_analysis.pdf')
    plt.close('all')

    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(linemeas_df['NII6585_FLUX'], linemeas_df['nii_combined_flux'], color='black', marker='o', ls='None')
    x_vals = np.arange(0,2e-16,1e-18)
    y_vals = 1.5*x_vals
    ax.plot(x_vals, y_vals, color='red', marker='None', ls='--', label='y=1.5*x')
    ax.plot(x_vals, x_vals, color='green', marker='None', ls='--', label='y=x')
    ax.set_xlabel('NII6585 Flux')
    ax.set_ylabel('NII Combined Flux')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    fig.savefig('/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/nii_relations/compare_nii.pdf')
    plt.close('all')

    # Searching for corrrelations
    def check_cor(yvar):
        fig, ax = plt.subplots(figsize=(6,6))
        # Filter to positive data
        mask = merged_linemeas_df[yvar] > -98
        ax.plot(merged_linemeas_df[mask]['nii_ha_ratio'], merged_linemeas_df[mask][yvar], color='black', marker='o', ls='None')
        ax.set_xlabel('NII (combined) / H$\\alpha$')
        ax.set_ylabel(yvar)
        fig.savefig(f'/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/nii_relations/nii_correlation_{yvar}.pdf')
        plt.close('all')
    check_cor('LMASS')
    check_cor('LSFR')
    check_cor('Z_MOSFIRE')
    check_cor('12LOGOH_PP04_N2')

    return merged_linemeas_df

def check_sanders_plane(merged_linemeas_df, sfrs_df, change_offset=False):
    fig, axarr = plt.subplots(1,3,figsize=(15,5))
    sfrs_df = sfrs_df[sfrs_df['SFR_CORR'] > 0]
    merged_linemeas_df = sfrs_df.merge(merged_linemeas_df, on='ID')
    mosdef_log_sfr = np.log10(merged_linemeas_df['SFR_CORR'])
    mosdef_log_mass = merged_linemeas_df['LMASS']
    mosdef_metallicity = merged_linemeas_df['12LOGOH_PP04_N2']
    # mosdef_nii6865_ha = merged_linemeas_df['nii6865_ha_ratio']
    mosdef_ha_abscor = merged_linemeas_df['HA6565_PREFERREDFLUX_x'] + merged_linemeas_df['HA6565_ABS_FLUX']
    mosdef_nii6865 = merged_linemeas_df['NII6585_PREFERREDFLUX'] 
    mosdef_nii6865_ha = mosdef_nii6865 / mosdef_ha_abscor

    predicted_metallicity = sanders_plane(mosdef_log_mass, mosdef_log_sfr)
    if change_offset:
        linear_factor = 8.5
        add_str='_scaled'
    else:
        linear_factor = 8.69
        add_str=''
    predicted_nii = sanders_nii_ratio(predicted_metallicity, linear_scale=linear_factor)
    predicted_nii_met = sanders_nii_ratio(mosdef_metallicity, linear_scale=linear_factor)

    ax_fmr = axarr[0]
    ax_nii = axarr[1]
    ax_nii_met = axarr[2]

    ax_fmr.plot(mosdef_metallicity, predicted_metallicity, marker='o', color='black', ls='None')
    ax_fmr.set_xlabel('MOSDEF Metallicity')
    ax_fmr.set_ylabel('Predicted Metallicity from M+SFR')
    ax_fmr.set_xlim(7.75, 9.25)
    ax_fmr.set_ylim(7.75, 9.25)
    
    ax_nii.plot(mosdef_nii6865_ha, predicted_nii, marker='o', color='black', ls='None')
    ax_nii.set_xlabel('MOSDEF NII/Ha')
    ax_nii.set_ylabel('Predicted NII/Ha from M+SFR')
    ax_nii.set_xlim(0, 1.5)
    ax_nii.set_ylim(0, 1.5)

    ax_nii_met.plot(mosdef_nii6865_ha, predicted_nii_met, marker='o', color='black', ls='None')
    ax_nii_met.set_xlabel('MOSDEF NII/Ha')
    ax_nii_met.set_ylabel('Predicted NII/Ha from MOSDEF Metallicity')
    ax_nii_met.set_xlim(0, 1.5)
    ax_nii_met.set_ylim(0, 1.5)

    for ax in axarr:
        ax.plot([-100, 100], [-100, 100], ls='--', color='red', marker='None')


    fig.savefig(f'/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/nii_relations/MOSDEF_calibration{add_str}.pdf')

merged_linemeas_df = mosdef_nii(linemeas_df, mosdef_df)
check_sanders_plane(merged_linemeas_df, sfrs_df, change_offset=False)