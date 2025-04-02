from mosdef_obj_data_funcs import *
from read_data import mosdef_df
import matplotlib.pyplot as plt
from scipy import integrate, interpolate
import initialize_mosdef_dirs as imd
from read_data import linemeas_df, mosdef_df, metal_df, sfrs_df
import pandas as pd
import numpy as np
from astropy.io import ascii
import matplotlib as mpl
from plot_vals import *

save_dir = '/Users/brianlorenz/uncover/Data/ha_pab_ews_mosdef/'


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

    return merged_linemeas_df



def fit_absorption_lines(mosdef_obj):
    cont_df = read_fast_continuum(mosdef_obj)

    flux = cont_df['f_lambda_rest']
    wave = cont_df['rest_wavelength']

    optical_idxs = np.logical_and(wave>5000, wave<7500)
    halpha_idxs = np.logical_and(wave>6300, wave<6700)

    inner_region_ha = [6400, 6700]
    outer_region_ha = [6000, 7100]
    ha_mask, ha_inner_mask, ha_outer_mask = mask_waves(wave, inner_region_ha, outer_region_ha)
    ha_cont = fit_continuum(wave, flux, ha_mask)
    ha_ew_value, ha_ew_flux = measure_ew(wave, flux, ha_cont, ha_inner_mask)

    inner_region_pab = [12700, 12900]
    outer_region_pab = [12500, 13100]
    pab_mask, pab_inner_mask, pab_outer_mask = mask_waves(wave, inner_region_pab, outer_region_pab)
    pab_cont = fit_continuum(wave, flux, pab_mask)
    pab_ew_value, pab_ew_flux = measure_ew(wave, flux, pab_cont, pab_inner_mask)

    
    
    #  = interpolate.interp1d(wave[ha_mask], flux[ha_mask], fill_value=-99, bounds_error=False)

    if mosdef_obj['V4ID'] == 11802:
        fig, axarr = plt.subplots(1, 2, figsize=(12,6))

        axarr[0].step(wave[pab_outer_mask], flux[pab_outer_mask], label='FAST')
        axarr[0].step(wave[pab_outer_mask], pab_cont[pab_outer_mask], label='linear continuum')
        axarr[1].step(wave[~pab_inner_mask], pab_ew_flux[~pab_inner_mask])
        axarr[0].legend()
        plt.show()
        plt.close('all')
        breakpoint()
    # plt.show()

    return ha_ew_value, pab_ew_value

def mask_waves(wave, inner_region, outer_region):
    inner_mask = np.logical_or(wave<inner_region[0], wave>inner_region[1])
    outer_mask = np.logical_and(wave>outer_region[0], wave<outer_region[1])
    total_mask = np.logical_and(inner_mask, outer_mask)
    return total_mask, inner_mask, outer_mask

def fit_continuum(wave, flux, mask):
    coefficients = np.polyfit(wave[mask], flux[mask], 1)
    slope = coefficients[0]
    intercept = coefficients[1]
    cont_fit = slope*wave + intercept
    return cont_fit

def measure_ew(wave, flux, cont, inner_mask):
    ew_flux = (cont-flux) / cont
    ew_measure = np.trapz(ew_flux[~inner_mask], wave[~inner_mask])
    return ew_measure, ew_flux


def measure_all_ews():
    merged_linemeas_df = mosdef_nii(linemeas_df, mosdef_df)
    ha_ews = []
    pab_ews = []
    for i in range(len(merged_linemeas_df)):
        if merged_linemeas_df.iloc[i]['V4ID'] < 0:
            ha_ews.append(-99)
            pab_ews.append(-99)
            continue
        mosdef_obj = get_mosdef_obj(merged_linemeas_df.iloc[i]['FIELD_STR'],merged_linemeas_df.iloc[i]['V4ID'])
        ha_ew, pab_ew = fit_absorption_lines(mosdef_obj)
        ha_ews.append(ha_ew)
        pab_ews.append(pab_ew)
    merged_linemeas_df['ha_eq_width'] = ha_ews
    merged_linemeas_df['pab_eq_width'] = pab_ews
    simple_df = merged_linemeas_df[['FIELD_STR', 'V4ID', 'ha_eq_width', 'pab_eq_width']].copy()
    simple_df.to_csv(save_dir + 'ews_simple.csv', index=False)
    merged_linemeas_df.to_csv(save_dir + 'ews_alldata.csv', index=False)

def plot_ews(cvar='mass'):
    merged_linemeas_df = ascii.read(save_dir + 'ews_alldata.csv').to_pandas()
    merged_linemeas_df = merged_linemeas_df[merged_linemeas_df['ha_eq_width']>-10]

    fig, ax = plt.subplots(figsize=(7,6))

    for i in range(len(merged_linemeas_df)):
        row = merged_linemeas_df.iloc[i]

        if cvar == 'mass':
            cmap = mpl.cm.inferno
            norm = mpl.colors.Normalize(vmin=8, vmax=11) 
            rgba = cmap(norm(row['LMASS']))
            cbar_label = stellar_mass_label
        if cvar == 'sfr':
            cmap = mpl.cm.viridis
            norm = mpl.colors.LogNorm(vmin=0.1, vmax=1) 
            rgba = cmap(norm(row['LSFR']))
            cbar_label = sfr_label

        ax.plot(row['pab_eq_width'], row['ha_eq_width'], color=rgba, marker='o', mec='black', ls='None')
    
    ax.plot([-100, 100], [-100, 100], color='darkblue', ls='--', label='y=x')
    ax.plot([-100, 100], [-200, 200], color='blue', ls='--', label='y=2x')
    ax.plot([-100, 100], [-300, 300], color='cornflowerblue', ls='--', label='y=3x')

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    ax.set_xlim(-1, 12)
    ax.set_ylim(-1, 12)
    ax.set_ylabel('Ha Eq Width', fontsize=14)
    ax.set_xlabel('PaB Eq Width', fontsize=14)
    ax.legend()
    ax.tick_params(labelsize=14)
    scale_aspect(ax)

    fig.savefig(save_dir + f'ha_vs_pab_ew_mosdef_{cvar}.pdf', bbox_inches='tight')
    # breakpoint()

         

# measure_all_ews()
plot_ews(cvar='mass')
plot_ews(cvar='sfr')