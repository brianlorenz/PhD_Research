import sys
# sys.path.insert(0, '/Users/brianlorenz/code/mosdef_code/')
# from mosdef_obj_data_funcs import *
# from read_data import mosdef_df
import matplotlib.pyplot as plt
from scipy import integrate, interpolate
import initialize_mosdef_dirs as imd
# from read_data im/port linemeas_df, mosdef_df, metal_df, sfrs_df
import pandas as pd
import numpy as np
from astropy.io import ascii
import matplotlib as mpl
from plot_vals import *
import time
from uncover_make_sed import read_full_phot_sed
from uncover_read_data import read_supercat, read_spec_cat, read_fluxcal_spec


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


def read_absorp_npz(id_dr3):
    npz_loc = '/Users/brianlorenz/uncover/Data/prospector_spectra_no_emission/spectra.npz'
    np_df = np.load(npz_loc, allow_pickle=True)
    np_dict = np_df['spectra'].item()
    rest_wave = np_dict[id_dr3][:,0]
    full_model = np_dict[id_dr3][:,1]
    absorp_model = np_dict[id_dr3][:,2]
    cont_df = pd.DataFrame(zip(rest_wave, full_model, absorp_model), columns=['rest_wave', 'rest_full_model', 'rest_absorp_model_maggies'])
    
    cont_df['rest_absorp_model_jy'] =  cont_df['rest_absorp_model_maggies'] * 3631
    cont_df['rest_absorp_model_10njy'] =  cont_df['rest_absorp_model_jy'] / 1e8
    cont_df['rest_full_model_jy'] =  cont_df['rest_full_model'] * 3631
    c = 299792458 # m/s
    cont_df['rest_absorp_model_erg_aa'] = cont_df['rest_absorp_model_jy'] * (1e-23*1e10*c / (cont_df['rest_wave']**2))
    return cont_df


def fit_absorption_lines(id_dr3, plot='None'):
    supercat_df = read_supercat()
    id_msa = supercat_df[supercat_df['id']==id_dr3]['id_msa'].iloc[0]
    cont_df = read_absorp_npz(id_dr3)
    spec_df = read_fluxcal_spec(id_msa)

    wave = cont_df['rest_wave']


    # Scaling model to match the spectra
    outer_scaling_region_ha = [5200, 7800]
    inner_scaling_region_ha = [6000, 7000]
    spec_scaling_idxs_ha, _, optical_region_mask_spec = mask_waves(spec_df['rest_wave_aa'], inner_scaling_region_ha, outer_scaling_region_ha)
    cont_scaling_idxs_ha, _, optical_region_mask= mask_waves(wave, inner_scaling_region_ha, outer_scaling_region_ha)
    spec_waves_arr = spec_df[spec_scaling_idxs_ha]['rest_wave_aa'].to_numpy()
    cont_df_match_spec = cont_df.iloc[abs(cont_df['rest_wave'].to_numpy()[:,None] - spec_waves_arr).argmin(0)]
    f1 = spec_df[spec_scaling_idxs_ha]['rest_flux_calibrated_erg_aa'].to_numpy()
    f2 = cont_df_match_spec['rest_absorp_model_erg_aa'].to_numpy()
    a12 = np.sum(f1 * f2) / np.sum(f2**2)
    cont_df['rest_absorp_model_erg_aa_hascaled'] = cont_df['rest_absorp_model_erg_aa'] * a12
    print(f'ha scale {a12}')

    outer_scaling_region_pab = [11700, 14000]
    inner_scaling_region_pab = [12300, 13300]
    spec_scaling_idxs_pab, _, ir_region_mask_spec = mask_waves(spec_df['rest_wave_aa'], inner_scaling_region_pab, outer_scaling_region_pab)
    cont_scaling_idxs_pab, _, ir_region_mask= mask_waves(wave, inner_scaling_region_pab, outer_scaling_region_pab)
    spec_waves_arr = spec_df[spec_scaling_idxs_pab]['rest_wave_aa'].to_numpy()
    cont_df_match_spec = cont_df.iloc[abs(cont_df['rest_wave'].to_numpy()[:,None] - spec_waves_arr).argmin(0)]
    f1 = spec_df[spec_scaling_idxs_pab]['rest_flux_calibrated_erg_aa'].to_numpy()
    f2 = cont_df_match_spec['rest_absorp_model_erg_aa'].to_numpy()
    a12 = np.sum(f1 * f2) / np.sum(f2**2)
    cont_df['rest_absorp_model_erg_aa_pabscaled'] = cont_df['rest_absorp_model_erg_aa'] * a12
    print(f'pab scale {a12}')

    # plt.plot(cont_df[optical_region_mask]['rest_wave'], cont_df[optical_region_mask]['rest_absorp_model_erg_aa_pabscaled'])
    # plt.plot(spec_df[optical_region_mask_spec]['rest_wave_aa'], spec_df[optical_region_mask_spec]['rest_flux_calibrated_erg_aa'])
    # plt.show()
    # plt.close()

    # plt.plot(cont_df[ir_region_mask]['rest_wave'], cont_df[ir_region_mask]['rest_absorp_model_erg_aa_pabscaled'])
    # plt.plot(spec_df[ir_region_mask_spec]['rest_wave_aa'], spec_df[ir_region_mask_spec]['rest_flux_calibrated_erg_aa'])


    ha_flux = cont_df['rest_absorp_model_erg_aa_hascaled']
    pab_flux = cont_df['rest_absorp_model_erg_aa_pabscaled']

    inner_region_ha = [6400, 6700]
    outer_region_ha = [6000, 7100]
    ha_mask, ha_inner_mask, ha_outer_mask = mask_waves(wave, inner_region_ha, outer_region_ha)
    ha_cont = fit_continuum(wave, ha_flux, ha_mask)
    ha_ew_value, ha_ew_flux = measure_ew(wave, ha_flux, ha_cont, ha_inner_mask)

    inner_region_pab = [12700, 12900]
    outer_region_pab = [12500, 13100]
    if id_dr3 == 62937:
        inner_region_pab = [12750, 12900]
        outer_region_pab = [12600, 13000]
    if plot != 'None':
        # outer_region_pab = [6200, 14000] # Messes up the value, but lets us see apb more clearly
        pass
    pab_mask, pab_inner_mask, pab_outer_mask = mask_waves(wave, inner_region_pab, outer_region_pab)
    pab_cont = fit_continuum(wave, pab_flux, pab_mask)
    pab_ew_value, pab_ew_flux = measure_ew(wave, pab_flux, pab_cont, pab_inner_mask)
    

    
    if plot != 'None':
        if plot == 'ha':
            outer_mask = ha_outer_mask
            inner_mask = ha_inner_mask
            cont = ha_cont
            ew_flux = ha_ew_flux
            flux = ha_flux
        elif plot == 'pab':
            outer_mask = pab_outer_mask
            inner_mask = pab_inner_mask
            cont = pab_cont
            ew_flux = pab_ew_flux
            flux = pab_flux
        fig, axarr = plt.subplots(1, 2, figsize=(12,6))
        # axarr[0].step(wave[outer_mask], cont_df['rest_full_model'][outer_mask], label='Prospector Emission', color='black')
        # axarr[0].step(wave[outer_mask], cont_df['rest_absorp_model_maggies'][outer_mask], label='Prospector Continuum', color='red')
        axarr[0].step(wave[outer_mask], flux[outer_mask], label='Prospector Continuum', color='black')
        axarr[0].step(wave[outer_mask], cont[outer_mask], label='linear continuum')
        axarr[0].axvline(x=6565, color='r', linestyle='--', alpha=0.5)
        axarr[0].axvline(x=12820, color='r', linestyle='--', alpha=0.5)
        sed_df = read_full_phot_sed(id_dr3)
        zqual_df = read_spec_cat()
        redshift = zqual_df[zqual_df['id_msa']==id_msa]['z_spec'].iloc[0]
        sed_df['rest_wave'] = sed_df['eff_wavelength']*10000 / (1+redshift)
        sed_df['rest_flux'] = sed_df['flux']* (1+redshift)
        # axarr[0].plot(sed_df['rest_wave'], sed_df['rest_flux'], marker='o', ls='None', color='orange')
        axarr[1].step(wave[~inner_mask], ew_flux[~inner_mask])
        axarr[0].legend()
        axarr[0].set_xlim(outer_region_pab)
        
        plt.show()
        plt.close('all')

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

         
if __name__ == "__main__":
    # print(fit_absorption_lines(62937, plot='pab'))


    
    
    
    # ids = np.array([30052,
    #     30804,
    #     31608,
    #     37182,
    #     37776,
    #     44283,
    #     46339,
    #     47771,
    #     49023,
    #     52140,
    #     52257,
    #     54625,
    #     60579,
    #     62937])
    # for id_dr3 in ids:
    #     ha_abs, pab_abs = fit_absorption_lines(id_dr3)
    #     print(f'{id_dr3}, ha_abs: {ha_abs}, pab_abs: {pab_abs}')




    ### OLD MOSDEF STUFF
    # measure_all_ews()
    # plot_ews(cvar='mass')
    # plot_ews(cvar='sfr')
    
    pass


