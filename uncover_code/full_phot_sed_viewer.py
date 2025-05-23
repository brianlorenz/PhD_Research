from uncover_make_sed import read_full_phot_sed
from astropy.io import ascii
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from full_phot_read_data import read_merged_lineflux_cat
from uncover_prospector_seds import read_prospector, make_prospector
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from uncover_sed_filters import unconver_read_filters
from sedpy import observate
from uncover_read_data import flux_jy_to_erg, read_fluxcal_spec, read_supercat, read_raw_spec
from full_phot_make_prospector_models import prospector_abs_spec_folder
import os
from simple_flux_calibration import flux_calibrate_spectrum


phot_df_loc = '/Users/brianlorenz/uncover/Data/generated_tables/phot_calcs/phot_linecoverage_ha_pab_paa.csv'
colors = ['red', 'mediumseagreen']

def plot_sed(id_dr3, phot_sample_df, ha_pab=False, pab_paa=False, overplot_bump=[], plot_rest=True, plot_median_model=False):
    sed_df = read_full_phot_sed(id_dr3)
    lineflux_df = read_merged_lineflux_cat()
    prospector_spec_df, prospector_sed_df, mu = read_prospector(id_dr3, id_dr3=True)
    

    phot_sample_row = phot_sample_df[phot_sample_df['id'] == id_dr3]
    lineflux_row = lineflux_df[lineflux_df['id_dr3']==id_dr3]
    redshift = phot_sample_row['z_50'].iloc[0]
    ha_snr = lineflux_row['Halpha_snr'].iloc[0]
    pab_snr = lineflux_row['PaBeta_snr'].iloc[0]
    paa_snr = lineflux_row['PaAlpha_snr'].iloc[0]
    if ha_pab:
        line_filters = [phot_sample_row['Halpha_filter_obs'].iloc[0], phot_sample_row['PaBeta_filter_obs'].iloc[0]]
        cont_filters = [phot_sample_row['Halpha_filter_bluecont'].iloc[0], phot_sample_row['Halpha_filter_redcont'].iloc[0], phot_sample_row['PaBeta_filter_bluecont'].iloc[0], phot_sample_row['PaBeta_filter_redcont'].iloc[0]]
    if pab_paa:
        line_filters = [phot_sample_row['PaBeta_filter_obs'].iloc[0], phot_sample_row['PaAlpha_filter_obs'].iloc[0]]
        cont_filters = [phot_sample_row['PaAlpha_filter_bluecont'].iloc[0], phot_sample_row['PaAlpha_filter_redcont'].iloc[0], phot_sample_row['PaBeta_filter_bluecont'].iloc[0], phot_sample_row['PaBeta_filter_redcont'].iloc[0]]

    sed_df['rest_wave'] = sed_df['eff_wavelength'] / (1+redshift)
    sed_df['rest_flux'] = sed_df['flux'] * (1+redshift)
    sed_df['err_rest_flux'] = sed_df['err_flux'] * (1+redshift)
    prospector_spec_df['rest_wave_um'] = prospector_spec_df['wave_um'] / (1+redshift)
    prospector_spec_df['rest_flux_jy'] = prospector_spec_df['flux_jy'] * (1+redshift)
    fig, ax = plt.subplots(figsize=(6,6))
    if plot_rest:
        ax.errorbar(sed_df['rest_wave'], sed_df['rest_flux'], yerr=sed_df['err_rest_flux'], color='black', ecolor='grey', ls='None', marker='o', zorder=5, label='SED')
        ax.plot(prospector_spec_df['rest_wave_um'], prospector_spec_df['rest_flux_jy'], color='orange', ls='-', marker='None', zorder=5, label='SPS catalog model')
        if os.path.exists(f'{prospector_abs_spec_folder}{id_dr3}_prospector_no_neb.csv'):
            prospector_no_neb_df = ascii.read(f'{prospector_abs_spec_folder}{id_dr3}_prospector_no_neb.csv').to_pandas()
            supercat_df = read_supercat()
            id_msa = supercat_df[supercat_df['id']==id_dr3]['id_msa'].iloc[0]
            spec_df = read_fluxcal_spec(id_msa)
            # spec_df_raw = read_raw_spec(id_msa)
            fluxcal_func, popt = flux_calibrate_spectrum(id_msa, save_fluxcal=False)
            prospector_no_neb_df['rest_full_model_jy_scaled'] = fluxcal_func((1+redshift)*prospector_no_neb_df['rest_wave']/10000, prospector_no_neb_df['rest_full_model_jy'], popt)
            spec_df['fluxcal_recalc'] = fluxcal_func((1+redshift)*spec_df['rest_wave_aa']/10000, spec_df['flux'], popt)
            
            # breakpoint()
            ax.plot(prospector_no_neb_df['rest_wave']/10000, prospector_no_neb_df['rest_full_model_jy']*mu*(1+redshift), color='green', ls='-', marker='None', zorder=5, label='Full Model')
            ax.plot(prospector_no_neb_df['rest_wave']/10000, prospector_no_neb_df['rest_absorp_model_jy']*mu*(1+redshift), color='mediumseagreen', ls='-', marker='None', zorder=5, label='Absorption')
            # ax.plot(spec_df['rest_wave_aa']/10000, spec_df['flux'], color='black', ls='-', marker='None', zorder=5, alpha=0.8, label='Real spec without (1+z)')
            # ax.plot(spec_df['rest_wave_aa']/10000, spec_df['flux_calibrated_jy'], color='black', ls='-', marker='None', zorder=5, alpha=0.5, label='Real spec scaled and *(1+z)')
            ax.legend(loc=1, fontsize=10)
        wave_type = 'Rest'
    else:
        ax.errorbar(sed_df['eff_wavelength'], sed_df['flux'], yerr=sed_df['err_flux'], color='black', ecolor='grey', ls='None', marker='o', zorder=5)
        ax.plot(prospector_spec_df['wave_um'], prospector_spec_df['flux_jy'], color='orange', ls='-', marker='None', zorder=5)
        wave_type = 'Observed'

    for i in range(len(line_filters)):
        line_filt = line_filters[i]
        sed_row_line = sed_df[sed_df['filter']==line_filt]
        if plot_rest:
            ax.plot(sed_row_line['rest_wave'].iloc[0], sed_row_line['rest_flux'].iloc[0], color=colors[i], ls='None', marker='o', zorder=10, mec='black')
        else:
            ax.plot(sed_row_line['eff_wavelength'].iloc[0], sed_row_line['flux'].iloc[0], color=colors[i], ls='None', marker='o', zorder=10, mec='black')
    for cont_filt in cont_filters:
        sed_row_cont = sed_df[sed_df['filter']==cont_filt]
        if plot_rest:
            ax.plot(sed_row_cont['rest_wave'].iloc[0], sed_row_cont['rest_flux'].iloc[0], color='blue', ls='None', marker='o', zorder=10, mec='black')
        else:
            ax.plot(sed_row_cont['eff_wavelength'].iloc[0], sed_row_cont['flux'].iloc[0], color='blue', ls='None', marker='o', zorder=10, mec='black')
    
    if ha_pab:
        ax.text(0.02, 0.95, f'Ha SNR = {ha_snr:0.2f}', transform=ax.transAxes, color=colors[0])
        ax.text(0.02, 0.90, f'PaB SNR = {pab_snr:0.2f}', transform=ax.transAxes, color=colors[1])
    if pab_paa:
        ax.text(0.02, 0.95, f'PaB SNR = {pab_snr:0.2f}', transform=ax.transAxes, color=colors[0])
        ax.text(0.02, 0.90, f'PaA SNR = {paa_snr:0.2f}', transform=ax.transAxes, color=colors[1])

    # line_red = Line2D([0], [0], color='red', marker='o',  ls='None', mec='black')
    # line_blue = Line2D([0], [0], color='blue', marker='o', ls='None', mec='black')
    # custom_lines = [line_red, line_blue]
    # custom_labels = ['Line', 'Continuum']
    # ax.legend(custom_lines, custom_labels, loc=4)
        
    if plot_median_model == True:
        scale_wave = 1.4 #um
        prospector_spec_df = add_rest_cols(prospector_spec_df, redshift)
        spec_val = get_spec_val(prospector_spec_df, scale_wave=scale_wave)
        median_model_df = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/median_prospector_model_{scale_wave}um.csv').to_pandas()
        median_model_df['rest_wave_aa'] = median_model_df['rest_wave_um'] * 10000
        median_model_df['wave_aa'] = median_model_df['rest_wave_aa']*(1+redshift)
        median_model_df['flux_jy'] = median_model_df['rest_flux_jy']/(1+redshift)
        median_model_df['flux_erg_aa'] = flux_jy_to_erg(median_model_df['flux_jy'], median_model_df['wave_aa'])

        # Scale the model to the current ID
        filt_dict, filters = unconver_read_filters()
        
        wavelength = median_model_df['wave_aa'].to_numpy()
        f_lambda = median_model_df['flux_erg_aa'].to_numpy()
        sed_abmag = observate.getSED(wavelength, f_lambda, filterlist=filters)
        sed_jy = 10**(-0.4*(sed_abmag-8.9))
        waves = [filters[i].wave_effective for i in range(len(filters))]
        int_spec_df = pd.DataFrame(zip(waves, sed_jy), columns=['wave_aa', 'flux_jy'])
        int_spec_df['rest_flux_jy'] = int_spec_df['flux_jy'] * (1+redshift)
        int_spec_df['rest_wave_aa'] = int_spec_df['wave_aa'] / (1+redshift)
        int_spec_df['rest_wave_um'] = int_spec_df['rest_wave_aa'] / 10000

        # Only matchings on wavelengths greater than the threshold
        match_idxs = sed_df['rest_wave'] > 1#um
        f1 = int_spec_df[match_idxs]['rest_flux_jy']
        f2 = sed_df[match_idxs]['rest_flux']
        a12 = np.sum(f1 * f2) / np.sum(f2**2)
        int_spec_df['rest_flux_jy_scaled'] = int_spec_df['rest_flux_jy'] * a12
        
        # b12 = np.sqrt(np.sum((f1 - a12 * f2)**2) / np.sum(f1**2))
        
        # spec_val_model = get_spec_val(median_model_df, scale_wave=scale_wave)
        # scale_factor = spec_val/spec_val_model
        # median_model_df['rest_flux_jy_scaled'] = scale_factor*median_model_df['rest_flux_jy']
        median_model_df['rest_flux_jy_scaled'] = median_model_df['rest_flux_jy']/a12
        # ax.plot(int_spec_df['rest_wave_um'], int_spec_df['rest_flux_jy'], color='purple', ls='None', marker='o', zorder=1)
        ax.plot(median_model_df['rest_wave_um'], median_model_df['rest_flux_jy_scaled'], color='purple', ls='-', marker='None', zorder=1)
        # ax.plot(median_model_df['rest_wave_um'], median_model_df['rest_flux_jy_scaled'], color='purple', ls='-', marker='None', zorder=1)



    if len(overplot_bump) > 0:
        scale_wave = 1.4 #um
        prospector_spec_df = add_rest_cols(prospector_spec_df, redshift)
        spec_val = get_spec_val(prospector_spec_df, scale_wave=scale_wave)
        for id_dr3_bump in overplot_bump:
            # Read in the prospector model
            prospector_spec_df_bump, prospector_sed_df_bump, mu = read_prospector(id_dr3_bump, id_dr3=True)
            phot_sample_row_bump = phot_sample_df[phot_sample_df['id'] == id_dr3_bump]
            redshift_bump = phot_sample_row_bump['z_50'].iloc[0]
            prospector_spec_df_bump = add_rest_cols(prospector_spec_df_bump, redshift_bump)

            # Scale the model to the current ID
            spec_val_bump = get_spec_val(prospector_spec_df_bump, scale_wave=scale_wave)
            scale_factor = spec_val/spec_val_bump
            prospector_spec_df_bump['rest_flux_jy_scaled'] = scale_factor*prospector_spec_df_bump['rest_flux_jy']
            prospector_spec_df_bump['flux_jy_scaled'] = prospector_spec_df_bump['rest_flux_jy_scaled'] / (1+redshift)
            prospector_spec_df_bump['wave_um_scaled'] = prospector_spec_df_bump['rest_wave_um'] * (1+redshift)

            # Plot
            if plot_rest:
                ax.plot(prospector_spec_df_bump['rest_wave_um'], prospector_spec_df_bump['rest_flux_jy_scaled'], color='purple', ls='-', marker='None', zorder=1, alpha=0.2)
            else:
                ax.plot(prospector_spec_df_bump['wave_um_scaled'], prospector_spec_df_bump['flux_jy_scaled'], color='black', ls='-', marker='None', zorder=1, alpha=0.2)


   
    if plot_rest:
        ax.set_xlim(0, 3) #um
        ax.set_ylim(0.95*np.min(sed_df['rest_flux']), 1.35*np.max(sed_df['rest_flux']))    
    else:
        ax.set_xlim(0, 5) #um
        ax.set_ylim(0.95*np.min(sed_df['flux']), 1.35*np.max(sed_df['flux']))    
    ax.set_xlabel(f'{wave_type} Wavelength (um)')
    ax.set_ylabel('Flux (Jy)')
    fig.savefig(f'/Users/brianlorenz/uncover/Figures/PHOT_sample/sed_diagnositcs_p1/{id_dr3}_sed.pdf')
    plt.close('all')
    pass
    
def add_rest_cols(df, redshift):
    df['rest_wave_um'] = df['wave_um'] / (1+redshift)
    df['rest_flux_jy'] = df['flux_jy'] * (1+redshift)
    return df

def get_spec_val(spec_df, scale_wave=1):
    """Scale val in um"""
    idx_1um = np.argmin(np.abs(spec_df['rest_wave_um'] - scale_wave))
    val_1um = spec_df['rest_flux_jy'].iloc[idx_1um]
    return val_1um

def plot_all_seds():
    ha_pab_list = [17757, 17758, 30052, 32180, 32181, 36076, 37784, 40135, 46831, 47758, 48104, 49020, 49712, 49932, 50707, 51980, 54343, 59550, 64780]
    pab_paa_list = [13130, 20686, 22045, 23395, 29959, 30351, 32536, 33247, 33588, 33775, 35090, 40504, 40522, 43970, 46261, 46855, 47958, 54239, 54240, 54614, 54674, 55357, 55594, 57422, 60576, 60577, 60973, 64472, 64786, 67410]

    phot_sample_df = ascii.read(phot_df_loc).to_pandas()


    for id_dr3 in ha_pab_list:
        plot_sed(id_dr3, phot_sample_df, ha_pab=True)
    for id_dr3 in pab_paa_list:
        plot_sed(id_dr3, phot_sample_df, pab_paa=True, plot_median_model=True)

def get_median_model(phot_sample_df, id_dr3_list):
    scale_wave = 1.4 #um
    scale_flux = 1e-6 #jy
    rest_waves = np.arange(0, 5, 0.01)
    interp_fluxes = []
    for id_dr3 in id_dr3_list:
        # Read in the prospector model
        prospector_spec_df_bump, prospector_sed_df_bump, mu = read_prospector(id_dr3, id_dr3=True)
        phot_sample_row_bump = phot_sample_df[phot_sample_df['id'] == id_dr3]
        redshift_bump = phot_sample_row_bump['z_50'].iloc[0]
        prospector_spec_df_bump = add_rest_cols(prospector_spec_df_bump, redshift_bump)

        # Scale the model to the current ID
        spec_val_bump = get_spec_val(prospector_spec_df_bump, scale_wave=scale_wave)
        scale_factor = scale_flux/spec_val_bump
        prospector_spec_df_bump['rest_flux_jy_scaled'] = scale_factor*prospector_spec_df_bump['rest_flux_jy']

        interp_trasm_func = interp1d(prospector_spec_df_bump['rest_wave_um'], prospector_spec_df_bump['rest_flux_jy_scaled'], kind='linear', fill_value=0, bounds_error=False)
        interp_rest_fluxes = interp_trasm_func(rest_waves)
        interp_fluxes.append(interp_rest_fluxes)
    median_flux = np.median(interp_fluxes, axis=0)
    median_model_df = pd.DataFrame(zip(rest_waves, median_flux), columns=['rest_wave_um', 'rest_flux_jy'])
    median_model_df.to_csv(f'/Users/brianlorenz/uncover/Data/generated_tables/median_prospector_model_{scale_wave}um.csv', index=False)

if __name__ == "__main__":
    # plot_all_seds()
    

    # gals_list = [17757, 17758, 30052, 32180, 32181, 36076, 37784, 40135, 46831, 47758, 48104, 49020, 49712, 49932, 50707, 51980, 54343, 59550, 64780, 13130, 22045, 23395, 29959, 30351, 32536, 33247, 33588, 33775, 35090, 40504, 40522, 43970, 46261, 46855, 47958, 54239, 54240, 54614, 54674, 55357, 55594, 57422, 60576, 60577, 60973, 64472, 64786, 67410]
    phot_sample_df = ascii.read(phot_df_loc).to_pandas()
    # plot_sed(13130, phot_sample_df=phot_sample_df, pab_paa=True, plot_median_model=True)
    # plot_sed(20686, phot_sample_df=phot_sample_df, pab_paa=True, overplot_bump=gals_list)
    # plot_sed(30351, phot_sample_df=phot_sample_df, pab_paa=True, overplot_bump=gals_list)

    # full_gals_list = [17757, 17758, 30052, 30351, 32180, 32181, 36076, 37784, 40135, 46831, 47758, 48104, 49020, 49712, 49932, 50707, 51980, 54343, 59550, 64780, 13130, 22045, 23395, 29959, 30351, 32536, 33247, 33588, 33775, 35090, 40504, 40522, 43970, 46261, 46855, 47958, 54239, 54240, 54614, 54674, 55357, 55594, 57422, 60576, 60577, 60973, 64472, 64786, 67410]
    # get_median_model(phot_sample_df, full_gals_list)
    # plot_sed(37182, phot_sample_df=phot_sample_df, pab_paa=False, plot_median_model=False, ha_pab=True)

    project_1_ids = np.array([30052, 30804, 31608, 37182, 37776, 44283, 46339, 47771, 49023, 52140, 52257, 54625, 60579, 62937])
    for id in project_1_ids:
        plot_sed(id, phot_sample_df=phot_sample_df, pab_paa=False, plot_median_model=False, ha_pab=True)
