from uncover_make_sed import read_full_phot_sed
from astropy.io import ascii
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from full_phot_read_data import read_merged_lineflux_cat
from uncover_prospector_seds import read_prospector, make_prospector
import numpy as np


phot_df_loc = '/Users/brianlorenz/uncover/Data/generated_tables/phot_calcs/phot_linecoverage_ha_pab_paa.csv'
colors = ['red', 'mediumseagreen']

def plot_sed(id_dr3, phot_sample_df, ha_pab=False, pab_paa=False, overplot_bump=[]):
    sed_df = read_full_phot_sed(id_dr3)
    lineflux_df = read_merged_lineflux_cat()
    prospector_spec_df, prospector_sed_df = read_prospector(id_dr3, id_dr3=True)

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

    fig, ax = plt.subplots(figsize=(6,6))
    ax.errorbar(sed_df['eff_wavelength'], sed_df['flux'], yerr=sed_df['err_flux'], color='black', ecolor='grey', ls='None', marker='o', zorder=5)
    ax.plot(prospector_spec_df['wave_um'], prospector_spec_df['flux_jy'], color='orange', ls='-', marker='None', zorder=5)
    
    for i in range(len(line_filters)):
        line_filt = line_filters[i]
        sed_row_line = sed_df[sed_df['filter']==line_filt]
        ax.plot(sed_row_line['eff_wavelength'].iloc[0], sed_row_line['flux'].iloc[0], color=colors[i], ls='None', marker='o', zorder=10, mec='black')
    for cont_filt in cont_filters:
        sed_row_cont = sed_df[sed_df['filter']==cont_filt]
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
        
    if len(overplot_bump) > 0:
        scale_wave = 1.4 #um
        prospector_spec_df = add_rest_cols(prospector_spec_df, redshift)
        spec_val = get_spec_val(prospector_spec_df, scale_wave=scale_wave)
        for id_dr3_bump in overplot_bump:
            # Read in the prospector model
            prospector_spec_df_bump, prospector_sed_df_bump = read_prospector(id_dr3_bump, id_dr3=True)
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
            ax.plot(prospector_spec_df_bump['wave_um_scaled'], prospector_spec_df_bump['flux_jy_scaled'], color='black', ls='-', marker='None', zorder=1, alpha=0.2)


    ax.set_xlim(0, 5) #um
    ax.set_ylim(0.95*np.min(sed_df['flux']), 1.35*np.max(sed_df['flux']))    
    ax.set_xlabel('Observed Wavelength (um)')
    ax.set_ylabel('Flux (Jy)')
    fig.savefig(f'/Users/brianlorenz/uncover/Figures/PHOT_sample/sed_diagnositcs/{id_dr3}_sed.pdf')
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
        plot_sed(id_dr3, phot_sample_df, pab_paa=True)


if __name__ == "__main__":
    # plot_all_seds()

    gals_list = [17757, 17758, 30052, 32180, 32181, 36076, 37784, 40135, 46831, 47758, 48104, 49020, 49712, 49932, 50707, 51980, 54343, 59550, 64780, 13130, 22045, 23395, 29959, 30351, 32536, 33247, 33588, 33775, 35090, 40504, 40522, 43970, 46261, 46855, 47958, 54239, 54240, 54614, 54674, 55357, 55594, 57422, 60576, 60577, 60973, 64472, 64786, 67410]
    phot_sample_df = ascii.read(phot_df_loc).to_pandas()
    plot_sed(20686, phot_sample_df=phot_sample_df, pab_paa=True, overplot_bump=gals_list)