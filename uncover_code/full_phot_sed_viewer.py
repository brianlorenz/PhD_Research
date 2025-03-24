from uncover_make_sed import read_full_phot_sed
from astropy.io import ascii
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from full_phot_read_data import read_merged_lineflux_cat
from uncover_prospector_seds import read_prospector


phot_df_loc = '/Users/brianlorenz/uncover/Data/generated_tables/phot_calcs/phot_linecoverage_ha_pab_paa.csv'
colors = ['red', 'mediumseagreen']

def plot_sed(id_dr3, phot_sample_df, ha_pab=False, pab_paa=False):
    sed_df = read_full_phot_sed(id_dr3)
    lineflux_df = read_merged_lineflux_cat()
    prospector_spec_df, prospector_sed_df = read_prospector(id_msa)

    phot_sample_row = phot_sample_df[phot_sample_df['id'] == id_dr3]
    lineflux_row = lineflux_df[lineflux_df['id_dr3']==id_dr3]
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
    ax.errorbar(sed_df['eff_wavelength'], sed_df['flux'], yerr=sed_df['err_flux'], color='black', ecolor='grey', ls='None', marker='o', zorder=1)
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
    ax.set_xlabel('Wavelength (um)')
    ax.set_ylabel('Flux (Jy)')
    fig.savefig(f'/Users/brianlorenz/uncover/Figures/PHOT_sample/sed_diagnositcs/{id_dr3}_sed.pdf')
    pass
    
    
def plot_all_seds():
    ha_pab_list = [17757, 17758, 30052, 32180, 32181, 36076, 37784, 40135, 46831, 47758, 48104, 49020, 49712, 49932, 50707, 51980, 54343, 59550, 64780]
    pab_paa_list = [13130, 20686, 22045, 23395, 29959, 30351, 32536, 33247, 33588, 33775, 35090, 40504, 40522, 43970, 46261, 46855, 47958, 54239, 54240, 54614, 54674, 55357, 55594, 57422, 60576, 60577, 60973, 64472, 64786, 67410]

    phot_sample_df = ascii.read(phot_df_loc).to_pandas()


    for id_dr3 in ha_pab_list:
        plot_sed(id_dr3, phot_sample_df, ha_pab=True)
    for id_dr3 in pab_paa_list:
        plot_sed(id_dr3, phot_sample_df, pab_paa=True)


if __name__ == "__main__":
    plot_all_seds()