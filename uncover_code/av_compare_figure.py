import matplotlib.pyplot as plt
from astropy.io import ascii
import numpy as np
from compute_av import ha_factor, pab_factor, compute_ratio_from_av, compute_ha_pab_av, compute_ha_pab_av_from_dustmap, read_catalog_av
from uncover_read_data import read_supercat, read_raw_spec, read_spec_cat, read_segmap, read_SPS_cat


def generate_filtered_lineratio_df():
    # Read in the data
    lineratio_df = ascii.read('/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/lineratio_df.csv').to_pandas()
    zqual_df_cont_covered = ascii.read('/Users/brianlorenz/uncover/zqual_df_cont_covered.csv').to_pandas()
    good_rows = np.logical_and(zqual_df_cont_covered['ha_trasm_flag'] == 0, zqual_df_cont_covered['pab_trasm_flag'] == 0)
    good_ids = zqual_df_cont_covered[good_rows]['id_msa']
    filtered_linartio_df = lineratio_df[lineratio_df['id_msa'].isin(good_ids)]

    # Read in prospector fits
    zqual_df = read_spec_cat()
    av_16s = []
    av_50s = []
    av_84s = []
    for i in range(len(filtered_linartio_df)):
        id_msa = filtered_linartio_df.iloc[i]['id_msa']
        av_16, av_50, av_84 = read_catalog_av(id_msa, zqual_df)
        av_16s.append(av_16)
        av_50s.append(av_50)
        av_84s.append(av_84)
    filtered_linartio_df['av_16'] = av_16s
    filtered_linartio_df['av_50'] = av_50s
    filtered_linartio_df['av_84'] = av_84s


    # Convert the line ratios to av
    sed_av = compute_ha_pab_av(1/filtered_linartio_df['sed_lineratio'])
    sed_av_16 = compute_ha_pab_av(1/filtered_linartio_df['sed_lineratio_16'])
    sed_av_84 = compute_ha_pab_av(1/filtered_linartio_df['sed_lineratio_84'])
    emission_av = compute_ha_pab_av(1/filtered_linartio_df['emission_fit_lineratio'])
    emission_av_high = compute_ha_pab_av(1/(filtered_linartio_df['emission_fit_lineratio']-filtered_linartio_df['err_emission_fit_lineratio_low']))
    emission_av_low = compute_ha_pab_av(1/(filtered_linartio_df['err_emission_fit_lineratio_high']+filtered_linartio_df['emission_fit_lineratio']))

    filtered_linartio_df['sed_av'] = sed_av
    filtered_linartio_df['sed_av_16'] = sed_av_16
    filtered_linartio_df['sed_av_84'] = sed_av_84
    filtered_linartio_df['emission_fit_av'] = emission_av
    filtered_linartio_df['emission_fit_av_low'] = emission_av_low
    filtered_linartio_df['emission_fit_av_high'] = emission_av_high

    filtered_linartio_df.to_csv('/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/filtered_lineratio_df.csv', index=False)
    return 

def make_av_compare_figure(regenerate = False):
    if regenerate == True:
        generate_filtered_lineratio_df()

    filtered_lineratio_df = ascii.read('/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/filtered_lineratio_df.csv').to_pandas()
    

    # Setup plots
    fig, axarr = plt.subplots(1, 2, figsize=(12,6))
    ax_prospector = axarr[0]
    ax_sed = axarr[1]

    y_var = 'av_50'
    for i in range(len(filtered_lineratio_df)):
        row = filtered_lineratio_df.iloc[i]
        prospector_av_err=np.array([[row[y_var]-row[y_var.replace('50','16')], row[y_var.replace('50','84')]-row[y_var]]]).T
        emission_av_err=np.array([[row['emission_fit_av']-row['emission_fit_av_low'], row['emission_fit_av_high']-row['emission_fit_av']]]).T

        ax_prospector.errorbar(row['emission_fit_av'], row['av_50'], xerr=emission_av_err, yerr=prospector_av_err, ls='None', marker='o', color='black')

        sed_av_err=np.array([[row['sed_av']-row['sed_av_16'], row['sed_av_84']-row['sed_av']]]).T
        ax_sed.errorbar(row['emission_fit_av'], row['sed_av'], xerr=emission_av_err, yerr=sed_av_err, ls='None', marker='o', color='black')

    for ax in axarr:
        ax.set_xlabel('Emission Fit A$_V$')
        ax.tick_params(labelsize=12)
        ax.plot([-100, 100], [-100, 100], ls='--', color='red', marker='None')
        ax.set_xlim(-1,2)

    ax_prospector.set_ylim(-1,2)
    ax_prospector.set_ylabel('Prospector Fit A$_V$')
    
    ax_sed.set_ylim(-1,3.5)
    ax_sed.set_ylabel('SED A$_V$')

    fig.savefig('/Users/brianlorenz/uncover/Figures/paper_figures/av_comparison.pdf')
    
    
    return

# generate_filtered_lineratio_df()
make_av_compare_figure(regenerate=False)