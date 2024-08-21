from uncover_read_data import read_supercat, read_raw_spec, read_spec_cat, read_segmap, read_SPS_cat
from uncover_make_sed import get_sed
from uncover_sed_filters import unconver_read_filters
from sedpy import observate
from make_dust_maps import make_3color
import matplotlib.pyplot as plt
import numpy as np

def measure_feature(id_msa):
    spec_df = read_raw_spec(id_msa)
    sed_df = get_sed(id_msa)

    ha_filters, ha_images, wht_ha_images, obj_segmap = make_3color(id_msa, line_index=0, plot=False)
    pab_filters, pab_images, wht_pab_images, obj_segmap = make_3color(id_msa, line_index=1, plot=False)

    ha_filters = ['f_'+filt for filt in ha_filters]
    pab_filters = ['f_'+filt for filt in pab_filters]

    wavelength = spec_df['wave_aa'].to_numpy()
    f_lambda = spec_df['flux_erg_aa'].to_numpy()
    f_jy = spec_df['flux'].to_numpy()
    filt_dict, filters = unconver_read_filters()
    filter_names = [sedpy_filt.name for sedpy_filt in filters]
    integrated_sed_abmag = observate.getSED(wavelength, f_lambda, filterlist=filters)
    integrated_sed_jy = 10**(-0.4*(integrated_sed_abmag-8.9))
    effective_waves_aa = sed_df['eff_wavelength']*10000

    ha_idxs = []
    pab_idxs = []
    for ha_filt in ha_filters:
        ha_filt = ha_filt[2:]
        for index, item in enumerate(filter_names):
            if ha_filt in item:
                ha_idxs.append(index)
    for pab_filt in pab_filters:
        pab_filt = pab_filt[2:]
        for index, item in enumerate(filter_names):
            if pab_filt in item:
                pab_idxs.append(index)

    wave_idxs = np.logical_and(wavelength>35000, wavelength<45000)
    feature_idxs = np.logical_and(wavelength>40300, wavelength<43000)
    f_jy_cor = f_jy.copy()
    f_jy_cor[feature_idxs] = 1.06*f_jy_cor[feature_idxs]
    f_lambda_cor = f_lambda.copy()
    f_lambda_cor[feature_idxs] = 1.06*f_lambda_cor[feature_idxs]
    
    # breakpoint()
    integrated_sed_abmag_cor = observate.getSED(wavelength, f_lambda_cor, filterlist=filters)
    integrated_sed_jy_cor = 10**(-0.4*(integrated_sed_abmag_cor-8.9))

    effect_pct = integrated_sed_jy_cor[pab_idxs[0]]/integrated_sed_jy[pab_idxs[0]]

    plt.plot(wavelength[wave_idxs], f_jy[wave_idxs], color='red', label='original')
    plt.plot(wavelength[wave_idxs], f_jy_cor[wave_idxs], color='black', alpha = 0.8, label='corrected')
    plt.title(f'Increase in pab red flux: {effect_pct}')

    plt.savefig(f'/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/pab_red_cor{id_msa}.pdf')

# measure_feature(25147)
measure_feature(47875)