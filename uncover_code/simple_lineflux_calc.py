from uncover_make_sed import read_sed
from simple_make_dustmap import make_3color, compute_cont_pct, compute_line
from filter_integrals import get_transmission_at_line
from uncover_read_data import read_spec_cat, read_lineflux_cat, get_id_msa_list
from sedpy import observate
from fit_emission_uncover_old import line_list
from astropy.io import ascii
import numpy as np
import pandas as pd
from compute_av import get_nii_correction, get_fe_correction

def calc_lineflux(id_msa, fluxcal_str=''):
    sed_df = read_sed(id_msa)
    mock_sed_df = ascii.read('/Users/brianlorenz/uncover/Data/integrated_specs/mock_ratio_15_flux_32_flat_shifted_47875.csv').to_pandas()
    sed_df = sed_df.join(mock_sed_df)
    zqual_df = read_spec_cat()
    redshift = zqual_df[zqual_df['id_msa']==id_msa]['z_spec'].iloc[0]

    ha_filters, ha_images, wht_ha_images, obj_segmap, ha_photfnus, ha_all_lines = make_3color(id_msa, line_index=0, plot=False)
    pab_filters, pab_images, wht_pab_images, obj_segmap, pab_photfnus, pab_all_lines = make_3color(id_msa, line_index=1, plot=False)
    ha_sedpy_name = ha_filters[1].replace('f', 'jwst_f')
    ha_sedpy_filt = observate.load_filters([ha_sedpy_name])[0]
    pab_sedpy_name = pab_filters[1].replace('f', 'jwst_f')
    pab_sedpy_filt = observate.load_filters([pab_sedpy_name])[0]

    ha_red_sedpy_name = ha_filters[0].replace('f', 'jwst_f')
    ha_red_sedpy_filt = observate.load_filters([ha_red_sedpy_name])[0]
    pab_red_sedpy_name = pab_filters[0].replace('f', 'jwst_f')
    pab_red_sedpy_filt = observate.load_filters([pab_red_sedpy_name])[0]
    ha_blue_sedpy_name = ha_filters[2].replace('f', 'jwst_f')
    ha_blue_sedpy_filt = observate.load_filters([ha_blue_sedpy_name])[0]
    pab_blue_sedpy_name = pab_filters[2].replace('f', 'jwst_f')
    pab_blue_sedpy_filt = observate.load_filters([pab_blue_sedpy_name])[0]

    ha_sedpy_filts = [ha_red_sedpy_filt, ha_sedpy_filt, ha_blue_sedpy_filt]
    pab_sedpy_filts = [pab_red_sedpy_filt, pab_sedpy_filt, pab_blue_sedpy_filt]

    ha_sedpy_width = ha_sedpy_filt.rectangular_width
    pab_sedpy_width = pab_sedpy_filt.rectangular_width
    ha_sedpy_wave = ha_sedpy_filt.wave_effective
    pab_sedpy_wave = pab_sedpy_filt.wave_effective
    ha_sedpy_transmission = get_transmission_at_line(ha_sedpy_filt, line_list[0][1] * (1+redshift), trasm_type='raw')
    pab_sedpy_transmission = get_transmission_at_line(pab_sedpy_filt, line_list[1][1] * (1+redshift), trasm_type='raw')

    ha_filters = ['f_'+filt for filt in ha_filters]
    pab_filters = ['f_'+filt for filt in pab_filters]

    ha_line_scaled_transmission = get_transmission_at_line(ha_sedpy_filt, line_list[0][1] * (1+redshift))
    pab_line_scaled_transmission = get_transmission_at_line(pab_sedpy_filt, line_list[1][1] * (1+redshift))

    ha_flux_erg_s_cm2 = measure_lineflux(id_msa, sed_df, redshift, ha_filters, ha_line_scaled_transmission, ha_sedpy_transmission, line_list[0][1], ha_sedpy_width, ha_sedpy_wave, ha_sedpy_filts)
    pab_flux_erg_s_cm2 = measure_lineflux(id_msa, sed_df, redshift, pab_filters, pab_line_scaled_transmission, pab_sedpy_transmission, line_list[1][1], pab_sedpy_width, pab_sedpy_wave, pab_sedpy_filts)
    # Line flux corrections
    nii_cor_ha_flux_erg_s_cm2 = ha_flux_erg_s_cm2*get_nii_correction(id_msa)
    fe_cor_pab_flux_erg_s_cm2 = pab_flux_erg_s_cm2*get_fe_correction(id_msa)

    lines_df = read_lineflux_cat()
    lines_df_row = lines_df[lines_df['id_msa'] == id_msa]
    ha_flux_cat_erg_s_cm2 = lines_df_row['f_Ha+NII'].iloc[0] * 1e-20
    pab_flux_cat_erg_s_cm2 = lines_df_row['f_PaB'].iloc[0] * 1e-20
    err_ha_flux_cat_erg_s_cm2 = lines_df_row['e_Ha+NII'].iloc[0] * 1e-20
    err_pab_flux_cat_erg_s_cm2 = lines_df_row['e_PaB'].iloc[0] * 1e-20

    fit_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting{fluxcal_str}/{id_msa}_emission_fits.csv').to_pandas()
    ha_flux_emission_fit = fit_df['flux'].iloc[0]
    pab_flux_emission_fit = fit_df['flux'].iloc[1]
    nii_cor_ha_flux_emission_fit = fit_df['nii_cor_flux'].iloc[0]
    # print(f'ha width = {fit_df["sigma"].iloc[0]}')
    # print(f'pab width = {fit_df["sigma"].iloc[1]}')

    ha_offset_factor = ha_flux_erg_s_cm2 / ha_flux_emission_fit
    pab_offset_factor = fe_cor_pab_flux_erg_s_cm2 / pab_flux_emission_fit

    print(f'All erg/s/cm^2')
    print(f'Ha+Nii flux: {ha_flux_erg_s_cm2}')
    print(f'Ha+Nii flux fit compare: {ha_offset_factor}')
    print(f'PaB flux corrected for Fe: {fe_cor_pab_flux_erg_s_cm2}')
    print(f'PaB flux fit compare: {pab_offset_factor}')

    return ha_flux_erg_s_cm2, pab_flux_erg_s_cm2, ha_flux_cat_erg_s_cm2, err_ha_flux_cat_erg_s_cm2, pab_flux_cat_erg_s_cm2, err_pab_flux_cat_erg_s_cm2, ha_flux_emission_fit, nii_cor_ha_flux_emission_fit, pab_flux_emission_fit, nii_cor_ha_flux_erg_s_cm2, fe_cor_pab_flux_erg_s_cm2

    # print(f'Ha/PaB: {(ha_flux/pab_flux) / (line_list[0][1]/line_list[1][1])**2}')
    # print(f'Cat Ha/PaB: {(ha_flux_cat_jy/pab_flux_cat_jy)}')

def measure_lineflux(id_msa, sed_df, redshift, filters, scaled_transmission, raw_transmission, line_wave, filter_width, filter_wave, line_filts):
    for i in range(len(filters)):
        sed_row = sed_df[sed_df['filter'] == filters[i]]
        if i==0:
            red_wave = sed_row['eff_wavelength'].iloc[0]
            # red_flux = sed_row['integrated_spec_flux_jy'].iloc[0]
            red_flux = sed_row['flux'].iloc[0] # jy
            red_flux_erg_s_cm2 = compute_filter_F(red_flux, line_filts[0])
        if i == 1:
            green_wave = sed_row['eff_wavelength'].iloc[0]
            # green_flux = sed_row['integrated_spec_flux_jy'].iloc[0]
            green_flux = sed_row['flux'].iloc[0]
            green_flux_erg_s_cm2 = compute_filter_F(green_flux, line_filts[1])
        if i == 2:
            blue_wave = sed_row['eff_wavelength'].iloc[0]
            # blue_flux = sed_row['integrated_spec_flux_jy'].iloc[0]
            blue_flux = sed_row['flux'].iloc[0]
            blue_flux_erg_s_cm2 = compute_filter_F(blue_flux, line_filts[2])
    def cor_he1(id_msa, blue_flux):
        he1_emfit_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting/helium/{id_msa}_emission_fits_helium1.csv').to_pandas()
        he1_flux = he1_emfit_df['flux'].iloc[1]
        he1_wave = 10830
        c = 299792458 # m/s
        he1_flux_fnu = he1_flux / ((c*1e10) / (he1_wave)**2) 
        he1_flux_jy = he1_flux_fnu / 1e-23
        he1_flux_filt_spread = he1_flux_jy / filter_width
        blue_flux = blue_flux - he1_flux_filt_spread  # THis is only about a 2% correction to total pab flux
        return blue_flux
    # blue_flux = cor_he1(id_msa, blue_flux)
    cont_percentile = compute_cont_pct(blue_wave, green_wave, red_wave, blue_flux, red_flux)
    print(blue_flux)
    print(green_flux)
    print(red_flux)
    # cont_percentile2 = compute_cont_pct(blue_wave, green_wave, red_wave, blue_flux_erg_s_cm2, red_flux_erg_s_cm2)
    line_flux, cont_value = compute_line(cont_percentile, red_flux, green_flux, blue_flux, redshift, raw_transmission, filter_width, line_wave)
    if line_wave < 8000:
        line_name = 'ha'
    if line_wave > 8000:
        line_name = 'pab'
    # line_transmission = get_line_coverage(id_msa, line_filts[1], redshift=redshift, line_name=line_name)
    # line_flux = line_flux 
    

    return line_flux


def compute_line_already_erg(cont_pct, red_flx, green_flx, blue_flx, redshift, scaled_transmission, raw_transmission, filter_width, filter_wave, line_wave):
        cont_value = np.percentile([blue_flx, red_flx], cont_pct*100)

        line_value = green_flx - cont_value # erg/s/cm^2

        # Scale by raw transmission curve
        line_value = line_value / raw_transmission

        return line_value, cont_value




def calc_all_lineflux(id_msa_list, full_sample=False, fluxcal=True):
    if fluxcal:
        fluxcal_str = ''
    else:
        fluxcal_str = '_no_fluxcal'

    ha_sed_fluxes = []
    pab_sed_fluxes = []
    ha_cat_fluxes = []
    err_ha_cat_fluxes = []
    pab_cat_fluxes = []
    err_pab_cat_fluxes = []
    ha_emission_fits = []
    nii_cor_ha_flux_emission_fits = []
    pab_emission_fits = []
    nii_cor_ha_flux_erg_s_cm2s = []
    fe_cor_pab_flux_erg_s_cm2s = []
    for id_msa in id_msa_list:
        ha_flux_erg_s_cm2, pab_flux_erg_s_cm2, ha_flux_cat_erg_s_cm2, err_ha_flux_cat_erg_s_cm2, pab_flux_cat_erg_s_cm2, err_pab_flux_cat_erg_s_cm2, ha_flux_emission_fit, nii_cor_ha_flux_emission_fit, pab_flux_emission_fit, nii_cor_ha_flux_erg_s_cm2, fe_cor_pab_flux_erg_s_cm2 = calc_lineflux(id_msa, fluxcal_str=fluxcal_str)
        ha_sed_fluxes.append(ha_flux_erg_s_cm2)
        pab_sed_fluxes.append(pab_flux_erg_s_cm2)
        ha_cat_fluxes.append(ha_flux_cat_erg_s_cm2)
        err_ha_cat_fluxes.append(err_ha_flux_cat_erg_s_cm2)
        err_pab_cat_fluxes.append(err_pab_flux_cat_erg_s_cm2)
        pab_cat_fluxes.append(pab_flux_cat_erg_s_cm2)
        ha_emission_fits.append(ha_flux_emission_fit)
        nii_cor_ha_flux_emission_fits.append(nii_cor_ha_flux_emission_fit)
        pab_emission_fits.append(pab_flux_emission_fit)
        nii_cor_ha_flux_erg_s_cm2s.append(nii_cor_ha_flux_erg_s_cm2)
        fe_cor_pab_flux_erg_s_cm2s.append(fe_cor_pab_flux_erg_s_cm2)
    emission_offset_df = pd.DataFrame(zip(id_msa_list, ha_sed_fluxes, pab_sed_fluxes, nii_cor_ha_flux_erg_s_cm2s, fe_cor_pab_flux_erg_s_cm2s, ha_cat_fluxes, err_ha_cat_fluxes, pab_cat_fluxes, err_pab_cat_fluxes, ha_emission_fits, nii_cor_ha_flux_emission_fits, pab_emission_fits), columns=['id_msa', 'ha_sed_flux', 'pab_sed_flux', 'nii_cor_ha_sed_flux', 'fe_cor_pab_sed_flux', 'ha_cat_flux', 'err_ha_cat_flux', 'pab_cat_flux', 'err_pab_cat_flux', 'ha_emfit_flux', 'nii_cor_ha_emfit_flux', 'pab_emfit_flux'])
    emission_offset_df['ha_sed_div_cat'] = emission_offset_df['ha_sed_flux'] / emission_offset_df['ha_cat_flux']
    emission_offset_df['pab_sed_div_cat'] = emission_offset_df['fe_cor_pab_sed_flux'] / emission_offset_df['pab_cat_flux']
    emission_offset_df['ha_sed_div_emfit'] = emission_offset_df['nii_cor_ha_sed_flux'] / emission_offset_df['nii_cor_ha_emfit_flux']
    emission_offset_df['pab_sed_div_emfit'] = emission_offset_df['fe_cor_pab_sed_flux'] / emission_offset_df['pab_emfit_flux']
    emission_offset_df['ha_cat_div_emfit'] = emission_offset_df['ha_cat_flux'] / emission_offset_df['ha_emfit_flux']
    emission_offset_df['pab_cat_div_emfit'] = emission_offset_df['pab_cat_flux'] / emission_offset_df['pab_emfit_flux']
    emission_offset_df['difference_in_offset_ratio_cat'] = emission_offset_df['pab_sed_div_cat'] / emission_offset_df['ha_sed_div_cat']
    emission_offset_df['difference_in_offset_ratio_emfit'] = emission_offset_df['pab_sed_div_emfit'] / emission_offset_df['ha_sed_div_emfit']
    
    if full_sample == False:
        emission_offset_df.to_csv(f'/Users/brianlorenz/uncover/Data/generated_tables/lineflux_df{fluxcal_str}.csv', index=False)
    if full_sample == True:
        emission_offset_df.to_csv(f'/Users/brianlorenz/uncover/Data/generated_tables/lineflux_df_all.csv{fluxcal_str}', index=False)
    print(f'\n\n\n')
    print(f'median offset in Ha {np.median(emission_offset_df["ha_sed_div_emfit"])}')
    print(f'median offset in PaB {np.median(emission_offset_df["pab_sed_div_emfit"])}')
    print(f'median PaB offset / Ha offset {np.median(emission_offset_df["difference_in_offset_ratio_emfit"])}')

def compute_filter_F(f_nu_jy, sedpy_filt):
    filter_wave = sedpy_filt.wave_effective
    filter_width = sedpy_filt.rectangular_width
    median_filter_trasm = np.median(sedpy_filt.transmission)
    # filter_area = median_filter_trasm * filter_width

    f_nu = f_nu_jy * 1e-23

    c = 299792458 # m/s
    f_lambda_aa = f_nu * ((c*1e10) / (filter_wave)**2) 
    f_total = f_lambda_aa * filter_width  # erg/s/cm^2
    return f_total

if __name__ == "__main__":
    # calc_lineflux(47875)
    # calc_lineflux(14573)
    # calc_lineflux(25774)
    # calc_lineflux(39855)
    
    # id_msa_list = get_id_msa_list(full_sample=True)
    # calc_all_lineflux(id_msa_list, full_sample=True)  

    id_msa_list = get_id_msa_list(full_sample=False)
    calc_all_lineflux(id_msa_list, full_sample=False, fluxcal=True)  
    pass