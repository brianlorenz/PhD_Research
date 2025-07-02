from full_phot_read_data import read_lineflux_cat
from compute_av import get_nii_correction_dr3
from uncover_read_data import read_SPS_cat_all
from astropy.io import ascii
from fit_emission_uncover_wave_divide import line_list


def apply_nii_cor():
    print('Adding NII correction')
    halpha_df, halpha_df_loc = read_lineflux_cat('Halpha')
    sps_df = read_SPS_cat_all()
    nii_cor_values = [get_nii_correction_dr3(id_dr3, sps_df=sps_df) for id_dr3 in halpha_df['id_dr3']]
    halpha_df['nii_fraction'] = nii_cor_values
    halpha_df['nii_cor_Halpha_flux'] = halpha_df['Halpha_flux'] * halpha_df['nii_fraction']
    halpha_df['err_nii_cor_Halpha_flux_low'] = halpha_df['err_Halpha_flux_low'] * halpha_df['nii_fraction']
    halpha_df['err_nii_cor_Halpha_flux_high'] = halpha_df['err_Halpha_flux_high'] * halpha_df['nii_fraction']
    halpha_df.to_csv(halpha_df_loc, index=False)

    # Add eqw
    add_ew_column('Halpha')
    
    

def add_eq_width(line_rest_wave, redshift, cont_flux_jy, line_flux_erg):
    cont_value_erg_hz = cont_flux_jy * 1e-23
    # # Convert from f_nu to f_lambda
    c = 299792458 # m/s
    observed_wave = line_rest_wave * (1+redshift)
    cont_flux_erg_s_cm2_aa = (1+redshift) * cont_value_erg_hz * ((c*1e10) / (observed_wave)**2) # erg/s/cm2/angstrom
    eq_width = line_flux_erg / cont_flux_erg_s_cm2_aa
    return eq_width

def apply_fe_cor():
    print('Adding Fe correction')
    sps_df = read_SPS_cat_all()
    pabeta_df, pabeta_df_loc = read_lineflux_cat('PaBeta')
    pab_correction_factor = get_fe_correction()
    pabeta_df['pab_correction_factor'] = [pab_correction_factor for i in range(len(pabeta_df))]
    pabeta_df['fe_cor_PaBeta_flux'] = pabeta_df['PaBeta_flux'] * pabeta_df['pab_correction_factor']
    pabeta_df['err_fe_cor_PaBeta_flux_low'] = pabeta_df['err_PaBeta_flux_low'] * pabeta_df['pab_correction_factor']
    pabeta_df['err_fe_cor_PaBeta_flux_high'] = pabeta_df['err_PaBeta_flux_high'] * pabeta_df['pab_correction_factor']
    pabeta_df.to_csv(pabeta_df_loc, index=False)

    # Add eqw
    add_ew_column('PaBeta')
    


def add_ew_column(line_name):
    """
    line_name - 'PaAlpha' 'PaBeta' or 'Halpha'
    """
    df, df_loc = read_lineflux_cat(line_name)
    sps_df = read_SPS_cat_all()
    # Add eqw
    sps_rows = df.merge(sps_df, left_on='id_dr3', right_on='id', how='inner')
    line_names = [line_list[i][0] for i in range(len(line_list))]
    line_index = line_names.index(line_name)
    line_rest_waves = [line_list[line_index][1] for i in range(len(df))]
    sps_rows[f'{line_name}_rest'] = line_rest_waves
    eqw = add_eq_width(sps_rows[f'{line_name}_rest'], sps_rows['z_50'], df[f'{line_name}_cont_value'], df[f'{line_name}_flux'])
    df[f'{line_name}_eqw'] = eqw
    df.to_csv(df_loc, index=False)

def get_fe_correction():
    fe_cor_df_indiv = ascii.read('/Users/brianlorenz/uncover/Data/generated_tables/fe_cor_df_indiv.csv').to_pandas()
    predicted_fe_pab_ratio = fe_cor_df_indiv['median_fe_pab_ratios'].iloc[0]
    pab_correction_factor = 1 / (1+predicted_fe_pab_ratio)
    return pab_correction_factor

if __name__ == '__main__':
    # add_ew_column('PaAlpha')
    # apply_nii_cor()
    # apply_fe_cor()
    
    # breakpoint()
    pass