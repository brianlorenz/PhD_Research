from full_phot_read_data import read_lineflux_cat
from compute_av import get_nii_correction_dr3
from uncover_read_data import read_SPS_cat_all
from astropy.io import ascii


def apply_nii_cor():
    halpha_df, halpha_df_loc = read_lineflux_cat('Halpha')
    sps_df = read_SPS_cat_all()
    nii_cor_values = [get_nii_correction_dr3(id_dr3, sps_df=sps_df) for id_dr3 in halpha_df['id_dr3']]
    halpha_df['nii_fraction'] = nii_cor_values
    halpha_df['nii_cor_Halpha_flux'] = halpha_df['Halpha_flux'] * halpha_df['nii_fraction']
    halpha_df['err_nii_cor_Halpha_flux_low'] = halpha_df['err_Halpha_flux_low'] * halpha_df['nii_fraction']
    halpha_df['err_nii_cor_Halpha_flux_high'] = halpha_df['err_Halpha_flux_high'] * halpha_df['nii_fraction']
    halpha_df.to_csv(halpha_df_loc, index=False)

def apply_fe_cor():
    pabeta_df, pabeta_df_loc = read_lineflux_cat('PaBeta')
    pab_correction_factor = get_fe_correction()
    pabeta_df['pab_correction_factor'] = [pab_correction_factor for i in range(len(pabeta_df))]
    pabeta_df['fe_cor_PaBeta_flux'] = pabeta_df['PaBeta_flux'] * pabeta_df['pab_correction_factor']
    pabeta_df['err_fe_cor_PaBeta_flux_low'] = pabeta_df['err_PaBeta_flux_low'] * pabeta_df['pab_correction_factor']
    pabeta_df['err_fe_cor_PaBeta_flux_high'] = pabeta_df['err_PaBeta_flux_high'] * pabeta_df['pab_correction_factor']
    pabeta_df.to_csv(pabeta_df_loc, index=False)


def get_fe_correction():
    fe_cor_df_indiv = ascii.read('/Users/brianlorenz/uncover/Data/generated_tables/fe_cor_df_indiv.csv').to_pandas()
    predicted_fe_pab_ratio = fe_cor_df_indiv['median_fe_pab_ratios'].iloc[0]
    pab_correction_factor = 1 / (1+predicted_fe_pab_ratio)
    return pab_correction_factor

if __name__ == '__main__':
    # apply_nii_cor()
    apply_fe_cor()
    
    # breakpoint()
    pass