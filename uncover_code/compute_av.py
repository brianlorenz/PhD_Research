import numpy as np
from fit_emission_uncover import line_list
from uncover_read_data import read_SPS_cat
from dust_equations_prospector import dust2_to_AV

# theoretical scalings (to Hb, from naveen's paper)
ha_factor = 2.79
hb_factor = 1
pab_factor = 0.155

ha_wave = line_list[0][1]/10000
pab_wave = line_list[1][1]/10000
hb_wave = 0.4861

def calzetti_law(wavelength_um):
    if wavelength_um >= 0.6300 and wavelength_um <= 2.2000:
        k_lambda = 2.659 * (-1.857 + 1.040 / wavelength_um) + 4.05
    if wavelength_um >= 0.1200  and wavelength_um < 0.6300:
        k_lambda = 2.659 * (-2.156 + 1.509 / wavelength_um - 0.198 / wavelength_um**2 + 0.011 / wavelength_um**3) + 4.05
    return k_lambda


def compute_ha_pab_av(pab_ha_ratio):
    """ PaB / Ha is the ratio you need, should be slightly greater than 1/20"""
    R_V_value = 4.05
    intrinsic_ratio = pab_factor / ha_factor
    k_factor = 2.5/(calzetti_law(ha_wave) - calzetti_law(pab_wave))
    A_V_value = R_V_value*k_factor*np.log10(pab_ha_ratio/intrinsic_ratio)
    return A_V_value

def compute_ratio_from_av(A_V_value):
    """ PaB / Ha is the ratio you need, should be slightly greater than 1/20"""
    R_V_value = 4.05
    intrinsic_ratio = pab_factor / ha_factor
    k_factor = 2.5/(calzetti_law(ha_wave) - calzetti_law(pab_wave))
    pab_ha_ratio = 10**(A_V_value / (R_V_value*k_factor)) * intrinsic_ratio
    return pab_ha_ratio

def compute_ha_pab_av_from_dustmap(dustmap_ratio):
    """ PaB / Ha is the ratio you need, should be slightly greater than 1/20. Already have pab ratio there, so don'tneed intrinsice"""
    R_V_value = 4.05
    k_factor = 2.5/(calzetti_law(ha_wave) - calzetti_law(pab_wave))
    intrinsic_ratio = pab_factor / ha_factor
    A_V_value = R_V_value*k_factor*np.log10(dustmap_ratio / intrinsic_ratio)
    return A_V_value

def compute_balmer_av(balmer_dec):
    R_V_value = 4.05
    intrinsic_ratio = ha_factor / hb_factor
    k_factor = 2.5/(calzetti_law(hb_wave) - calzetti_law(ha_wave))
    A_V_value = R_V_value*k_factor*np.log10(balmer_dec/intrinsic_ratio)
    return A_V_value

def read_catalog_av(id_msa, zqual_df):
    sps_df = read_SPS_cat()
    id_dr2 = zqual_df[zqual_df['id_msa']==id_msa]['id_DR2'].iloc[0]
    sps_row = sps_df[sps_df['id']==id_dr2]
    dust_16 = sps_row['dust2_16'].iloc[0]
    dust_50 = sps_row['dust2_50'].iloc[0]
    dust_84 = sps_row['dust2_84'].iloc[0]
    av_16 = dust2_to_AV(dust_16)
    av_50 = dust2_to_AV(dust_50)
    av_84 = dust2_to_AV(dust_84)
    print(f'A_V 50 for id_msa {id_msa}: {av_50}')
    return av_16, av_50, av_84


# print(compute_ha_pab_av(1/15))
