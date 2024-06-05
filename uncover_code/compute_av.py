import numpy as np
from fit_emission_uncover import line_list
from uncover_read_data import read_SPS_cat

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

def compute_ha_pab_av_from_dustmap(dustmap_ratio):
    """ PaB / Ha is the ratio you need, should be slightly greater than 1/20. Already have pab ratio there, so don'tneed intrinsice"""
    R_V_value = 4.05
    k_factor = 2.5/(calzetti_law(ha_wave) - calzetti_law(pab_wave))
    A_V_value = R_V_value*k_factor*np.log10(dustmap_ratio)
    return A_V_value

def compute_balmer_av(balmer_dec):
    R_V_value = 4.05
    intrinsic_ratio = ha_factor / hb_factor
    k_factor = 2.5/(calzetti_law(hb_wave) - calzetti_law(ha_wave))
    A_V_value = R_V_value*k_factor*np.log10(balmer_dec/intrinsic_ratio)
    return A_V_value

