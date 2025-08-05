import numpy as np
from fit_emission_uncover_old import line_list
from uncover_read_data import read_SPS_cat, read_SPS_cat_all
from dust_equations_prospector import dust2_to_AV
from astropy.io import ascii
# from uncover_read_data import read_raw_spec, read_lineflux_cat, get_id_msa_list, read_fluxcal_spec
import sys

avneb_str = 'A$_{\\mathrm{V,neb}}$'

# theoretical scalings (to Hb, from naveen's paper)
ha_factor = 2.79
hb_factor = 1
hg_factor = 0.473
pab_factor = 0.155
paa_factor = 0.305

# nii_correction_ha_flux = 0.75
def get_nii_correction(id_msa, sps_df = []):
    if len(sps_df) == 0:
        sps_df = read_SPS_cat()
    sps_row = sps_df[sps_df['id_msa'] == id_msa]
    logmass = sps_row['mstar_50'].iloc[0]
    logsfr = np.log10(sps_row['sfr100_50'].iloc[0])
    predicted_metallicity = sanders_plane(logmass, logsfr)
    nii6585_ha_rat = sanders_nii_ratio(predicted_metallicity) # Adjusted relation
    # print(f'id: {id_msa}, niiha ratio: {nii6585_ha_rat}')
    niicombined_ha_rat = nii6585_ha_rat * (4/3) # From theory
    nii_correction_factor = 1 / (1+niicombined_ha_rat)
    return nii_correction_factor

def get_nii_correction_dr3(id_dr3, sps_df = []):
    if len(sps_df) == 0:
        sps_df = read_SPS_cat_all()
    sps_row = sps_df[sps_df['id'] == id_dr3]
    logmass = sps_row['mstar_50'].iloc[0]
    logsfr = np.log10(sps_row['sfr100_50'].iloc[0])
    predicted_metallicity = sanders_plane(logmass, logsfr)
    nii6585_ha_rat = sanders_nii_ratio(predicted_metallicity) # Adjusted relation
    # print(f'id: {id_msa}, niiha ratio: {nii6585_ha_rat}')
    niicombined_ha_rat = nii6585_ha_rat * (4/3) # From theory
    nii_correction_factor = 1 / (1+niicombined_ha_rat)
    return nii_correction_factor

def get_nii_correction_formmet(id_msa, sps_df = []):
    if len(sps_df) == 0:
        sps_df = read_SPS_cat()
    sps_row = sps_df[sps_df['id_msa'] == id_msa]
    metallicity = sps_row['met_50'].iloc[0]
    nii6585_ha_rat = sanders_nii_ratio(metallicity, linear_scale=0) # Adjusted relation
    niicombined_ha_rat = nii6585_ha_rat * (4/3) # From theoretical reatio
    nii_correction_factor = 1 / (1+niicombined_ha_rat)
    return nii_correction_factor

def get_fe_correction(id_msa, boot=False): # From fe_diagnostics.py
    fe_cor_df_indiv = ascii.read('/Users/brianlorenz/uncover/Data/generated_tables/fe_cor_df_indiv.csv').to_pandas()
    fe_cor_df_row = fe_cor_df_indiv[fe_cor_df_indiv['id_msa'] == id_msa]
    if len(fe_cor_df_row) == 0:
        predicted_fe_pab_ratio = fe_cor_df_indiv['median_fe_pab_ratios'].iloc[0]
    else:
        predicted_fe_pab_ratio = fe_cor_df_row['fe_pab_ratio'].iloc[0]
    if boot == True:
        fe_scatter = np.std(fe_cor_df_indiv['fe_pab_ratio'])
        predicted_fe_pab_ratio = predicted_fe_pab_ratio+np.random.normal(loc=0, scale=fe_scatter)
    
    # if id_msa in [14573]:
    #     predicted_fe_pab_ratio = 0
    pab_correction_factor = 1 / (1+predicted_fe_pab_ratio)
    return pab_correction_factor

def get_fe_correction_bymass(id_msa, sps_df = []): # From fe_diagnostics.py
    if len(sps_df) == 0:
        sps_df = read_SPS_cat()
    sps_row = sps_df[sps_df['id_msa'] == id_msa]
    log_mass = sps_row['mstar_50'].iloc[0]
    fe_cor_df = ascii.read('/Users/brianlorenz/uncover/Data/generated_tables/fe_cor_df.csv').to_pandas()
    predicted_fe_pab_ratio = fe_cor_df['y_int'].iloc[0]+fe_cor_df['slope'].iloc[0]*log_mass
    pab_correction_factor = 1 / (1+predicted_fe_pab_ratio)
    return pab_correction_factor

ha_wave = line_list[0][1]/10000
pab_wave = line_list[1][1]/10000
hg_wave = 0.4341
paa_wave = 18750/10000
hb_wave = 0.4861

def calzetti_law(wavelength_um):
    if wavelength_um >= 0.6300 and wavelength_um <= 2.2000:
        k_lambda = 2.659 * (-1.857 + 1.040 / wavelength_um) + 4.05
    if wavelength_um >= 0.1200  and wavelength_um < 0.6300:
        k_lambda = 2.659 * (-2.156 + 1.509 / wavelength_um - 0.198 / wavelength_um**2 + 0.011 / wavelength_um**3) + 4.05
    return k_lambda

def reddy_aurora_law(wavelength_um): # https://arxiv.org/pdf/2506.17396, eq 11
    # RV ranges from 5.5 to 6.8
    # Rest wave
    if wavelength_um < 0.35:
        print('Only valid for 0.35<wave<1.28')
        sys.exit()
    if wavelength_um > 1.283:
        print('Only valid for 0.35<wave<1.28')
        sys.exit()
    a0 = -14.198
    a1 = 17.002
    a2 = -8.086
    a3 = 2.177
    a4 = -0.319
    a5 = 0.021
    k_lambda = -a0 + a1 * (1/wavelength_um) + a2 * (1/wavelength_um**2) + a3 * (1/wavelength_um**3) + a4 * (1/wavelength_um**4) + a5 * (1/wavelength_um**5)
    return k_lambda

def compute_ha_paalpha_av(paalpha_ha_ratio):
    """ PaB / Ha is the ratio you need, should be slightly greater than 1/20"""
    intrinsic_ratio = paa_factor / ha_factor
    R_V_value = 4.05
    k_factor = 2.5/(calzetti_law(ha_wave) - calzetti_law(paa_wave))
    A_V_value = R_V_value*k_factor*np.log10(paalpha_ha_ratio/intrinsic_ratio)
    return A_V_value

def compute_paalpha_pabeta_av(paalpha_pabeta_ratio):
    """ PaB / Ha is the ratio you need, should be slightly greater than 1/20"""
    R_V_value = 4.05
    intrinsic_ratio = paa_factor / pab_factor
    k_factor = 2.5/(calzetti_law(pab_wave) - calzetti_law(paa_wave))
    A_V_value = R_V_value*k_factor*np.log10(paalpha_pabeta_ratio/intrinsic_ratio)
    return A_V_value


def compute_pab_paa_av(paa_pab_ratio):
    """ PaA / PaB is the ratio you need"""
    R_V_value = 4.05
    intrinsic_ratio = paa_factor / pab_factor
    k_factor = 2.5/(calzetti_law(pab_wave) - calzetti_law(paa_wave))
    A_V_value = R_V_value*k_factor*np.log10(paa_pab_ratio/intrinsic_ratio)
    return A_V_value


def compute_ha_pab_av(pab_ha_ratio, law='calzetti'):
    """ PaB / Ha is the ratio you need, should be slightly greater than 1/20"""
    intrinsic_ratio = pab_factor / ha_factor
    if law == 'calzetti':
        R_V_value = 4.05
        k_factor = 2.5/(calzetti_law(ha_wave) - calzetti_law(pab_wave))
    if law == 'reddy':
        R_V_value = 5.5 # or 6.8
        k_factor = 2.5/(reddy_aurora_law(ha_wave) - reddy_aurora_law(pab_wave))
    if law == 'cardelli':
        R_V_value = 3.1
        k_factor = 2.5/(cardelli_k(ha_wave) - cardelli_k(pab_wave))
    A_V_value = R_V_value*k_factor*np.log10(pab_ha_ratio/intrinsic_ratio)
    return A_V_value
def compute_ha_pab_av2(ha_pab_ratio): # Does the same thin but it's more intuitive this way
    """ PaB / Ha is the ratio you need, should be slightly greater than 1/20"""
    R_V_value = 4.05
    intrinsic_ratio = ha_factor / pab_factor
    k_factor = 2.5/(calzetti_law(pab_wave) - calzetti_law(ha_wave))
    A_V_value = R_V_value*k_factor*np.log10(ha_pab_ratio/intrinsic_ratio)
    return A_V_value

def compute_ratio_from_av(A_V_value, law='calzetti'):
    """ PaB / Ha is the ratio you need, should be slightly greater than 1/20"""
    intrinsic_ratio = pab_factor / ha_factor
    if law == 'calzetti':
        R_V_value = 4.05
        k_factor = 2.5/(calzetti_law(ha_wave) - calzetti_law(pab_wave))
    if law == 'reddy':
        R_V_value = 5.5 # or 6.8
        k_factor = 2.5/(reddy_aurora_law(ha_wave) - reddy_aurora_law(pab_wave))
    if law == 'cardelli':
        R_V_value = 3.1
        k_factor = 2.5/(cardelli_k(ha_wave) - cardelli_k(pab_wave))
    pab_ha_ratio = 10**(A_V_value / (R_V_value*k_factor)) * intrinsic_ratio
    return pab_ha_ratio

def compute_ha_pab_av_from_dustmap(dustmap_ratio):
    """ PaB / Ha is the ratio you need, should be slightly greater than 1/20. Already have pab ratio there, so don'tneed intrinsice"""
    R_V_value = 4.05
    k_factor = 2.5/(calzetti_law(ha_wave) - calzetti_law(pab_wave))
    intrinsic_ratio = pab_factor / ha_factor
    A_V_value = R_V_value*k_factor*np.log10(dustmap_ratio / intrinsic_ratio)
    return A_V_value

def compute_balmer_av(balmer_dec, law='calzetti'):
    intrinsic_ratio = ha_factor / hb_factor
    if law == 'calzetti':
        R_V_value = 4.05
        k_factor = 2.5/(calzetti_law(hb_wave) - calzetti_law(ha_wave))
    if law == 'reddy':
        R_V_value = 5.5 # or 6.8
        k_factor = 2.5/(reddy_aurora_law(hb_wave) - reddy_aurora_law(ha_wave))
    if law == 'cardelli':
        R_V_value = 3.1
        k_factor = 2.5/(cardelli_k(hb_wave) - cardelli_k(ha_wave))
    A_V_value = R_V_value*k_factor*np.log10(balmer_dec/intrinsic_ratio)
    return A_V_value

def compute_balmer_ratio_from_av(av_value, law='calzetti'):
    intrinsic_ratio = ha_factor / hb_factor
    if law == 'calzetti':
        R_V_value = 4.05
        k_factor = 2.5/(calzetti_law(hb_wave) - calzetti_law(ha_wave))
    if law == 'reddy':
        R_V_value = 5.5 # or 6.8
        k_factor = 2.5/(reddy_aurora_law(hb_wave) - reddy_aurora_law(ha_wave))
    if law == 'cardelli':
        R_V_value = 3.1
        k_factor = 2.5/(cardelli_k(hb_wave) - cardelli_k(ha_wave))
    balmer_dec = intrinsic_ratio * 10**(av_value / (R_V_value*k_factor)) 
    return balmer_dec

def read_catalog_av(id_msa, zqual_df):
    sps_df = read_SPS_cat()
    # id_dr2 = zqual_df[zqual_df['id_msa']==id_msa]['id_DR2'].iloc[0]
    sps_row = sps_df[sps_df['id_msa']==id_msa]
    dust_16 = sps_row['dust2_16'].iloc[0]
    dust_50 = sps_row['dust2_50'].iloc[0]
    dust_84 = sps_row['dust2_84'].iloc[0]
    av_16 = dust2_to_AV(dust_16)
    av_50 = dust2_to_AV(dust_50)
    av_84 = dust2_to_AV(dust_84)
    print(f'A_V 50 for id_msa {id_msa}: {av_50}')
    return av_16, av_50, av_84


def sanders_plane(log_mass, log_sfr):
    u60 = log_mass - 0.6*log_sfr
    y = u60 - 10
    metallicity = 8.8 + (0.188*y) + (-0.22 * y**2) + (-0.0531 * y**3)
    return metallicity

def sanders_nii_ratio(met_12_log_OH, linear_scale = 8.69):
    c0 = -0.606
    c1 = 1.28
    c2 = -0.435
    c3 = -0.485
    x = met_12_log_OH - linear_scale
    log_nii_ratio = c0 + (c1*x) + (c2 * x**2) + (c3 * x**3)
    nii_ratio = 10**log_nii_ratio
    return nii_ratio

# def cardelli_law(wavelength_um):
#     #valid for 0.3um < wave < 1.1um
#     wavelength_inv_um = 1/wavelength_um
#     if wavelength_inv_um > 0.3 and wavelength_inv_um < 1.1:
#         a_x = 0.574*wavelength_inv_um**(1.61)
#         b_x = -0.527*wavelength_inv_um**(1.61)
#     elif wavelength_inv_um >= 1.1 and wavelength_inv_um < 3.3:
#         y = wavelength_inv_um - 1.82
#         a_x = 1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
#         b_x = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7
#     A_lambda_divide_AV = (a_x+b_x/3.1)
#     return A_lambda_divide_AV

def cardelli_k(wavelength_um):
    #valid for 0.3um < wave < 1.1um
    wavelength_inv_um = 1/wavelength_um
    if wavelength_inv_um > 0.3 and wavelength_inv_um < 1.1:
        a_x = 0.574*wavelength_inv_um**(1.61)
        b_x = -0.527*wavelength_inv_um**(1.61)
    elif wavelength_inv_um >= 1.1 and wavelength_inv_um < 3.3:
        y = wavelength_inv_um - 1.82
        a_x = 1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
        b_x = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7
    k_cardelli = 3.1*a_x+b_x
    return k_cardelli
# id_msa_list = get_id_msa_list(full_sample=False)
# rats = []
# for id_msa in id_msa_list:
#     rat = get_nii_correction(id_msa, sps_df = [])
#     rats.append(rat)
# breakpoint()
# print(compute_ha_pab_av(1/16))
# print(get_fe_correction(18471))

# import matplotlib.pyplot as plt
# av_values = np.arange(0, 2, 0.1)
# for law in ['calzetti', 'cardelli', 'reddy']:
#     balmer_decs = compute_balmer_ratio_from_av(av_values, law=law)
#     pabeta_decs = compute_ratio_from_av(av_values, law=law)
#     plt.plot(av_values, pabeta_decs)
# plt.show()
