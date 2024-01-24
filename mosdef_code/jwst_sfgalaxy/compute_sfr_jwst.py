from fit_emission_jwst import emission_fit_dir, lines_dict
from read_jwst_spectrum import z_sfgalaxy
from astropy.io import ascii
import numpy as np
from compute_cluster_sfrs import flux_to_luminosity
from compute_new_sfrs import correct_lum_for_dust, ha_lum_to_sfr

save_name = 'combined'

intrinsic_ratios = {
    'halpha': 2.860,
    'hbeta': 1,
    'hgamma': 0.468,
    'hdelta': 0.259,
    'hepsilon': 0.159
}

def compute_sfr():
    # Read in measured fluxes and ratios
    fit_df = ascii.read(emission_fit_dir + f'/{save_name}_emission_fits.csv').to_pandas()
    hgamma_flux = fit_df[fit_df['line_name']=='Hgamma']['flux'].iloc[0]
    err_hgamma_flux = fit_df[fit_df['line_name']=='Hgamma']['err_flux'].iloc[0]
    hg_hd_ratio = fit_df.iloc[0]['hg_hd_ratio']
    intrinsic_hg_hd_ratio = intrinsic_ratios['hgamma']/intrinsic_ratios['hdelta']
    
    # Compute dust attenuation from line ratio
    attenuation_hgamma = compute_attenuation(hg_hd_ratio, intrinsic_hg_hd_ratio) #Attenuation really high??
    # Convert flux to luminosity
    hgamma_lum, err_hgamma_lum = flux_to_luminosity(hgamma_flux, z_sfgalaxy, flux_errs = np.array([err_hgamma_flux]))
    err_hgamma_lum = err_hgamma_lum[0]
    # Aplpy dust correction with computed attenuation
    hgamma_dust_cor = correct_lum_for_dust(hgamma_lum, attenuation_hgamma)
    err_hgamma_dust_cor = correct_lum_for_dust(err_hgamma_lum, attenuation_hgamma)
    # Convert hgamma lum to halpha lum using ratios
    ha_lum_dust_cor = hgamma_dust_cor*(intrinsic_ratios['halpha']/intrinsic_ratios['hgamma'])
    err_ha_lum_dust_cor = err_hgamma_dust_cor*(intrinsic_ratios['halpha']/intrinsic_ratios['hgamma'])
    halpha_sfr = ha_lum_to_sfr(ha_lum_dust_cor, imf='subsolar')
    err_halpha_sfr = ha_lum_to_sfr(err_ha_lum_dust_cor, imf='subsolar')
    log_halpha_sfr = np.log10(halpha_sfr)
    breakpoint()
    
    # Then to SFR

def calzetti_law(target_wave_angstrom, R_V=4.05):
    """Apply a Calzetti dust law

    Parameters:
    target_wave (float) = wavelenght to correct the extinction to (angstrom)
    R_V (float): R_V value fromt he Calzetti law

    Returns:
    avs_balmer_target (pd.DataFrame): Extinction now shifted to target wavelength
    """
    target_wave_um = target_wave_angstrom / 10**4
    # Compute k(lambda) using Calzetti
    if target_wave_angstrom < 6360:
        k_lambda = 2.659*(-2.156 + 1.509/target_wave_um - 0.198/(target_wave_um**2) + 0.011/(target_wave_um**3))+R_V
    elif target_wave_angstrom >= 6360:
        k_lambda = 2.659*(-1.857 + 1.040/(target_wave_um)) + R_V

    return k_lambda



# See appendix in Momcheva 2012 for details
# https://www.arxiv.org/pdf/1207.5479.pdf
# also   https://iopscience.iop.org/article/10.1088/0004-637X/763/2/145/pdf
def compute_attenuation(hg_hd_ratio, intrinsic_hg_hd_ratio):
    color_excess = -2.5/(calzetti_law(lines_dict['Hdelta'])-calzetti_law(lines_dict['Hgamma'])) * np.log10(intrinsic_hg_hd_ratio/hg_hd_ratio)
    attenuation_V = calzetti_law(5500) * color_excess
    attenuation_hgamma = calzetti_law(lines_dict['Hgamma']) * color_excess
    print(f'AV = {attenuation_V}')
    return attenuation_hgamma
compute_sfr()