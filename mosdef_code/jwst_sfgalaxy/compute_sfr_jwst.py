from fit_emission_jwst import emission_fit_dir, lines_dict
from astropy.io import ascii
import numpy as np

save_name = 'combined'

intrinsic_ratios = {
    'halpha': 2.860,
    'hbeta': 1,
    'hgamma': 0.468,
    'hdelta': 0.259,
    'hepsilon': 0.159
}

def compute_sfr():
    fit_df = ascii.read(emission_fit_dir + f'/{save_name}_emission_fits.csv').to_pandas()
    hg_hd_ratio = fit_df.iloc[0]['hg_hd_ratio']
    intrinsic_hg_hd_ratio = intrinsic_ratios['hgamma']/intrinsic_ratios['hdelta']
    compute_attenuation(hg_hd_ratio, intrinsic_hg_hd_ratio)
    # Need to evaluate k(lambda) at hg and hd wavelengths to compute the A_V of the dust between them. See formula
    # https://iopscience.iop.org/article/10.1088/0004-637X/763/2/145/pdf
    breakpoint()
    # Then dust correct to get full Hg flux
    # Then convert Hg to Ha flux using the ratios
    # Then convert to luminosity was redshift
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
def compute_attenuation(hg_hd_ratio, intrinsic_hg_hd_ratio):
    color_excess = -2.5/(calzetti_law(lines_dict[4])-calzetti_law(lines_dict['Hgamma'])) * np.log10(hg_hd_ratio/intrinsic_hg_hd_ratio)
    attenuation_hgamma = calzetti_law(lines_dict['Hgamma']) * color_excess
    print(attenuation_hgamma)
# print(calzetti_law(lines_dict['Hdelta']))
# print(calzetti_law(lines_dict['Hgamma']))
compute_sfr()