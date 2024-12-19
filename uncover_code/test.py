from uncover_read_data import read_raw_spec, read_prism_lsf
from scipy.interpolate import interp1d
from astropy.io import ascii
from fit_emission_uncover_old import line_list, sig_to_velocity, velocity_to_sig
import numpy as np

def check_emission_lsf(id_msa):
    emission_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting/{id_msa}_emission_fits.csv').to_pandas()
    ha_sig = emission_df['sigma'].iloc[0]
    pab_sig = emission_df['sigma'].iloc[1]
    gaussian_vels = [sig_to_velocity(line_list[0][1], ha_sig), sig_to_velocity(line_list[1][1], pab_sig)]

    # Read in the lsf
    lsf = read_prism_lsf()
    # interpolate the lsf to match the wavelengths of the data
    lsf['wave_aa'] = lsf['WAVELENGTH'] * 10000
    interp_lsf = interp1d(lsf['wave_aa'], lsf['R'], kind='linear')
    lsf_FWHMs = [line_list[i][1] / interp_lsf(line_list[i][1]) for i in range(len(line_list))]
    # sigma = wavelength / (R * 2.355)
    lsf_sigs = [lsf_FWHMs[i] / 2.355 for i in range(len(line_list))]
    c = 299792 #km/s
    lsf_sigma_v_kms = [c/(interp_lsf(line_list[i][1])*2.355) for i in range(len(line_list))]
    true_vels = [np.sqrt(gaussian_vels[i]**2 - lsf_sigma_v_kms[i]**2)  for i in range(len(line_list))]
    breakpoint()

check_emission_lsf(47875)