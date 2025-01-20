from scipy import integrate
import numpy as np
from fit_emission_uncover_wave_divide import gaussian_func
from fit_emission_uncover_wave_divide import line_list, line_centers_rest
from astropy.io import ascii
from scipy.interpolate import interp1d


def integrate_filter(sedpy_filt):
    transmission = sedpy_filt.transmission
    wavelength_aa = sedpy_filt.wavelength
    scale_factor = integrate.trapz(transmission, wavelength_aa)
    scaled_transmission = transmission/scale_factor
    return scaled_transmission

def get_transmission_at_line(sedpy_filt, line_wave_aa, estimated_sigma=50, trasm_type='scaled'):
    ## Sigma in angstroms
    scaled_transmission = integrate_filter(sedpy_filt)

    if trasm_type == 'raw':
        scaled_transmission = sedpy_filt.transmission

    # Old method - only at central wavelength
    # wave_idx = np.argmin(np.abs(line_wave_aa-sedpy_filt.wavelength))  
    # line_transmission = scaled_transmission[wave_idx]

    # Now takes the median over the line width
    wave_idxs = np.logical_and((line_wave_aa-estimated_sigma) < sedpy_filt.wavelength, sedpy_filt.wavelength<(line_wave_aa+estimated_sigma))
    line_transmission = np.median(scaled_transmission[wave_idxs])

    return line_transmission

def get_line_coverage_old(sedpy_filt, line_wave_aa, line_width_aa):
    if line_width_aa < 5:
        line_width_aa = 25
    transmission = sedpy_filt.transmission
    wavelength_aa = sedpy_filt.wavelength
    transmission_max1 = transmission / np.max(transmission)
    line_start = line_wave_aa - line_width_aa/2
    line_end = line_wave_aa + line_width_aa/2
    line_waves = np.logical_and(wavelength_aa > line_start, wavelength_aa < line_end)  
    line_transmissions = transmission_max1[line_waves]
    line_avg_transmission = np.mean(line_transmissions)
    if np.isnan(line_avg_transmission) == True:
        line_avg_transmission = 0
    return line_avg_transmission

def get_line_coverage(id_msa, sedpy_filt, redshift, line_name):
    fit_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting/{id_msa}_emission_fits.csv').to_pandas()
    if line_name == 'ha':
        line_wave = fit_df['line_center_rest'].iloc[0] + fit_df['z_offset'].iloc[0]
        line_width = fit_df['sigma'].iloc[0]
        line_amp = fit_df['amplitude'].iloc[0]
    if line_name == 'pab':
        line_wave = fit_df['line_center_rest'].iloc[1] + fit_df['z_offset'].iloc[1]
        line_width = fit_df['sigma'].iloc[1]
        line_amp = fit_df['amplitude'].iloc[1]
    wave_start = line_wave - 3*line_width
    wave_end = line_wave + 3*line_width
    waves_rest = np.arange(wave_start, wave_end, 0.05)
    waves_obs = waves_rest * (1+redshift)
    gauss_ys_full = gaussian_func(waves_rest, line_wave, line_amp, line_width)
    gauss_ys_full_obs = gauss_ys_full / (1+redshift)
    int_flux = integrate.trapz(gauss_ys_full_obs, waves_obs)
    scaled_trasms = sedpy_filt.transmission / np.max(sedpy_filt.transmission)
    interp_trasm_func = interp1d(sedpy_filt.wavelength, scaled_trasms, kind='linear', fill_value=0, bounds_error=False)
    interp_trasms = interp_trasm_func(waves_obs)
    gauss_ys_trasm = gauss_ys_full_obs * interp_trasms
    int_flux_trasm_scaled = integrate.trapz(gauss_ys_trasm, waves_obs)
    line_coverage = int_flux_trasm_scaled / int_flux
    return line_coverage
    