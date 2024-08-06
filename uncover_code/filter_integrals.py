from scipy import integrate
import numpy as np

def integrate_filter(sedpy_filt):
    transmission = sedpy_filt.transmission
    wavelength_aa = sedpy_filt.wavelength
    scale_factor = integrate.trapz(transmission, wavelength_aa)
    scaled_transmission = transmission/scale_factor
    return scaled_transmission

def get_transmission_at_line(sedpy_filt, line_wave_aa):
    scaled_transmission = integrate_filter(sedpy_filt)
    wave_idx = np.argmin(np.abs(line_wave_aa-sedpy_filt.wavelength))  
    line_transmission = scaled_transmission[wave_idx]
    return line_transmission

def get_line_coverage(sedpy_filt, line_wave_aa, line_width_aa):
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