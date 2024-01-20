# Scales the spectrum to the sed
import numpy as np
from read_jwst_spectrum import flux_columns

def scale_spec(sed, spec):
    wavelength_range = (4000, 4500)
    sed_idxs = np.logical_and(sed['peak_wavelength'] > wavelength_range[0], sed['peak_wavelength'] < wavelength_range[1])
    spec_idxs = np.logical_and(spec['rest_wavelength'] > wavelength_range[0], spec['rest_wavelength'] < wavelength_range[1])
    sed_median = np.median(sed[sed_idxs]['f_lambda'])
    spec_median = np.median(spec[spec_idxs]['rest_flux_total'])
    scale_factor = sed_median / spec_median
    for name in flux_columns:
        spec[f'rest_{name}_scaled'] = spec[f'rest_{name}'] * scale_factor
    return scale_factor, spec