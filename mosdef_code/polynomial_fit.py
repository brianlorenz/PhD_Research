# Fits a polynomial to one of the mosdef SEDs

import sys
import os
import string
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
from read_data import mosdef_df
from mosdef_obj_data_funcs import get_mosdef_obj, read_sed


def poly_fit(field, v4id):
    sed = read_sed(field, v4id)
    sed['rest_wavelength'] = sed['peak_wavelength'] / (1 + sed['Z_MOSFIRE'])
    good_idxs = sed['f_lambda'] > -98
    # Poly_fit says to use 1/sigma weights for gaussian uncertainties
    # What degree to use for fitting?
    coeff = np.polyfit(np.log10(sed[good_idxs]['rest_wavelength']),
                       sed[good_idxs]['f_lambda'], deg=8, w=(1 / sed[good_idxs]['err_f_lambda']))
    fit_func = np.poly1d(coeff)
    return fit_func
