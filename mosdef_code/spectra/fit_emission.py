# Codes for simultaneously fitting the emission lines in a spectrum

import sys
import os
import string
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
from read_data import mosdef_df
from mosdef_obj_data_funcs import read_sed, read_mock_sed, get_mosdef_obj, read_composite_sed
from filter_response import lines, overview, get_index, get_filter_response
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.integrate as integrate
from query_funcs import get_zobjs
import initialize_mosdef_dirs as imd
import cluster_data_funcs as cdf
from spectra_funcs import read_composite_spectrum


def setup_emission():
	"""Sets the lines and ratios that we are trying to fit

    Parameters:

    Returns:
    """

    line_names = ['Halpha', 'Hbeta', 'N2_']

    # Figure out how we want to store the line info



def fit_emission(groupID):
    """Fits emission lines in a spectrum simultaneously

    Parameters:
    groupID (int): ID of the cluster to fit

    Returns:
    """

