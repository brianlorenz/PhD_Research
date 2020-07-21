# Deals with thee linemeas_latest file, reading emission lines

import sys
import os
import string
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
from astropy.table import Table
from read_data import mosdef_df
from mosdef_obj_data_funcs import read_sed, read_mock_sed, get_mosdef_obj, read_composite_sed
from filter_response import lines, overview, get_index, get_filter_response
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.integrate as integrate
from query_funcs import get_zobjs
import initialize_mosdef_dirs as imd
import cluster_data_funcs as cdf
from axis_ratio_funcs import read_axis_ratio


def setup_emission_df():
    """Converts the linemeas_latest.fits file into a pandas dataframe as a csvf for ease of use

    Parameters:

    Returns:
    """
    breakpoint()
    file = imd.mosdef_dir + '/linemeas_latest.fits'
    lines_df = Table.read(file, format='fits').to_pandas()
    lines_df['FIELD_STR'] = [lines_df.iloc[i]['FIELD'].decode(
        "utf-8").rstrip() for i in range(len(lines_df))]
    lines_df['v4id'] = [lines_df.iloc[i]['ID'] for i in range(len(lines_df))]
    lines_df.to_csv(file.replace('.fits', '.csv'), index=False)
    return

    # PROBLEM WITH READING IN THE DATA DOESNT SAVE PROPERLY
    # CHECK IF WE SHOULD CALL IT V$iD or maybe need to match with mosdef obj
    # to get proer id. probably safer


def read_emission_df():
    """Reads-in the emission line data from linemeas

    Parameters:

    Returns:
    emission_df (pd.DataFrame): Dataframe containing emission line measurements and info
    """
    file = imd.mosdef_dir + '/linemeas_latest.csv'
    emission_df = ascii.read(file).to_pandas
    return emission_df
