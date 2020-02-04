# Reads mesa's History file into a pandas table

import sys
import os
import string
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt


def read_history(file_loc):
    """Uses the translate file to produce a catalog with the appropriate filters

    Parameters:
    file_loc (string): location of the folder containing the history.data file


    Returns:
    df (pd.DataFrame): Pandas dataframe with columsn as the labels provided by mesa
    """
    df = ascii.read(file_loc+'history.data', header_start=4,
                    data_start=5).to_pandas()
    return df
