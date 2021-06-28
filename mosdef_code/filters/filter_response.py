# get_filter_response - gets the response curve for a given filter

import sys
import os
import string
import numpy as np
import pandas as pd
import re
from astropy.io import ascii
import initialize_mosdef_dirs as imd


f = open(imd.mosdef_filter_translate)
lines = f.readlines()
f.close()

# Read the overview file as a dataframe
overview = pd.read_csv(imd.mosdef_filter_overview, header=None, sep='\n')


def get_index(filter_num):
    """Given a filter number in FILTER.RES.latest (overview), returns the index of the line containing this filter in the FILTER.RES.latest file

    Parameters:
    filter_num (int): number of the filter in overview


    Returns:
    idx (int): index of the line containing this
    """

    # Subtract one here since python dataframe is 0 indexed but document is 1
    # indexed!
    overview_filt = overview.iloc[filter_num - 1]
    # Name of the full line to search for in the overview textfile
    filt_fullname = overview_filt.iloc[0][7:]  # .split(' ', 1)[0]

    index = [idx for idx, line in enumerate(lines) if filt_fullname in line][0]
    return index


def get_filter_response(filter_num, lines=lines, overview=overview):
    """Given the filter number in FILTER.RES.latest (overview), returns the response curve
    Ran this in unix to create the overview file
    > more FILTER.RES.latest | grep lambda_c. > overview
    > more overview


    Parameters:
    filter_num (int): number of the filter in overview
    lines (list): the lines from the full FILTER.RES.latest, computed above
    overview (pd.DataFrame): overview file as a dataframe, computed above

    Returns:
    header_line (str): Line of the filter that it's pulling from
    response_df (pd.DataFrame): dataframe containing the response curve for the filter
    """

    # Get the index of the current filter (start), and of the next filter
    # (end), then create a dataframe of everything in between
    start_idx = get_index(filter_num)
    end_idx = get_index(filter_num + 1)

    # Line with the info about the filter
    header_line = lines[start_idx]

    # Get all the lines in between to make dataframe, ignoring the first line
    response_lines = lines[start_idx + 1:end_idx]

    # chop off the \n at the end and number and whitespace at beginning of
    # each line:
    response_list = [(float(line[7:18]), float(line[19:-1]))
                     for line in response_lines]

    response_df = pd.DataFrame(response_list, columns=[
                               'wavelength', 'transmission'])

    return header_line, response_df
