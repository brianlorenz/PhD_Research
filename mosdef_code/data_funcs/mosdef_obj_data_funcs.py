# Functions that deal with the mosdef_df and data reading

import sys
import os
import string
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
from tabulate import tabulate
from astropy.table import Table
from read_data import mosdef_df
import initialize_mosdef_dirs as imd


def get_mosdef_obj(field, v4id):
    """Given a field and id, find the object in the mosdef_df dataframe

    Parameters:
    field (string): name of the field of the object
    id (int): HST V4.1 id of the object

    Returns:
    mosdef_obj (pd.DataFrame): Datatframe with one entry corresponding to the current object
    """
    mosdef_obj = mosdef_df[np.logical_and(
        mosdef_df['FIELD_STR'] == field, mosdef_df['V4ID'] == v4id)]
    # There should be a unique match - exit with an error if not
    if len(mosdef_obj) < 1:
        sys.exit('No match found on FIELD_STR and V4ID')
    # If there's a duplicate, take the first one
    if len(mosdef_obj) > 1:
        mosdef_obj = mosdef_obj.iloc[0]
        # WHAT TO DO HERE WHEN YOU FIND A REPEAT? NEED TO STORE AND MOVE ONE
        print('Duplicate obj, taking the first instance')
        return mosdef_obj
    return mosdef_obj.iloc[0]


def read_sed(field, v4id):
    """Given a field and id, read in the sed

    Parameters:
    field (string): name of the field of the object
    v4id (int): HST V4.1 id of the object


    Returns:
    """
    sed_location = imd.home_dir + f'/mosdef/sed_csvs/{field}_{v4id}_sed.csv'
    if not os.path.exists(sed_location):
        sed_location = imd.home_dir + f'/mosdef/sed_csvs/{field}_{v4id}_3DHST_sed.csv'
    sed = ascii.read(sed_location).to_pandas()
    return sed


def read_mock_sed(field, v4id):
    """Given a field and id, read in the sed

    Parameters:
    field (string): name of the field of the object
    v4id (int): HST V4.1 id of the object


    Returns:
    """
    sed_location = imd.home_dir + f'/mosdef/mock_sed_csvs/{field}_{v4id}_sed.csv'
    if not os.path.exists(sed_location):
        sed_location = imd.home_dir + f'/mosdef/mock_sed_csvs/{field}_{v4id}_3DHST_sed.csv'
    sed = ascii.read(sed_location).to_pandas()
    return sed


def read_composite_sed(groupID):
    """Given a groupID, read in the composite sed

    Parameters:
    groupID (int): id of the cluster to read


    Returns:
    """
    sed_location = imd.home_dir + f'/mosdef/composite_sed_csvs/{groupID}_sed.csv'
    sed = ascii.read(sed_location).to_pandas()
    return sed
