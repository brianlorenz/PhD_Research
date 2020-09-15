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
        mosdef_df['Z_MOSFIRE'] > 0, np.logical_and(
            mosdef_df['FIELD_STR'] == field, mosdef_df['V4ID'] == v4id))]
    # There should be a unique match - exit with an error if not
    if len(mosdef_obj) < 1:
        sys.exit(f'No match found on FIELD_STR {field} and V4ID {v4id}')
    # If there's a duplicate, take the first one
    if len(mosdef_obj) > 1:
        mosdef_obj = mosdef_obj.iloc[0]
        print('Duplicate obj, taking the first instance with redshift')
        return mosdef_obj
    return mosdef_obj.iloc[0]


def read_sed(field, v4id, norm=False):
    """Given a field and id, read in the sed

    Parameters:
    field (string): name of the field of the object
    v4id (int): HST V4.1 id of the object
    norm (boolean): set to True to read the normalized SEDs


    Returns:
    """
    sed_location = imd.home_dir + f'/mosdef/sed_csvs/{field}_{v4id}_sed.csv'
    if norm:
        sed_location = imd.home_dir + f'/mosdef/sed_csvs/norm_sed_csvs/{field}_{v4id}_norm.csv'
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


def read_mock_composite_sed(groupID):
    """Given a groupID, read in the mock composite sed

    Parameters:
    groupID (int): id of the cluster to read


    Returns:
    """
    sed_location = imd.home_dir + f'/mosdef/mock_sed_csvs/mock_composite_sed_csvs/{groupID}_mock_sed.csv'
    sed = ascii.read(sed_location).to_pandas()
    return sed


def read_fast_continuum(mosdef_obj):
    """Given a field and id, read in the fast fit continuum for that SED

    Parameters:
    mosdef_obj (pd.DataFrame): From the get_mosdef_obj function

    Returns:
    """
    field = mosdef_obj['FIELD_STR']
    mosdef_id = mosdef_obj['V4ID']
    cont_location = imd.home_dir + f'/mosdef/FAST/{field}_BEST_FITS/{field}_v4.1_zall.fast_{mosdef_id}.fit'
    cont = ascii.read(cont_location).to_pandas()
    cont.columns = ['observed_wavelength', 'f_lambda']
    cont['rest_wavelength'] = cont['observed_wavelength'] / \
        (1 + mosdef_obj['Z_MOSFIRE'])
    cont['f_lambda'] = 10**(-19) * cont['f_lambda']
    return cont


def setup_get_AV():
    """Run this before running get_AV, asince what this returns needs to be passed to get_Av

    Parameters:

    Returns:
    fields (list): Strings of the fields in mosdef
    av_dfs (list of pd.DataFrames): Conatins dataframes in the order of fields with FAST fit info
    """
    fields = ['AEGIS', 'COSMOS', 'GOODS-N', 'GOODS-S', 'UDS']
    av_dfs = [ascii.read(imd.mosdef_dir + f'/Fast/{field}_v4.1_zall.fast.fout', header_start=17).to_pandas() for field in fields]
    return fields, av_dfs


def get_AV(fields, av_dfs, mosdef_obj):
    """Pass in the outputs of setup_get_AV, then a mosdef_obj to return the Av

    Parameters:
    fields (list): Strings of the fields in mosdef
    av_dfs (list of pd.DataFrames): Conatins dataframes in the order of fields with FAST fit info

    Returns:
    Av (float): AV  of the object
    """
    field_index = [i for i, field in enumerate(
        fields) if field == mosdef_obj['FIELD_STR']][0]
    av_df = av_dfs[field_index]
    Av = av_df[av_df['id'] == mosdef_obj['V4ID']]['Av'].iloc[0]
    return Av
