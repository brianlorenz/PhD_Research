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
    sed_location = imd.sed_csvs_dir + f'/{field}_{v4id}_sed.csv'
    if not os.path.exists(sed_location):
        sed_location = imd.sed_csvs_dir + f'/{field}_{v4id}_3DHST_sed.csv'
    if norm:
        sed_location = imd.norm_sed_csvs_dir + f'/{field}_{v4id}_norm.csv'
        if not os.path.exists(sed_location):
            sed_location = imd.norm_sed_csvs_dir + f'/{field}_{v4id}_3DHST_sed.csv'
    
    sed = ascii.read(sed_location).to_pandas()
    return sed


def read_mock_sed(field, v4id):
    """Given a field and id, read in the sed

    Parameters:
    field (string): name of the field of the object
    v4id (int): HST V4.1 id of the object


    Returns:
    """
    sed_location = imd.mock_sed_csvs_dir + f'/{field}_{v4id}_sed.csv'
    if not os.path.exists(sed_location):
        sed_location = imd.mock_sed_csvs_dir + f'/{field}_{v4id}_3DHST_sed.csv'
    sed = ascii.read(sed_location).to_pandas()
    return sed


def read_composite_sed(groupID):
    """Given a groupID, read in the composite sed

    Parameters:
    groupID (int): id of the cluster to read


    Returns:
    """
    sed_location = imd.composite_sed_csvs_dir + f'/{groupID}_sed.csv'
    sed = ascii.read(sed_location).to_pandas()
    return sed


def read_mock_composite_sed(groupID):
    """Given a groupID, read in the mock composite sed

    Parameters:
    groupID (int): id of the cluster to read


    Returns:
    """
    sed_location = imd.mock_composite_sed_csvs_dir + f'/{groupID}_mock_sed.csv'
    # sed_location = imd.home_dir + \
    #     f'/mosdef/mock_sed_csvs/mock_composite_sed_csvs/{groupID}_mock_sed.csv'
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
    cont_location = imd.home_dir + \
        f'/mosdef/FAST/{field}_BEST_FITS/{field}_v4.1_zall.fast_{mosdef_id}.fit'
    cont = ascii.read(cont_location).to_pandas()
    cont.columns = ['observed_wavelength', 'f_lambda']
    cont['rest_wavelength'] = cont['observed_wavelength'] / \
        (1 + mosdef_obj['Z_MOSFIRE'])
    cont['f_lambda'] = 10**(-19) * cont['f_lambda']
    cont['f_lambda_rest'] = cont['f_lambda'] * (1 + mosdef_obj['Z_MOSFIRE'])
    return cont


def setup_get_AV():
    """Run this before running get_AV, asince what this returns needs to be passed to get_Av

    Parameters:

    Returns:
    fields (list): Strings of the fields in mosdef
    av_dfs (list of pd.DataFrames): Conatins dataframes in the order of fields with FAST fit info
    """
    fields = ['AEGIS', 'COSMOS', 'GOODS-N', 'GOODS-S', 'UDS']
    av_dfs = [ascii.read(
        imd.mosdef_dir + f'/Fast/{field}_v4.1_zall.fast.fout', header_start=17).to_pandas() for field in fields]
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


def setup_get_ssfr():
    """Run this before running get_ssfr, asince what this returns needs to be passed to get_Av

    Parameters:

    Returns:
    ssfr_mosdef_merge_no_dups (pd.DataFrame): Pandas dataframe of the ssfr info, mosdef_df info, and duplicates removed
    """
    ssfr_dat = Table.read(
        imd.mosdef_dir + '/mosdef_sfrs_latest.fits', format='fits')
    ssfr_df = ssfr_dat.to_pandas()
    ssfr_df['FIELD_STR'] = [ssfr_df.iloc[i]['FIELD'].decode(
        "utf-8").rstrip() for i in range(len(ssfr_df))]
    # Merge with mosdef_df so that we are matching v4ids
    ssfr_mosdef_merge = mosdef_df.merge(ssfr_df, how='inner', left_on=[
                                        'FIELD_STR', 'ID', 'MASKNAME'], right_on=['FIELD_STR', 'ID', 'MASKNAME'])
    # Drop duplicates in favor of those that have measured ssfrs
    # Finds all the duplicates
    dupes = ssfr_mosdef_merge[ssfr_mosdef_merge.duplicated(
        ['FIELD_STR', 'V4ID'], keep=False)]
    # Find indicies of all with non-measurement for SSFR
    dupes[dupes['SFR_CORR'] < -1].index
    drop_idx_non_detect = dupes[dupes['SFR_CORR'] < -1].index
    dupes = dupes.drop(drop_idx_non_detect)
    drop_idx_still_dup = dupes[dupes.duplicated(['FIELD_STR', 'V4ID'])].index
    ssfr_mosdef_merge = ssfr_mosdef_merge.drop(drop_idx_non_detect)
    ssfr_mosdef_merge_no_dups = ssfr_mosdef_merge.drop(drop_idx_still_dup)
    ssfrs = ssfr_mosdef_merge_no_dups[
        'SFR_CORR'] / 10**ssfr_mosdef_merge_no_dups['LMASS']
    ssfrs[ssfrs < 0] = -999
    ssfr_mosdef_merge_no_dups['SSFR'] = ssfrs
    return ssfr_mosdef_merge_no_dups


def merge_ar_ssfr(ar_df, ssfr_mosdef_merge_no_dups):
    """Merges the ar_df with the ssfr_mosdef_merge_no_dups dataframe

    Parameters:

    Returns:
    ar_ssfr_merge (pd.DataFrame): Pandas dataframe of the ssfr info, mosdef_df info, and duplicates removed
    """
    ar_ssfr_merge = ar_df.merge(ssfr_mosdef_merge_no_dups, how='left', left_on=[
                                'field', 'v4id'], right_on=['FIELD_STR', 'V4ID'])
    return ar_ssfr_merge


def merge_emission(ar_df):
    """Run to merge ar_df with the line emission catalog

    Parameters:

    Returns:
    ssfr_mosdef_merge_no_dups (pd.DataFrame): Pandas dataframe of the ssfr info, mosdef_df info, and duplicates removed
    """
    lines_dat = Table.read(
        imd.mosdef_dir + '/linemeas_latest.fits', format='fits')
    line_df = lines_dat.to_pandas()
    line_df['FIELD_STR'] = [line_df.iloc[i]['FIELD'].decode(
        "utf-8").rstrip() for i in range(len(line_df))]
    # Merge with mosdef_df so that we are matching v4ids
    line_mosdef_merge = mosdef_df.merge(line_df, how='inner', left_on=['FIELD', 'ID', 'MASKNAME', 'APERTURE_NO', 'SLITOBJNAME', 'FIELD_STR'], right_on=[
                                        'FIELD', 'ID', 'MASKNAME', 'APERTURE_NO', 'SLITOBJNAME', 'FIELD_STR'])
    drop_idx_dup = line_mosdef_merge[
        line_mosdef_merge.duplicated(['FIELD_STR', 'V4ID'])].index
    line_mosdef_merge = line_mosdef_merge.drop(drop_idx_dup)
    ar_line_merge = ar_df.merge(line_mosdef_merge, how='inner', left_on=[
                                'field', 'v4id'], right_on=['FIELD_STR', 'V4ID'])
    return ar_line_merge
