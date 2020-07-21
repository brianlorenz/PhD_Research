# Functions that deal with reading and manipulating the axis ratio data

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


def make_axis_ratio_catalogs():
    '''Makes the galfit_data.csv file that is used for all of our objects

    Parameters:

    Returns:
    '''

    field_names = ['AEGIS', 'COSMOS', 'GOODS-N', 'GOODS-S', 'UDS']
    cat_names = os.listdir(imd.mosdef_dir + '/axis_ratio_data/AEGIS/')

    zobjs = get_zobjs()
    # Drop duplicates
    zobjs = list(dict.fromkeys(zobjs))
    # Remove objects with ID less than zero
    zobjs = [obj for obj in zobjs if obj[1] > 0]
    # Sort
    zobjs.sort()

    for cat_name in cat_names:
        cat_dfs = [ascii.read(imd.mosdef_dir + f'/axis_ratio_data/{field}/' +
                              cat_name).to_pandas() for field in field_names]
        for i in range(len(cat_dfs)):
            cat_dfs[i]['FIELD'] = field_names[i]
        rows = []
        for obj in zobjs:
            field = obj[0]
            v4id = obj[1]
            print(f'Finding Match for {field}, {v4id}')
            mosdef_obj = get_mosdef_obj(field, v4id)
            cat_idx = field_names.index(field)
            cat_df = cat_dfs[cat_idx]
            obj_row = cat_df[cat_df['NUMBER'] == v4id]
            ra_diff = obj_row['RA'] - mosdef_obj['RA']
            dec_diff = obj_row['DEC'] - mosdef_obj['DEC']
            if (ra_diff + dec_diff).iloc[0] > 0.01:
                sys.exit(f'ERROR! WRONG MATCH ON OBJECT FOR {field}, {v4id}')
            rows.append(obj_row)
        final_df = pd.concat(rows)
        final_df.to_csv(
            imd.mosdef_dir + '/axis_ratio_data/Merged_catalogs/mosdef_' + cat_name[:-3] + 'csv', index=False)


def read_axis_ratio(filt, objs):
    '''Gets the axis ratio for a particular object or list of objects

    Parameters:
    filt (int): Filter to read, either 125, 140, or 160
    objs (list): list of tuples of the form (field, v4id) to get the axis ratios for

    Returns:
    final_df (pd.DataFrame): Dataframe with the objects in order, containing axis ratios and other info from galfit
    '''
    ar_cat = ascii.read(imd.mosdef_dir + '/axis_ratio_data/Merged_catalogs/' + f'mosdef_F{filt}W_galfit_v4.0.csv').to_pandas()
    ar_cat = ar_cat.rename(
        columns={'NUMBER': 'v4id', 'q': 'axis_ratio', 'dq': 'err_axis_ratio'})

    rows = []
    for obj in objs:
        field = obj[0]
        v4id = obj[1]
        cat_obj = ar_cat[np.logical_and(
            ar_cat['v4id'] == v4id, ar_cat['FIELD'] == field)]
        rows.append(cat_obj)
    final_df = pd.concat(rows)
    return final_df
