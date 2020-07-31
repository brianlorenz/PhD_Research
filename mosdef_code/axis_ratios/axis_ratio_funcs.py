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
from query_funcs import get_zobjs, get_zobjs_sort_nodup
import initialize_mosdef_dirs as imd
import cluster_data_funcs as cdf


def make_axis_ratio_catalogs():
    '''Makes the galfit_data.csv file that is used for all of our objects

    Parameters:

    Returns:
    '''

    field_names = ['AEGIS', 'COSMOS', 'GOODS-N', 'GOODS-S', 'UDS']
    cat_names = os.listdir(imd.mosdef_dir + '/axis_ratio_data/AEGIS/')

    zobjs = get_zobjs_sort_nodup()

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


def interpolate_axis_ratio():
    '''Use the redshift of each object to make a catalog of what the axis ratio would be at 5000 angstroms

    Parameters:

    Returns:
    '''
    zobjs = get_zobjs_sort_nodup()
    F125_df = read_axis_ratio(125, zobjs)
    F140_df = read_axis_ratio(140, zobjs)
    F160_df = read_axis_ratio(160, zobjs)
    F125_df[['F125_axis_ratio', 'F125_err_axis_ratio']
            ] = F125_df[['axis_ratio', 'err_axis_ratio']]
    F140_df[['F140_axis_ratio', 'F140_err_axis_ratio']
            ] = F140_df[['axis_ratio', 'err_axis_ratio']]
    F160_df[['F160_axis_ratio', 'F160_err_axis_ratio']
            ] = F160_df[['axis_ratio', 'err_axis_ratio']]
    all_axis_ratios_df = pd.concat([F125_df[['F125_axis_ratio', 'F125_err_axis_ratio']], F140_df[
                                   ['F140_axis_ratio', 'F140_err_axis_ratio']]], axis=1)
    all_axis_ratios_df = pd.concat([all_axis_ratios_df, F160_df[
                                   ['F160_axis_ratio', 'F160_err_axis_ratio']]], axis=1)
    zs = []
    fields = []
    v4ids = []
    for obj in zobjs:
        mosdef_obj = get_mosdef_obj(obj[0], obj[1])
        zs.append(mosdef_obj['Z_MOSFIRE'])
        fields.append(obj[0])
        v4ids.append(obj[1])
    all_axis_ratios_df['Z_MOSFIRE'] = zs

    # If 5000 angstrom is outside of the range of observations, use the closest measurement
    # If 5000 is in the range, interpolate
    # DOESNT HANDLE NEGATIVE VALUES WELL
    # NOT SURE WHAT TO DO WITH ERRORS. IS THIS EVEN GOOD? SHOULD I IGNORE F140
    use_ratios = []
    use_errors = []
    for i in range(len(all_axis_ratios_df)):
        F125W_peak = 12471.0
        F140W_peak = 13924.0
        F160W_peak = 15396.0
        peaks = [F125W_peak, F140W_peak, F160W_peak]
        rest_waves = np.array([peak / (1 + all_axis_ratios_df.iloc[i]['Z_MOSFIRE'])
                               for peak in peaks])
        filters = ['125', '140', '160']
        axis_ratios = np.array([all_axis_ratios_df.iloc[i][f'F{j}_axis_ratio'] for j in filters])
        axis_errors = np.array([all_axis_ratios_df.iloc[i][f'F{j}_err_axis_ratio'] for j in filters])
        good_ratios = [ratio > -0.1 for ratio in axis_ratios]
        if len(axis_ratios[good_ratios]) == 1:
            use_ratio = axis_ratios[good_ratios]
            use_error = axis_errors[good_ratios]
            use_ratios.append(float(use_ratio))
            use_errors.append(float(use_error))
            continue
        if len(axis_ratios[good_ratios]) == 0:
            use_ratio = -999.0
            use_error = -999.0
            use_ratios.append(float(use_ratio))
            use_errors.append(float(use_error))
            continue
        axis_interp = interpolate.interp1d(rest_waves[good_ratios], axis_ratios[good_ratios], fill_value=(
            axis_ratios[good_ratios][0], axis_ratios[good_ratios][-1]), bounds_error=False)
        err_interp = interpolate.interp1d(rest_waves[good_ratios], axis_errors[good_ratios], fill_value=(
            axis_errors[good_ratios][0], axis_errors[good_ratios][-1]), bounds_error=False)
        use_ratio = axis_interp(5000)
        use_error = err_interp(5000)
        use_ratios.append(float(use_ratio))
        use_errors.append(float(use_error))
    all_axis_ratios_df['use_ratio'] = use_ratios
    all_axis_ratios_df['err_use_ratio'] = use_errors
    all_axis_ratios_df['field'] = fields
    all_axis_ratios_df['v4id'] = v4ids
    all_axis_ratios_df.to_csv(
        imd.mosdef_dir + '/axis_ratio_data/Merged_catalogs/mosdef_all_cats.csv', index=False)


def read_interp_axis_ratio():
    merged_ar_df = ascii.read(imd.mosdef_dir +
                              '/axis_ratio_data/Merged_catalogs/mosdef_all_cats.csv').to_pandas()
    return merged_ar_df
