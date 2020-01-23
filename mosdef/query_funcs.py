# Contains a variety offuncitons for querying the mosdef catalogs
""" Includes:
get_zobjs() - gets the field and id for all objects with measured redshifts in mosdef
get_sed() - given a  field and id, gets the SED for an object
"""

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


def get_zobjs(df=mosdef_df):
    """Uses the mosdef_df to find all objects with measured redshifts

    Parameters:
    df (pd.DataFrame): dataframe containing mosdef objects (made in read_data.py)


    Returns:
    zobjs (list of tuples): containing the *field* and *id* of each object
        field (string): name of the field of the object, all caps, no whitespace. eg 'COSMOS' or 'GOODS-N'
        id (int): HST V4.1 id of the object
    """
    # Non-detections are set to -999
    df_filt = df[df['Z_MOSFIRE'] > 0]
    # Mess of string formatting is to change default 'bytes' type to string with no whitespace
    # Uses V4ID for the 3DHST catalogs
    zobjs = [(df_filt.iloc[i]['FIELD'].decode("utf-8").rstrip(), df_filt.iloc[i]['V4ID'])
             for i in range(len(df_filt))]
    return zobjs


def read_cat(field):
    """Read the catalog for the given read and put it into a dataframe

    Parameters:
    field (string): name of the field. eg 'COSMOS' or 'GOODS-N'


    Returns:
    df (pd.DataFrame): Dataframe containing the information from the table
    """
    cat_dict = {
        'AEGIS': 'aegis_3dhst.v4.1.cat',
        'GOODS-S': 'cdfs.v1.6.9.cat',
        'COSMOS': 'cosmos.v1.3.6.cat',
        'GOODS-N': 'goodsn_3dhst.v4.1.cat',
        'UDS': 'uds.v1.5.8.cat'
    }
    # Read the data
    cat_location = '/Users/galaxies-air/mosdef/'+cat_dict[field]
    df = ascii.read(cat_location).to_pandas()
    return df


def get_sed(field, id):
    """Given a field and id, gets the photometry for a galaxy, including corrections. Queries 3DHST if in GOODS-N or AEGIS, otherwise matches ra/dec with ZFOURGE

    Parameters:
    field (string): name of the field of the object
    id (int): HST V4.1 id of the object


    Returns:
    """
    if field == 'GOODS-N' or field == 'AEGIS':
        print('Calling 3DHST Catalog')
        # Reads in the catalog matching the field provided
        cat = read_cat(field)
        obj = cat.loc[cat['id'] == id]
        # Now read int he filter list for 3DHST
        filters_df = ascii.read('3DHST_filters.cat').to_pandas()
        # Read off the fluxes from the catalog
        fluxes = [float(obj['f_'+filtname])
                  for filtname in filters_df['filter']]
        print(fluxes)
        errorfluxes = [float(obj['e_'+filtname])
                       for filtname in filters_df['filter']]
        flux_tuple = [(fluxes[i], fluxes[i]-errorfluxes[i],
                       fluxes[i]+errorfluxes[i]) for i in range(len(fluxes))]
        magnitude_tuples = [25.0-2.5*np.log10(flux) for flux in flux_tuple]
        sed = pd.DataFrame(magnitude_tuples, columns=[
                           'magnitude', 'magnitude_upper', 'magnitude_lower'])
        # Concatenate with wavelength
        sed = sed.merge(filters_df['wavelength'],
                        left_index=True, right_index=True)
        for tag in ['upper', 'lower']:
            sed['magnitude_err_' +
                tag] = abs(sed['magnitude_'+tag] - sed['magnitude'])
        return sed

    else:
        print('Matching with ZFOURGE')
