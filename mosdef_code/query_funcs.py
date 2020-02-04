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
from astropy import units as u
from astropy.coordinates import SkyCoord


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
    # Uses V4ID for the 3DHST catalogs
    zobjs = [(df_filt.iloc[i]['FIELD_STR'], df_filt.iloc[i]['V4ID'])
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


def get_sed(field, v4id):
    """Given a field and id, gets the photometry for a galaxy, including corrections. Queries 3DHST if in GOODS-N or AEGIS, otherwise matches ra/dec with ZFOURGE

    Parameters:
    field (string): name of the field of the object
    id (int): HST V4.1 id of the object


    Returns:
    """
    # Read the appropriate set of filters
    filters_df = ascii.read('catalog_filters/'+field +
                            '_filterlist.csv').to_pandas()

    # Here we split to create the obj - different based on if we need to match ra/dec or just look it up
    if field == 'GOODS-N' or field == 'AEGIS':
        print('Calling 3DHST Catalog')
        # Reads in the catalog matching the field provided
        cat = read_cat(field)
        obj = cat.loc[cat['id'] == v4id]
    else:
        print('Matching with ZFOURGE')
        # Match the mosdef catalog on field and V4ID to get the correct row
        mosdef_obj = mosdef_df[np.logical_and(
            mosdef_df['FIELD_STR'] == field, mosdef_df['V4ID'] == v4id)]
        # There should be a unique match - exit with an error if not
        if len(mosdef_obj) < 1:
            sys.exit('No match found on FIELD_STR and V4ID')
        if len(mosdef_obj) > 1:
            sys.exit('Could not find unique match on FIELD_STR and V4ID')

        cat = read_cat(field)
        # Coordinates for the object
        mosdef_obj_coords = SkyCoord(
            mosdef_obj['RA']*u.deg, mosdef_obj['DEC']*u.deg)
        # Convert the catalog RA and DEC to coordintes
        cat_coords = SkyCoord(cat['ra']*u.deg, cat['dec']*u.deg)
        # Performs the matching here
        idx_match, d2d_match, d3d_match = mosdef_obj_coords.match_to_catalog_sky(
            cat_coords)
        if d2d_match > 2.78*10**-4*u.deg:
            print(f'Match is larger than one arcsec! Distance: {d2d_match}')
            sys.exit()
        obj = cat.iloc[idx_match]
    # Read off the fluxes from the catalog
    fluxes = [float(obj[filtname])
              for filtname in filters_df['filter_name']]
    print(fluxes)
    errorfluxes = [float(obj[filtname.replace('f_', 'e_')])
                   for filtname in filters_df['filter_name']]
    flux_tuple = [(fluxes[i], fluxes[i]-errorfluxes[i],
                   fluxes[i]+errorfluxes[i]) for i in range(len(fluxes))]
    # Magnitude zeropoint conversion from: http://monoceros.astro.yale.edu/RELEASE_V4.0/Photometry/AEGIS/aegis_3dhst.v4.1.cats/aegis_readme.v4.1.txt
    magnitude_tuples = [25.0-2.5*np.log10(flux) for flux in flux_tuple]
    sed = pd.DataFrame(magnitude_tuples, columns=[
                       'magnitude', 'magnitude_upper', 'magnitude_lower'])
    # Concatenate with wavelength
    sed = sed.merge(filters_df,
                    left_index=True, right_index=True)
    flux_df = pd.DataFrame(zip(fluxes, errorfluxes), columns=[
        'flux', 'flux_error'])
    sed = sed.merge(flux_df,
                    left_index=True, right_index=True)
    for tag in ['upper', 'lower']:
        sed['magnitude_err_' +
            tag] = abs(sed['magnitude_'+tag] - sed['magnitude'])
    return sed

    # Convert AB Mags to f_lambda
    # How do we deal with negative fluxes if we're supposed to immediately take the log?
    # UDS and GOODS-S Filter lists are breaking in the translate_filters code. They are looking for the 266th and 291st filters in the list, which don't exist
    # What to do with the few nan values? how to better display them on plot?

    # Check the best FAST fit to the SED, check the fit
    # Put the sed and two images next to each other from F160
    # Plot the spectrum in there as well
    # So one window that has SED, spectrum, F160 image - nice overview of each galaxy. Maybe plot half-light radius?
    # Images are under v4.0 of 3DHST, use the _orig_sci image
    # Need to add the x y position for the galaxies

    # Way down the road, scale to running multiple at once
