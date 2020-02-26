# Contains a variety of funcitons for querying the mosdef catalogs
""" Includes:
get_zobjs() - gets the field and id for all objects with measured redshifts in mosdef
get_sed() - given a  field and id, gets the SED for an object
get_all_seds() - run this to generate seds, then run the next two functions to pick up the ones left out
get_nomatch_seds() - run this to get seds for objects that are not in the ZFOURGE catalog but are in COSMOS
get_duplicate_seds() - run this to pick up the objects that have multiple observations. Averages the spectra
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


def setup_transfer_script(mosdef_df=mosdef_df):
    """Sets up a script that transfers postage stamps and spectra of objects from pepper to local

    Parameters:
    mosdef_df (pd.Dataframe): See above, dataframe from read_data.py


    Returns:
    Nothing, but populates the script file
    """
    shorten_dict = {
        'AEGIS': 'ae',
        'GOODS-S': 'gs',
        'COSMOS': 'co',
        'GOODS-N': 'gn',
        'UDS': 'ud',
    }
    image_f = open(
        '/Users/galaxies-air/mosdef/HST_Images/transfer_images.sh', 'w')
    spec_f = open(
        '/Users/galaxies-air/mosdef/HST_Images/transfer_spectra.sh', 'w')
    for obj_num in range(len(mosdef_df)):
        image_f.write(f'scp blorenz@pepper.astro.berkeley.edu:/Users/mosdef/HST_Images/postage_stamps/{mosdef_df.iloc[obj_num]["FIELD_STR"]}_f160w_{mosdef_df.iloc[obj_num]["ID"]}.fits .\n')
        spec_f.write(f'scp blorenz@pepper.astro.berkeley.edu:/Users/mosdef/Data/Reduced/v0.5/{shorten_dict[mosdef_df.iloc[obj_num]["FIELD_STR"]]}######_f160w_{mosdef_df.iloc[obj_num]["ID"]}.fits .\n')
    image_f.close()
    spec_f.close()


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
        'UDS': 'uds.v1.5.8.cat',
        'GOODS-S_3DHST': 'goodss_3dhst.v4.1.cat',
        'COSMOS_3DHST': 'cosmos_3dhst.v4.1.cat',
        'UDS_3DHST': 'uds_3dhst.v4.1.cat'
    }
    # Read the data
    cat_location = '/Users/galaxies-air/mosdef/Catalogs/'+cat_dict[field]
    df = ascii.read(cat_location).to_pandas()
    return df


def get_sed(field, v4id, full_cats_dict=False):
    """Given a field and id, gets the photometry for a galaxy, including corrections. Queries 3DHST if in GOODS-N or AEGIS, otherwise matches ra/dec with ZFOURGE

    Parameters:
    field (string): name of the field of the object
    id (int): HST V4.1 id of the object
    full_cats_dict (set to list, optional): Set to the list of all read catalogs if they havebeen read elsewhere (e.g. looping over multiple objects)

    Returns:
    """
    # Read the appropriate set of filters
    str_3dhst = ''

    print(f'Getting SED for {field}, id={v4id}')

    # Here we split to create the obj - different based on if we need to match ra/dec or just look it up
    if field == 'GOODS-S' or field == 'UDS' or field == 'COSMOS':
        print('Matching with ZFOURGE')
        # Match the mosdef catalog on field and V4ID to get the correct row
        mosdef_obj = mosdef_df[np.logical_and(
            mosdef_df['FIELD_STR'] == field, mosdef_df['V4ID'] == v4id)].iloc[0]
        # There should be a unique match - exit with an error if not
        # if len(mosdef_obj) < 1:
        #     sys.exit('No match found on FIELD_STR and V4ID')
        # if len(mosdef_obj) > 1:
        #     f = open(
        #         '/Users/galaxies-air/mosdef/sed_csvs/sed_errors/duplicate_objs.txt', 'a')
        #     f.write(f'{field}, {v4id}\n')
        #     f.close()
        #     print('Could not find unique match on FIELD_STR and V4ID, skipping object')
        #     return None
        # Coordinates for the object
        if full_cats_dict:
            cat = full_cats_dict[field+str_3dhst]
        else:
            cat = read_cat(field+str_3dhst)
        mosdef_obj_coords = SkyCoord(
            mosdef_obj['RA']*u.deg, mosdef_obj['DEC']*u.deg)
        # Convert the catalog RA and DEC to coordintes
        cat_coords = SkyCoord(cat['ra']*u.deg, cat['dec']*u.deg)
        # Performs the matching here
        idx_match, d2d_match, d3d_match = mosdef_obj_coords.match_to_catalog_sky(
            cat_coords)
        # If it doesn't match, write it to a file, but move on
        if d2d_match > 8.33*10**-5*u.deg:
            print(f'Match is larger than 0.3 arcsec! Distance: {d2d_match}')
            f = open(
                '/Users/galaxies-air/mosdef/sed_csvs/sed_errors/zfourge_no_match.txt', 'a')
            f.write(f'{field}, {v4id}, {d2d_match}\n')
            f.close()
            # Add the _3DHST string so that it will use 3DHST when reading the catalog
            str_3dhst = '_3DHST'
            # Failed match, so we will need to re-read the cat later with the 3D_HST tag
            matched = False
        else:
            # If the match was successful, set matched to true since we have our obj and have read the cat
            obj = cat.iloc[int(idx_match)]
            matched = True
    else:
        # If not in ZFOURGE, match is false so it will read the obj and cat
        matched = False
    # If it is unmatched at this point, read the correct catalog and get the obj
    if matched == False:
        if full_cats_dict:
            cat = full_cats_dict[field+str_3dhst]
        else:
            cat = read_cat(field+str_3dhst)
        print('Calling 3DHST Catalog')
        obj = cat.loc[cat['id'] == v4id]
    filters_df = ascii.read('catalog_filters/' + field + str_3dhst +
                            '_filterlist.csv').to_pandas()
    if v4id == -9999:
        print(f'Error: ID of -9999')
        f = open(
            '/Users/galaxies-air/mosdef/sed_csvs/sed_errors/other_errors.txt', 'a')
        f.write(f'{field}, {v4id}\n')
        f.close()
        return None
    # Read off the fluxes from the catalog
    fluxes = [float(obj[filtname])
              for filtname in filters_df['filter_name']]
    errorfluxes = [float(obj[filtname.replace('f_', 'e_')])
                   for filtname in filters_df['filter_name']]
    flux_tuple = [(fluxes[i], errorfluxes[i]) for i in range(len(fluxes))]
    # Magnitude zeropoint conversion from: http://monoceros.astro.yale.edu/RELEASE_V4.0/Photometry/AEGIS/aegis_3dhst.v4.1.cats/aegis_readme.v4.1.txt
    convert_factor = 3.7325*10**(-30)
    # Convert from f_nu to f_lambda
    convert_lambda = 3*10**18
    f_lambda_tuple = [(convert_factor*convert_lambda*flux_tuple[i][0]/(filters_df.iloc[i]['peak_wavelength'])**2, convert_factor *
                       convert_lambda*flux_tuple[i][1]/(filters_df.iloc[i]['peak_wavelength'])**2) for i in range(len(flux_tuple))]
    sed = pd.DataFrame(f_lambda_tuple, columns=[
        'f_lambda', 'err_f_lambda'])
    # Concatenate with wavelength
    sed = sed.merge(filters_df,
                    left_index=True, right_index=True)
    flux_df = pd.DataFrame(zip(fluxes, errorfluxes), columns=[
        'flux_ab25', 'flux_error_ab25'])
    sed = sed.merge(flux_df,
                    left_index=True, right_index=True)
    # Continue to set the -99 to -99
    sed.loc[sed['flux_ab25'] == -99, 'f_lambda'] = -99
    sed.loc[sed['flux_ab25'] == -99, 'err_f_lambda'] = -99
    # Save with field nad id in the filename:
    sed.to_csv(f'/Users/galaxies-air/mosdef/sed_csvs/{field}_{v4id}{str_3dhst}_sed.csv', index=False)
    return sed


def get_all_seds(zobjs):
    """Given a field and id, gets the photometry for a galaxy, including corrections. Queries 3DHST if in GOODS-N or AEGIS, otherwise matches ra/dec with ZFOURGE

    Parameters:
    zobjs (list): Pass a list of tuples of the form (field, v4id)


    Returns:
    """

    # We prime the cats since we are looping over multiple objects

    full_cats_dict = {}
    print('Reading Catalogs')
    full_cats_dict['AEGIS'] = read_cat('AEGIS')
    full_cats_dict['GOODS-S'] = read_cat('GOODS-S')
    full_cats_dict['COSMOS'] = read_cat('COSMOS')
    full_cats_dict['GOODS-N'] = read_cat('GOODS-N')
    full_cats_dict['UDS'] = read_cat('UDS')
    full_cats_dict['GOODS-S_3DHST'] = read_cat('GOODS-S_3DHST')
    full_cats_dict['COSMOS_3DHST'] = read_cat('COSMOS_3DHST')
    full_cats_dict['UDS_3DHST'] = read_cat('UDS_3DHST')
    for obj in zobjs:
        field = obj[0]
        v4id = obj[1]
        get_sed(field, v4id, full_cats_dict)

    # Check the best FAST fit to the SED, check the fit
    # Plot the spectrum in there as well
    # So one window that has SED, spectrum, F160 image - nice overview of each galaxy. Maybe plot half-light radius?
    # What is unit of half-light radius?
    # Need to clean the error logs ech time before running the code
