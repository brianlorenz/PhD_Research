# Deals with thee linemeas_latest file, reading emission lines

import numpy as np
from astropy.io import ascii
from astropy.table import Table
from mosdef_obj_data_funcs import read_sed, read_mock_sed, get_mosdef_obj, read_composite_sed
import initialize_mosdef_dirs as imd



def setup_emission_df():
    """Converts the linemeas_latest.fits file into a pandas dataframe as a csvf for ease of use

    Parameters:

    Returns:
    """
    file = imd.mosdef_dir + '/linemeas_latest.fits'
    lines_df = Table.read(file, format='fits').to_pandas()
    lines_df['FIELD_STR'] = [lines_df.iloc[i]['FIELD'].decode(
        "utf-8").rstrip() for i in range(len(lines_df))]
    lines_df.to_csv(file.replace('.fits', '.csv'), index=False)
    return


def read_emission_df():
    """Reads-in the emission line data from linemeas

    Parameters:

    Returns:
    emission_df (pd.DataFrame): Dataframe containing emission line measurements and info
    """
    emission_df = ascii.read(imd.loc_linemeas).to_pandas()
    return emission_df


def get_emission_measurements(emission_df, obj):
    """Gets the row(s) corresponding to one object

    Parameters:
    emission_df (pd.DataFrame): Dataframe containing emission line measurements and info
    zobjs (tuple): tuple of the form (field, v4id)

    Returns:
    row (pd.DataFrame): Dataframe of emission measurements corresponding to the provided tuple
    """
    mosdef_obj = get_mosdef_obj(obj[0], obj[1])
    use_id = mosdef_obj['ID']
    field = mosdef_obj['FIELD_STR']
    row = emission_df[np.logical_and(
        emission_df['ID'] == use_id, emission_df['FIELD_STR'] == field)]
    return row


def print_O3_ratio():
    """Gets the ratio of O3 lines for all galaxies

    Parameters:

    Returns:
    """

    emission_df = read_emission_df()
    flux_5008 = emission_df['OIII5008_FLUX']
    flux_4960 = emission_df['OIII4960_FLUX']
    mask_5008 = flux_5008 > 0
    mask_4960 = flux_4960 > 0
    mask = np.logical_and(mask_5008, mask_4960)
    ratios = flux_5008[mask] / flux_4960[mask]
    [print(i) for i in ratios]
    filt = ratios < 6
    print(f'Median: {np.median(ratios)} \nStd Dev: {np.std(ratios)}')
    print(f'Median: {np.median(ratios[filt])} \nStd Dev: {np.std(ratios[filt])}')
    # Maybe want to stack with only the good ones here. Standard deviation is
    # still huge, some are wayyyy off
