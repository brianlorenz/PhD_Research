# Uses the .translate files to create filter lists for each of the catalogs

import sys
import os
import string
import numpy as np
import pandas as pd
import re
from astropy.io import ascii
import initialize_mosdef_dirs as imd


# Location of folders for the surveys is in imd

# Dictionary of locations for the translate files
# translate_dict = {
#     'AEGIS': imd.loc_3DHST + 'AEGIS/aegis_3dhst.v4.1.translate',
#     'GOODS-S': imd.loc_ZFOURGE + 'cdfs/cdfs.v1.6.9.translate',
#     'COSMOS': imd.loc_ZFOURGE + 'cosmos/cosmos.v1.3.6.translate',
#     'GOODS-N': imd.loc_3DHST + 'GOODS-N/GOODS-N_3dhst.v4.1.translate',
#     'UDS': imd.loc_ZFOURGE + 'uds/uds.v1.5.8.translate',
#     'GOODS-S_3DHST': imd.loc_3DHST + 'GOODS-S/goodss_3dhst.v4.1.translate',
#     'COSMOS_3DHST': imd.loc_3DHST + 'COSMOS/cosmos_3dhst.v4.1.translate',
#     'UDS_3DHST': imd.loc_3DHST + 'UDS/uds_3dhst.v4.1.translate'
# }
folder_loc = '/Users/brianlorenz/jwst_sfgalaxy/data/catalog'
translate_file = folder_loc + '/UVISTA_DR3_master_v1.1.translate'

'''
Ran this in unix to create the overview file
> more FILTER.RES.latest | grep lambda_c. > overview
> more overview
A few lines (308-313) gave issues with units of um. Manually changed these to e+04.
'''

# Location of the overview file
loc_overview = folder_loc + '/overview'

df = pd.read_csv(loc_overview, header=None, sep='\n')


df = df[0].str.split('_c=', expand=True)
df = df[1].str.split(' AB', expand=True)


# These are the peak wavelengths we want
filter_df = pd.to_numeric(df[0])
'''filter_df = pd.DataFrame(zip(filter_list, centers), columns=[
    'filter_num', 'peak_wavelength'])
'''
filter_df.to_csv(folder_loc + '/suspense_filter_df.csv', header=False)


def make_filter_csv():
    """Uses the translate file to produce a catalog with the appropriate filters

    Parameters:

    Returns:
    Nothing, but saves the dataframe as a csv to be read by other programs
    """
    # Read in the translate file
    translate_df = ascii.read(translate_file, data_start=0).to_pandas()
    translate_df.columns = ['filter_name', 'filter_num']
    # Selecting only the columfs with F - all the others start with E (error)
    translate_df = translate_df[translate_df['filter_num'].str.startswith(
        'F')]
    # Drops the F, leaving just the numbers
    translate_df['filter_num'] = translate_df.filter_num.str.replace('F', '')
    # Reset the indexing
    translate_df.reset_index(drop=True, inplace=True)
    # Have to subtract 1 since the catalog is 1 indexed and python uses 0-index
    peak_waves = [filter_df[filter_df.index ==
                            (int(translate_df['filter_num'].iloc[i]) - 1)] for i in range(len(translate_df))]
    # Change data type to float
    peak_waves = [float(i) for i in peak_waves]
    peak_waves_df = pd.DataFrame(peak_waves, columns=['peak_wavelength'])
    filter_csv_df = translate_df.merge(
        peak_waves_df, left_index=True, right_index=True)
    # Save the dataframe
    filter_csv_df.to_csv(folder_loc + '/suspense_filterlist.csv', index=False)



# make_filter_csv()
### Next need to read in the photometry, follow along in query_funcs
