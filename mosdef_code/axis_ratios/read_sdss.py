from astropy.table import Table
import initialize_mosdef_dirs as imd
import pandas as pd
from read_data import read_file
import matplotlib.pyplot as plt



def read_and_filter_sdss():
    extra_cat = imd.mosdef_dir + '/Catalogs/sdss_dr8/galSpecExtra-dr8.fits'
    extra_df = read_file(extra_cat)
    extra_df = extra_df[extra_df['LGM_TOT_P50']>=8]
    extra_df = extra_df[extra_df['LGM_TOT_P50']<=11]

    lines_cat = imd.mosdef_dir + '/Catalogs/sdss_dr8/galSpecLine-dr8.fits'
    lines_df = read_file(lines_cat)

    extra_df = extra_df.merge(lines_df, on='SPECOBJID')
    extra_df['log_mass'] = extra_df['LGM_TOT_P50']
    extra_df['balmer_dec'] = extra_df['H_ALPHA_FLUX']/extra_df['H_BETA_FLUX']
    extra_df = extra_df[extra_df['balmer_dec']>0]
    extra_df = extra_df[extra_df['balmer_dec']<100]

    

    return extra_df

# read_and_filter_sdss()