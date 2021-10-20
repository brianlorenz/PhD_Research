from astropy.io import ascii
import pandas as pd
import initialize_mosdef_dirs as imd

def read_FAST_file(field, id):
    fast_file_df = ascii.read(imd.FAST_dir + f'/{field}_BEST_FITS/{field}_v4.1_zall.fast_{id}.fit').to_pandas()
    fast_file_df.columns = ['wavelength', 'f_lambda']
    fast_file_df['f_lambda'] =  fast_file_df['f_lambda'] * 10**-19
    return fast_file_df