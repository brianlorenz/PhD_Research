# Writes a text file with a list of groups that have too few galaxies or unusable data

from astropy.io import ascii
import pandas as pd
import initialize_mosdef_dirs as imd


def generate_skip_file():
    """Makes the file that lists which groups have too few galaxies
    
    """
    n_agn_df = ascii.read(imd.number_agn_file).to_pandas()
    bad_groups_df =  n_agn_df[n_agn_df['n_gals']<=10]['groupID']
    bad_groups_df.to_csv(imd.bad_groups_file, index=False)

