# Writes a text file with a list of groups that have too few galaxies or unusable data

from astropy.io import ascii
import pandas as pd
import initialize_mosdef_dirs as imd


def generate_skip_file(n_groups):
    """Makes the file that removes groups until you have 20 (savio node size)
    
    """
    n_agn_df = ascii.read(imd.number_agn_file).to_pandas()
    sorted_agn_df = n_agn_df.sort_values('n_gals')
    bad_groups_df =  sorted_agn_df.iloc[:-20]['groupID']
    bad_groups_df.to_csv(imd.bad_groups_file, index=False)

# generate_skip_file(23)