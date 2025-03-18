from astropy.io import ascii

def read_merged_lineflux_cat():
    lineflux_df = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/phot_calcs/phot_merged_lineflux.csv').to_pandas()
    return lineflux_df