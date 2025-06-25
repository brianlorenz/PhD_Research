from astropy.io import ascii

def read_merged_lineflux_cat():
    lineflux_df = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/phot_calcs/phot_merged_lineflux.csv').to_pandas()
    return lineflux_df

def read_phot_df():
    phot_df_loc = '/Users/brianlorenz/uncover/Data/generated_tables/phot_calcs/phot_linecoverage_ha_pab_paa.csv'
    phot_df = ascii.read(phot_df_loc).to_pandas()
    return phot_df

def read_line_sample_df(line_name):
    """
    
    line_name (str): Halpha, PaAlpha, or PaBeta
    """
    line_sample_df_loc = f'/Users/brianlorenz/uncover/Data/generated_tables/phot_calcs/phot_sample_select/{line_name}_sample.csv'
    line_sample_df = ascii.read(line_sample_df_loc).to_pandas()
    return line_sample_df