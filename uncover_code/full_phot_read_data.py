from astropy.io import ascii

def read_final_sample():
    sample_df = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/paper_data/final_sample.csv').to_pandas()
    return sample_df

def read_possible_sample():
    sample_df = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/paper_data/possible_sample.csv').to_pandas()
    return sample_df

def read_paper_df(df_name):
    df = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/paper_data/{df_name}.csv').to_pandas()
    return df

def read_merged_lineflux_cat():
    merge_lineflux_df = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/phot_calcs/phot_merged_lineflux.csv').to_pandas()
    return merge_lineflux_df

def read_lineflux_cat(line_name):
    df_loc = f'/Users/brianlorenz/uncover/Data/generated_tables/phot_calcs/phot_lineflux_{line_name}.csv'
    lineflux_df = ascii.read(df_loc).to_pandas()
    return lineflux_df, df_loc

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


def read_canucs_compare():
    import pandas as pd
    highz = ascii.read('/Users/brianlorenz/uncover/Catalogs/CANUCS/highz.dat').to_pandas()
    lowz = ascii.read('/Users/brianlorenz/uncover/Catalogs/CANUCS/lowz.dat').to_pandas()
    full_canucs = pd.concat([highz, lowz], axis=0)
    return full_canucs

# read_canucs_compare()