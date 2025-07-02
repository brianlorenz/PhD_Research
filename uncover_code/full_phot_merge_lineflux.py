import numpy as np
from astropy.io import ascii
from full_phot_sample_selection import line_list
import pandas as pd
from full_phot_read_data import read_merged_lineflux_cat
from full_phot_apply_flux_corrections import apply_fe_cor, apply_nii_cor, add_ew_column

def combine_phot_flux_cats():
    print('Merging Catalogs')
    flux_dfs = []
    for line in line_list:
        flux_df = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/phot_calcs/phot_lineflux_{line[0]}.csv').to_pandas()
        flux_dfs.append(flux_df)
    for i in range(len(flux_dfs)-1):
        if i == 0:
            flux_df_merge = pd.merge(flux_dfs[0], flux_dfs[1], on='id_dr3', how='outer')
        else:
            flux_df_merge = pd.merge(flux_df_merge, flux_dfs[i+1], on='id_dr3', how='outer')
    flux_df_merge = flux_df_merge.sort_values(by='id_dr3', ascending=True)

    ha_true = ~pd.isnull(flux_df_merge[f'use_flag_Halpha'])
    pab_true = ~pd.isnull(flux_df_merge[f'use_flag_PaBeta'])
    paa_true = ~pd.isnull(flux_df_merge[f'use_flag_PaAlpha'])
    flag_line_true = (3*ha_true) + (4*pab_true) + (5*paa_true)

    flux_df_merge['lines_measured'] = flag_line_true

    flux_df_merge.to_csv(f'/Users/brianlorenz/uncover/Data/generated_tables/phot_calcs/phot_merged_lineflux.csv', index=False)
    print('Merged and saved')

def count_objects():
    lineflux_df = read_merged_lineflux_cat()   
    breakpoint() 

    # Removes all bcg flagged objects
    lineflux_df = filter_bcg_flags(lineflux_df)

    




def filter_bcg_flags(lineflux_df):
    for line in line_list:
        lineflux_df = lineflux_df[lineflux_df[f'flag_reason_{line[0]}'] != '_bcg_flag/']
    return lineflux_df

def check_overlap():
    ha_cat = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/phot_calcs/phot_lineflux_Halpha.csv').to_pandas()
    pab_cat = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/phot_calcs/phot_lineflux_PaBeta.csv').to_pandas()
    
    count = 0
    for id_dr3 in pab_cat['id_dr3']:
        pab_row = pab_cat[pab_cat['id_dr3'] == id_dr3]
        ha_row = ha_cat[ha_cat['id_dr3'] == id_dr3]

        if len(ha_row) == 0:
            continue

        if pab_row['flag_reason_PaBeta'].iloc[0] == '_bcg_flag/':
            continue

        if ha_row['use_flag_Halpha'].iloc[0] == 0:
            continue

        count = count + 1

    breakpoint()

if __name__ == "__main__":
    
    apply_nii_cor() # Adds correction for nii line to halpha cat.  Also adds ew
    apply_fe_cor() # Adds correction for fe line to pabeta cat. Also adds ew
    add_ew_column('PaAlpha') # adds ew to dataframe
    combine_phot_flux_cats() # Merges Halpha, Pabeta, Paalpha cats

    # count_objects()
    pass