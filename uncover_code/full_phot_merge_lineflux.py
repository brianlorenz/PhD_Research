import numpy as np
from astropy.io import ascii
from full_phot_sample_selection import line_list
import pandas as pd

def combine_phot_flux_cats():
    flux_dfs = []
    for line in line_list:
        flux_df = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/phot_lineflux_{line[0]}.csv').to_pandas()
        flux_dfs.append(flux_df)
    for i in range(len(flux_dfs)-1):
        if i == 0:
            flux_df_merge = pd.merge(flux_dfs[0], flux_dfs[1], on='id_dr3', how='outer')
        else:
            flux_df_merge = pd.merge(flux_df_merge, flux_dfs[i+1], on='id_dr3', how='outer')
    flux_df_merge = flux_df_merge.sort_values(by='id_dr3', ascending=True)
    flux_df_merge.to_csv(f'/Users/brianlorenz/uncover/Data/generated_tables/phot_merged_lineflux.csv', index=False)
        
def check_overlap():
    ha_cat = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/phot_lineflux_Halpha.csv').to_pandas()
    pab_cat = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/phot_lineflux_PaBeta.csv').to_pandas()
    
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
    combine_phot_flux_cats()