import numpy as np
from astropy.io import ascii

# mosdef_eqw_df = ascii.read('/Users/brianlorenz/uncover/Data/ha_pab_ews_mosdef/ews_simple.csv').to_pandas()
# mosdef_eqw_df = mosdef_eqw_df[mosdef_eqw_df['ha_eq_width']>0] # Filters out the -99
# median_ha_eqw = np.median(mosdef_eqw_df['ha_eq_width'])
# median_pab_eqw = np.median(mosdef_eqw_df['pab_eq_width'])

def compute_lineratio(ha_flux, pab_flux, ha_eqw_fit, pab_eqw_fit, absorption_ha_eqw, absorption_pab_eqw):
    # Eq width corrections for absorption
    ha_cor_frac = absorption_ha_eqw / ha_eqw_fit
    ha_flux_cor = ha_flux * (1-ha_cor_frac)
    pab_cor_frac = absorption_pab_eqw / pab_eqw_fit
    pab_flux_cor = pab_flux * (1-pab_cor_frac)
    lineratio = ha_flux_cor / pab_flux_cor
    return lineratio