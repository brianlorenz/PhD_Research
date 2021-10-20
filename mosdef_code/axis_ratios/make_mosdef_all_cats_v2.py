import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import initialize_mosdef_dirs as imd
import cluster_data_funcs as cdf
from mosdef_obj_data_funcs import get_mosdef_obj

def make_mosdef_all_cats_2():
    '''Adds measurement of balmer decrement to the mosdef_all_cats.csv file generated in axis ratio funcs
    
    '''
    all_cats_df = ascii.read(imd.mosdef_dir + '/axis_ratio_data/Merged_catalogs/mosdef_all_cats.csv').to_pandas()
    linemeas_df = ascii.read(imd.loc_linemeas).to_pandas()
    
    
    fields = []
    v4ids = []

    agn_flags = []
    masses = []
    err_l_masses = []
    err_h_masses = []
    sfrs = []
    err_sfrs = []
    res = []
    err_res = []
    
    hb_values = []
    hb_errs = []
    ha_values = []
    ha_errs = []

    zs = []
    z_quals = []
    

    # Add the sfrs from the sfr_latest catalog
    dat = Table.read(imd.loc_sfrs_latest, format='fits')
    sfrs_df = dat.to_pandas()
    sfrs_df['FIELD_STR'] = [sfrs_df.iloc[i]['FIELD'].decode("utf-8").rstrip() for i in range(len(sfrs_df))]

        


    #Loop through the catalog and find the ha and hb value for each galaxy
    for i in range(len(all_cats_df)):
        v4id = all_cats_df.iloc[i]['v4id']
        field = all_cats_df.iloc[i]['field']
        v4ids.append(v4id)
        fields.append(field)

        obj = get_mosdef_obj(field, v4id)
        cat_id = obj['ID']

        agn_flags.append(obj['AGNFLAG'])
        masses.append(obj['LMASS'])
        err_l_masses.append(obj['L68_LMASS'])
        err_h_masses.append(obj['U68_LMASS'])
        res.append(obj['RE'])
        err_res.append(obj['DRE'])
        zs.append(obj['Z_MOSFIRE'])
        z_quals.append(obj['Z_MOSFIRE_ZQUAL'])

        linemeas_slice = np.logical_and(linemeas_df['ID']==cat_id, linemeas_df['FIELD_STR']==field)
        sfrs_slice = np.logical_and(sfrs_df['ID']==cat_id, sfrs_df['FIELD_STR']==field)
        
        hb_values.append(linemeas_df[linemeas_slice].iloc[0]['HB4863_FLUX'])
        hb_errs.append(linemeas_df[linemeas_slice].iloc[0]['HB4863_FLUX_ERR'])
        ha_values.append(linemeas_df[linemeas_slice].iloc[0]['HA6565_FLUX'])
        ha_errs.append(linemeas_df[linemeas_slice].iloc[0]['HA6565_FLUX_ERR'])


        # USE SFR2 or SFR_CORR?
        sfrs.append(sfrs_df[sfrs_slice].iloc[0]['SFR2'])
        err_sfrs.append(sfrs_df[sfrs_slice].iloc[0]['SFRERR2'])

    



    to_merge_df = pd.DataFrame(zip(fields, v4ids, agn_flags, masses, err_l_masses, err_h_masses, sfrs, err_sfrs, res, err_res, zs, z_quals, hb_values, hb_errs, ha_values, ha_errs), columns=['field', 'v4id', 'agn_flag', 'log_mass', 'err_log_mass_d', 'err_log_mass_u', 'sfr', 'err_sfr', 'half_light', 'err_half_light', 'z', 'z_qual_flag', 'hb_flux', 'err_hb_flux', 'ha_flux', 'err_ha_flux'])
    merged_all_cats = all_cats_df.merge(to_merge_df, left_on=['v4id', 'field'], right_on=['v4id', 'field'])
    merged_all_cats.to_csv(imd.loc_axis_ratio_cat, index=False)


make_mosdef_all_cats_2()