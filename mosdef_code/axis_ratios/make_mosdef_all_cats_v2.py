import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import initialize_mosdef_dirs as imd
import cluster_data_funcs as cdf
from mosdef_obj_data_funcs import get_mosdef_obj
from mosdef_obj_data_funcs import setup_get_AV, get_AV
from read_data import read_file       
from flag_galaxies_by_axis_ratio import flag_axis_ratios    
from compute_new_sfrs import convert_ha_to_sfr, add_use_sfr                                                        


def main():
    """Runs all the functions needed to update the catalog with new information"""
    make_mosdef_all_cats_2()
    flag_axis_ratios()
    convert_ha_to_sfr()
    add_use_sfr()
    flag_axis_ratios()


def make_mosdef_all_cats_2():
    '''Adds measurement of balmer decrement to the mosdef_all_cats.csv file generated in axis ratio funcs
    
    '''
    all_cats_df = ascii.read(imd.mosdef_dir + '/axis_ratio_data/Merged_catalogs/mosdef_all_cats.csv').to_pandas()
    linemeas_df = ascii.read(imd.loc_linemeas).to_pandas()
    line_eq_df = ascii.read(imd.loc_eqwidth_cat, header_start=0, data_start=1).to_pandas()
    
    fields = []
    v4ids = []

    agn_flags = []
    masses = []
    err_l_masses = []
    err_h_masses = []
    sfrs = []
    err_sfrs = []
    sfrs_corrs = []
    ha_det_sfrs = []
    hb_det_sfrs = []
    res = []
    err_res = []
    
    hb_values = []
    hb_errs = []
    ha_values = []
    ha_errs = []

    zs = []
    z_quals = []

    avs = []

    betas = []

    logoh_pps = []
    logoh_pps_ulims = []
    logoh_pps_llims = []
    n2_lim_flag = []

    eqwidth_has = []
    err_eqwidth_has = []
    eqwidth_hbs = []
    err_eqwidth_hbs = []

    #mips data
    mips_fluxs = [] 
    err_mips_fluxes = []
    mips_corr24s = []
    mips_exp24s = []
    

    # Add the sfrs from the sfr_latest catalog
    dat = Table.read(imd.loc_sfrs_latest, format='fits')
    sfrs_df = dat.to_pandas()
    sfrs_df['FIELD_STR'] = [sfrs_df.iloc[i]['FIELD'].decode("utf-8").rstrip() for i in range(len(sfrs_df))]

    betas_df = read_file(imd.mosdef_dir + '/Mosdef_cats/mosdef_sfrs_v0.5_calz_beta.fits')
    betas_df['FIELD_STR'] = [betas_df.iloc[i]['FIELD'].decode("utf-8").rstrip() for i in range(len(betas_df))]

    metals_df = read_file(imd.mosdef_dir + '/Mosdef_cats/mosdef_metallicity_latest.fits')
    metals_df['FIELD_STR'] = [metals_df.iloc[i]['FIELD'].decode("utf-8").rstrip() for i in range(len(metals_df))]

    mosdef_ir_df = read_file(imd.loc_ir_latest)
    mosdef_ir_df['FIELD_STR'] = [mosdef_ir_df.iloc[i]['FIELD'].decode("utf-8").rstrip() for i in range(len(mosdef_ir_df))]

    #### Sidenote, some objects are missing
    # np.sum((sfrs_df['ID'] - betas_df['ID']) != 0)    
    # sfrs_df[(sfrs_df['ID'] - betas_df['ID']) != 0]
    # Currently just skipping beta for these objects
    skip_beta_ids = sfrs_df[(sfrs_df['ID'] - betas_df['ID']) != 0]['ID']
    skip_beta_ids = [skip_beta_ids.iloc[i] for i in range(len(skip_beta_ids))]


    fields, av_dfs = setup_get_AV() # Prepares to search for AV value


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
        betas_slice = np.logical_and(betas_df['ID']==cat_id, betas_df['FIELD_STR']==field)
        metals_slice = np.logical_and(metals_df['ID']==cat_id, metals_df['FIELD_STR']==field)
        line_eq_slice = np.logical_and(line_eq_df['ID']==cat_id, line_eq_df['FIELD']==field)
        
        hb_values.append(linemeas_df[linemeas_slice].iloc[0]['HB4863_FLUX'])
        hb_errs.append(linemeas_df[linemeas_slice].iloc[0]['HB4863_FLUX_ERR'])
        ha_values.append(linemeas_df[linemeas_slice].iloc[0]['HA6565_FLUX'])
        ha_errs.append(linemeas_df[linemeas_slice].iloc[0]['HA6565_FLUX_ERR'])

        avs.append(get_AV(fields, av_dfs, obj))
        
        if cat_id in skip_beta_ids:
            betas.append(-999)
            logoh_pps.append(-999)
            logoh_pps_ulims.append(-999)
            logoh_pps_llims.append(-999)
            n2_lim_flag.append(-999)
        else:
            betas.append(betas_df[betas_slice].iloc[0]['BETAPHOT'])
            
            # Metallicities
            logoh_pps.append(metals_df[metals_slice].iloc[0]['12LOGOH_PP04_N2'])
            logoh_pps_ulims.append(metals_df[metals_slice].iloc[0]['U68_12LOGOH_PP04_N2'])
            logoh_pps_llims.append(metals_df[metals_slice].iloc[0]['L68_12LOGOH_PP04_N2'])
            n2_lim_flag.append(metals_df[metals_slice].iloc[0]['N2_LIMFLAG'])  # See readme for more info about columns
            

        # USE SFR2 or SFR_CORR? - SFR2 contains upper limits, SFR CORR has only hbeta detections that are good. See readme
        sfrs.append(sfrs_df[sfrs_slice].iloc[0]['SFR2'])
        err_sfrs.append(sfrs_df[sfrs_slice].iloc[0]['SFRERR2'])
        sfrs_corrs.append(sfrs_df[sfrs_slice].iloc[0]['SFR_CORR'])
        ha_det_sfrs.append(sfrs_df[sfrs_slice].iloc[0]['HA6565_DETFLAG'])
        hb_det_sfrs.append(sfrs_df[sfrs_slice].iloc[0]['HB4863_DETFLAG'])

        # Equivalent Widths
        # Catalog does not have all galaxies - input a -99 (instead of -999) when it is not in the catalog
        if len(line_eq_df[line_eq_slice]) == 0:
            eqwidth_has.append(-99)
            err_eqwidth_has.append(-99)
            eqwidth_hbs.append(-99)
            err_eqwidth_hbs.append(-99)

        else:
            eqwidth_has.append(line_eq_df[line_eq_slice].iloc[0]['WHA'])
            err_eqwidth_has.append(line_eq_df[line_eq_slice].iloc[0]['WHAERR'])
            eqwidth_hbs.append(line_eq_df[line_eq_slice].iloc[0]['WHB'])
            err_eqwidth_hbs.append(line_eq_df[line_eq_slice].iloc[0]['WHBERR'])
        


        

    



    to_merge_df = pd.DataFrame(zip(fields, v4ids, agn_flags, masses, err_l_masses, err_h_masses, sfrs, err_sfrs, sfrs_corrs, ha_det_sfrs, hb_det_sfrs, res, err_res, zs, z_quals, avs, betas, hb_values, hb_errs, ha_values, ha_errs, logoh_pps, logoh_pps_ulims, logoh_pps_llims, n2_lim_flag, eqwidth_has, err_eqwidth_has, eqwidth_hbs, err_eqwidth_hbs), columns=['field', 'v4id', 'agn_flag', 'log_mass', 'err_log_mass_d', 'err_log_mass_u', 'sfr', 'err_sfr', 'sfr_corr', 'ha_detflag_sfr', 'hb_detflag_sfr', 'half_light', 'err_half_light', 'z', 'z_qual_flag', 'AV', 'beta', 'hb_flux', 'err_hb_flux', 'ha_flux', 'err_ha_flux', 'logoh_pp_n2', 'u68_logoh_pp_n2', 'l68_logoh_pp_n2', 'n2flag_metals', 'eq_width_ha', 'err_eq_width_ha', 'eq_width_hb', 'err_eq_width_hb'])
    merged_all_cats = all_cats_df.merge(to_merge_df, left_on=['v4id', 'field'], right_on=['v4id', 'field']) 
    merged_all_cats.to_csv(imd.loc_axis_ratio_cat, index=False)


main()