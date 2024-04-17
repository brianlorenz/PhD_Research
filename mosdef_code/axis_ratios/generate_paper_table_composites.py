# Makes a csv with the proper columns in order for latex
import numpy as np
import pandas as pd
import initialize_mosdef_dirs as imd
from astropy.io import ascii
import pickle

def generate_sed_paper_table():
    """Reads in summary_df and uses it to make a table for Latex
    
    """

    cluster_summary_df = ascii.read(imd.loc_cluster_summary_df).to_pandas()    
    with open(imd.cluster_dir + '/groupID_dict.pkl', 'rb') as f:
        groupID_dict = pickle.load(f)
    paper_ids = []
    for i in range(len(cluster_summary_df)):
        paperid = groupID_dict[cluster_summary_df.iloc[i]['groupID']]
        paper_ids.append(paperid)
    cluster_summary_df['paperID'] = paper_ids
    cluster_summary_df.to_csv(imd.loc_cluster_summary_df, index=False)

    
    paper_df = pd.DataFrame()
    paper_df['paperID'] = cluster_summary_df['paperID']
    paper_df['median_z'] = cluster_summary_df['redshift']

    paper_df['log(StellarMass)'] = cluster_summary_df['median_log_mass']

    paper_df['log(SFR)'] = cluster_summary_df['computed_log_sfr_with_limit']
    paper_df['err_log(SFR)_low'] = cluster_summary_df['err_computed_log_sfr_with_limit_low']
    paper_df['err_log(SFR)_high'] = cluster_summary_df['err_computed_log_sfr_with_limit_high']
    
    
    paper_df['metallicity'] = cluster_summary_df['O3N2_metallicity']
    paper_df['err_metallictiy_low'] = cluster_summary_df['err_O3N2_metallicity_low']
    paper_df['err_metallictiy_high'] = cluster_summary_df['err_O3N2_metallicity_high']
    
    paper_df['Prospector_AV'] = cluster_summary_df['Prospector_AV_50']
    paper_df['Prospector_AV_low'] = cluster_summary_df['Prospector_AV_50']-cluster_summary_df['Prospector_AV_16']
    paper_df['Prospector_AV_high'] = cluster_summary_df['Prospector_AV_84']-cluster_summary_df['Prospector_AV_50']
    
    paper_df['balmer_dec'] = cluster_summary_df['balmer_dec']
    paper_df['err_balmer_dec_low'] = cluster_summary_df['err_balmer_dec_low']
    paper_df['err_balmer_dec_high'] = cluster_summary_df['err_balmer_dec_high']

    paper_df['U_V'] = cluster_summary_df['median_U_V']
    paper_df['V_J'] = cluster_summary_df['median_V_J']

    paper_df['Prospector_SFR'] = cluster_summary_df['log_Prospector_ssfr50_multiplied_normalized']
    paper_df['err_Prospector_SFR_low'] = cluster_summary_df['err_log_Prospector_ssfr50_multiplied_normalized_low']
    paper_df['err_Prospector_SFR_high'] = cluster_summary_df['err_log_Prospector_ssfr50_multiplied_normalized_high']

    paper_df['Prospector_met'] = cluster_summary_df['logzsol50']
    paper_df['err_Prospector_met_low'] = paper_df['Prospector_met'] - (cluster_summary_df['logzsol16'])
    paper_df['err_Prospector_met_high'] = (cluster_summary_df['logzsol84']) - paper_df['Prospector_met']

    for i in range(len(cluster_summary_df)):
        if cluster_summary_df.iloc[i]['flag_hb_limit'] == 1:
            # paper_df['err_Prospector_met_low'].iloc[i] = np.inf
            paper_df.loc[i, 'err_metallictiy_low'] = np.inf
            paper_df.loc[i, 'err_balmer_dec_high'] = np.inf
            paper_df.loc[i, 'err_log(SFR)_high'] = np.inf


    paper_df = paper_df.sort_values('paperID')
    

    f = open(imd.cluster_dir + f'/paper_figures/table_eline.tbl', "w")
    for i in range(len(paper_df)):
        row = paper_df.iloc[i]
        f.write(f"{int(row['paperID'])} & ${round(row['median_z'],2)}$ & ${round(row['log(StellarMass)'],2)}$ & ${round(row['log(SFR)'],2)}\pm_{{{round(row['err_log(SFR)_low'],2)}}}^{{{round(row['err_log(SFR)_high'],2)}}}$ & ${round(row['metallicity'],2)}\pm_{{{round(row['err_metallictiy_low'],2)}}}^{{{round(row['err_metallictiy_high'],2)}}}$ & ${round(row['balmer_dec'],2)}\pm_{{{round(row['err_balmer_dec_low'],2)}}}^{{{round(row['err_balmer_dec_high'],2)}}}$ & ${round(row['U_V'],2)}$ & ${round(row['V_J'],2)}$ \\\ \n")
        if i==(len(paper_df)-1):
            f.write('\\hline')
            # f.write(f"${round(row['log(StellarMass)'],2)}$ & ${round(row['log(SFR)'],2)}$ & ${round(row['AxisRatio'],2)}$ & ${round(row['median_z'],2)}$ & ${round(row['metallicity'],2)}\pm_{{{round(row['err_metallictiy_low'],2)}}}^{{{round(row['err_metallictiy_high'],2)}}}$ & ${round(row['median_AV'],2)}\pm{{{round(row['median_AV_std'],2)}}}$ & ${round(row['median_beta'],2)}\pm_{{{round(row['median_beta_low'],2)}}}^{{{round(row['median_beta_high'],2)}}}$ & ${round(row['balmer_dec'],2)}\pm_{{{round(row['err_balmer_dec_low'],2)}}}^{{{round(row['err_balmer_dec_high'],2)}}}$")
    f.close()

    f = open(imd.cluster_dir + f'/paper_figures/table_prospector.tbl', "w")
    for i in range(len(paper_df)):
        row = paper_df.iloc[i]
        f.write(f"{int(row['paperID'])} & ${round(row['Prospector_AV'],2)}\pm_{{{round(row['Prospector_AV_low'],2)}}}^{{{round(row['Prospector_AV_high'],2)}}}$ & ${round(row['Prospector_SFR'],2)}\pm_{{{round(row['err_Prospector_SFR_low'],2)}}}^{{{round(row['err_Prospector_SFR_high'],2)}}}$ & ${round(row['Prospector_met'],2)}\pm_{{{round(row['err_Prospector_met_low'],2)}}}^{{{round(row['err_Prospector_met_high'],2)}}}$ \\\ \n")
        if i==(len(paper_df)-1):
            f.write('\\hline')
            # f.write(f"${round(row['log(StellarMass)'],2)}$ & ${round(row['log(SFR)'],2)}$ & ${round(row['AxisRatio'],2)}$ & ${round(row['median_z'],2)}$ & ${round(row['metallicity'],2)}\pm_{{{round(row['err_metallictiy_low'],2)}}}^{{{round(row['err_metallictiy_high'],2)}}}$ & ${round(row['median_AV'],2)}\pm{{{round(row['median_AV_std'],2)}}}$ & ${round(row['median_beta'],2)}\pm_{{{round(row['median_beta_low'],2)}}}^{{{round(row['median_beta_high'],2)}}}$ & ${round(row['balmer_dec'],2)}\pm_{{{round(row['err_balmer_dec_low'],2)}}}^{{{round(row['err_balmer_dec_high'],2)}}}$")
    f.close()

generate_sed_paper_table()