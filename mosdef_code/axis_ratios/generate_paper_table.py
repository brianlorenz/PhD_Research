# Makes a csv with the proper columns in order for latex
import numpy as np
import pandas as pd
import initialize_mosdef_dirs as imd
from astropy.io import ascii

def generate_paper_table(save_name):
    """Reads in summary_df and uses it to make a table for Latex
    
    """

    summary_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/summary.csv').to_pandas()
    
    zs = []
    has = []
    err_has = []
    hbs = []
    err_hbs = []
    for axis_group in range(len(summary_df)):
        group_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_group_dfs/{axis_group}_df.csv').to_pandas()
        zs.append(np.median(group_df['Z_MOSFIRE']))

        emission_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_emission_fits/{axis_group}_emission_fits.csv').to_pandas()
        has.append(emission_df.iloc[0]['flux'])
        err_has.append(emission_df.iloc[0]['err_flux']) 
        hbs.append(emission_df.iloc[1]['flux']) 
        err_hbs.append(emission_df.iloc[1]['err_flux']) 
    
    summary_df['median_z'] = zs
    summary_df['halpha_flux'] = has
    summary_df['err_halpha_flux'] = err_has
    summary_df['hbeta_flux'] = hbs
    summary_df['err_hbeta_flux'] = err_hbs


    summary_df['sorted_order'] = [0, 1, 2, 3, 4, 5, 6, 7]
    summary_df = summary_df.sort_values(by=['sorted_order'])

    symbols = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII']
    
    paper_df = pd.DataFrame()
    paper_df['axis_group'] = symbols
    paper_df['log(StellarMass)'] = summary_df['log_mass_median']

    paper_df['log(SFR)'] = summary_df['log_use_sfr_median']
    
    paper_df['AxisRatio'] = summary_df['use_ratio_median']
    
    paper_df['median_z'] = summary_df['median_z']
    
    paper_df['metallicity'] = summary_df['metallicity_median']
    paper_df['err_metallictiy_low'] = summary_df['err_metallicity_median_low']
    paper_df['err_metallictiy_high'] = summary_df['err_metallicity_median_high']
    
    paper_df['median_AV'] = summary_df['av_median']
    # paper_df['median_AV_low'] = summary_df['err_av_median_low']
    # paper_df['median_AV_high'] = summary_df['err_av_median_high']
    paper_df['median_AV_std'] = summary_df['err_av_median']
    
    paper_df['median_beta'] = summary_df['beta_median']
    paper_df['median_beta_low'] = summary_df['err_beta_median_low']
    paper_df['median_beta_high'] = summary_df['err_beta_median_high']
    
    # paper_df['halpha_flux'] = summary_df['halpha_flux']
    # paper_df['err_halpha_flux'] = summary_df['err_halpha_flux']
    
    # paper_df['hbeta_flux'] = summary_df['hbeta_flux']
    # paper_df['err_hbeta_flux'] = summary_df['err_hbeta_flux']
    paper_df['balmer_dec'] = summary_df['balmer_dec']
    paper_df['err_balmer_dec_low'] = summary_df['err_balmer_dec_low']
    paper_df['err_balmer_dec_high'] = summary_df['err_balmer_dec_high']
    paper_df.to_csv(imd.axis_cluster_data_dir + f'/{save_name}/axis_ratio_data.csv', index=False)
    
    f = open(imd.axis_cluster_data_dir + f'/{save_name}/axis_ratio_data.dat', "w")
    for i in range(len(paper_df)):
        row = paper_df.iloc[i]
        f.write(f"{row['axis_group']} & ${round(row['log(StellarMass)'],2)}$ & ${round(row['log(SFR)'],2)}$ & ${round(row['AxisRatio'],2)}$ & ${round(row['median_z'],2)}$ & ${round(row['metallicity'],2)}\pm_{{{round(row['err_metallictiy_low'],2)}}}^{{{round(row['err_metallictiy_high'],2)}}}$ & ${round(row['median_AV'],2)}\pm{{{round(row['median_AV_std'],2)}}}$ & ${round(row['median_beta'],2)}\pm_{{{round(row['median_beta_low'],2)}}}^{{{round(row['median_beta_high'],2)}}}$ & ${round(row['balmer_dec'],2)}\pm_{{{round(row['err_balmer_dec_low'],2)}}}^{{{round(row['err_balmer_dec_high'],2)}}}$  \\\ \n")
        if i==(len(paper_df)-1):
            f.write('\\hline')
            # f.write(f"${round(row['log(StellarMass)'],2)}$ & ${round(row['log(SFR)'],2)}$ & ${round(row['AxisRatio'],2)}$ & ${round(row['median_z'],2)}$ & ${round(row['metallicity'],2)}\pm_{{{round(row['err_metallictiy_low'],2)}}}^{{{round(row['err_metallictiy_high'],2)}}}$ & ${round(row['median_AV'],2)}\pm{{{round(row['median_AV_std'],2)}}}$ & ${round(row['median_beta'],2)}\pm_{{{round(row['median_beta_low'],2)}}}^{{{round(row['median_beta_high'],2)}}}$ & ${round(row['balmer_dec'],2)}\pm_{{{round(row['err_balmer_dec_low'],2)}}}^{{{round(row['err_balmer_dec_high'],2)}}}$")
    f.close()

generate_paper_table('norm_1_sn5_filtered')