#Manually import masses to save
import pandas as pd
import initialize_mosdef_dirs as imd
from astropy.io import ascii
import os
from balmer_avs import compute_balmer_av
from compute_cluster_sfrs import compute_cluster_sfrs

def save_masses(n_clusters, run_name):
    groupIDs = []
    for groupID in range(n_clusters):
        groupIDs.append(groupID)
    
    mass_list = [13.83,
                 -99,
                 12.64,
                 11.76,
                 13.79,
                 12.78,
                 12.53,
                 12.70,
                 14.38,
                 12.94,
                 12.79,
                 13.73,
                 11.80,
                 12.38,
                 12.49,
                 13.35,
                 12.39,
                 -99,
                 -99,
                 12.16
    ]

    mass_df = pd.DataFrame(zip(groupIDs, mass_list), columns=['groupID', 'prospector_log_mass'])
    mass_df.to_csv(imd.prospector_output_dir + f'/{run_name}_masses.csv', index=False)

def add_masses_to_cluster_summary_df(n_clusters, run_name):
    mass_df = ascii.read(imd.prospector_output_dir + f'/{run_name}_masses.csv').to_pandas()
    cluster_summary_df = imd.read_cluster_summary_df()
    halphas = []
    err_halphas = []
    halpha_lums = []
    err_halpha_lums = []
    hbetas = []
    err_hbetas = []
    balmer_decs = []
    err_balmer_dec_lows = []
    err_balmer_dec_highs = []
    balmer_avs = []
    O3N2_metallicitys = []
    err_O3N2_metallicity_lows = []
    err_O3N2_metallicity_highs = []
    for groupID in range(n_clusters):
        emission_df_loc = imd.prospector_emission_fits_dir + f'/{run_name}_emission_fits/{groupID}_emission_fits.csv'
        if os.path.exists(emission_df_loc):
            pro_emission_df = ascii.read(emission_df_loc).to_pandas()
            ha_row = pro_emission_df[pro_emission_df['line_name']=='Halpha']
            hb_row = pro_emission_df[pro_emission_df['line_name']=='Hbeta']
            halpha = ha_row.iloc[0]['flux']
            err_halpha = ha_row.iloc[0]['err_flux']
            halpha_lum = ha_row.iloc[0]['luminosity']
            err_halpha_lum = ha_row.iloc[0]['err_luminosity']
            hbeta = hb_row.iloc[0]['flux']
            err_hbeta = hb_row.iloc[0]['err_flux']
            balmer_dec = pro_emission_df.iloc[0]['balmer_dec']
            err_balmer_dec_low = pro_emission_df.iloc[0]['err_balmer_dec_low']
            err_balmer_dec_high = pro_emission_df.iloc[0]['err_balmer_dec_high']
            balmer_av = compute_balmer_av(balmer_dec)
            O3N2_metallicity = pro_emission_df.iloc[0]['O3N2_metallicity']
            err_O3N2_metallicity_low = pro_emission_df.iloc[0]['err_O3N2_metallicity_low']
            err_O3N2_metallicity_high = pro_emission_df.iloc[0]['err_O3N2_metallicity_high']
        else:
            halpha = -99
            err_halpha = -99
            halpha_lum = -99
            err_halpah_lum = -99
            hbeta = -99
            err_hbeta = -99
            balmer_dec = -99
            err_balmer_dec_low = -99
            err_balmer_dec_high = -99
            balmer_av = -99
            O3N2_metallicity = -99
            err_O3N2_metallicity_low = -99
            err_O3N2_metallicity_high = -99

        halphas.append(halpha)
        err_halphas.append(err_halpha)
        halpha_lums.append(halpha_lum)
        err_halpha_lums.append(err_halpha_lum)
        hbetas.append(hbeta)
        err_hbetas.append(err_hbeta)
        balmer_decs.append(balmer_dec)
        err_balmer_dec_lows.append(err_balmer_dec_low)
        err_balmer_dec_highs.append(err_balmer_dec_high)
        balmer_avs.append(balmer_av)
        O3N2_metallicitys.append(O3N2_metallicity)
        err_O3N2_metallicity_lows.append(err_O3N2_metallicity_low)
        err_O3N2_metallicity_highs.append(err_O3N2_metallicity_high)

    cluster_summary_df['prospector_log_mass'] = mass_df['prospector_log_mass']
    cluster_summary_df['prospector_halpha_flux'] = halphas
    cluster_summary_df['err_prospector_halpha_flux'] = err_halphas
    cluster_summary_df['prospector_halpha_luminosity'] = halpha_lums
    cluster_summary_df['err_prospector_halpha_luminosity'] = err_halpha_lums
    cluster_summary_df['prospector_hbeta_flux'] = hbetas
    cluster_summary_df['err_prospector_hbeta_flux'] = err_hbetas
    cluster_summary_df['prospector_balmer_dec'] = balmer_decs
    cluster_summary_df['err_prospector_balmer_dec_low'] = err_balmer_dec_lows
    cluster_summary_df['err_prospector_balmer_dec_high'] = err_balmer_dec_highs
    cluster_summary_df['prospector_balmer_av'] = balmer_avs
    cluster_summary_df['prospector_O3N2_metallicity'] = O3N2_metallicitys
    cluster_summary_df['err_prospector_O3N2_metallicity_low'] = err_O3N2_metallicity_lows
    cluster_summary_df['err_prospector_O3N2_metallicity_high'] = err_O3N2_metallicity_highs
    cluster_summary_df.to_csv(imd.loc_cluster_summary_df, index=False)
    compute_cluster_sfrs(prospector=True)    

# save_masses(20, 'first_test_20groups')
# add_masses_to_cluster_summary_df(20, 'first_test_20groups')