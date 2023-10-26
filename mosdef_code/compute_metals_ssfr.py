from astropy.io import ascii
import initialize_mosdef_dirs as imd
import numpy as np
from axis_group_metallicities import compute_err_and_logerr, compute_O3N2_metallicity
import pandas as pd

def compute_metals(n_clusters, ignore_groups):
    groupIDs = []
    log_03_Hbs = []
    err_log_03_Hbs = []
    log_N2_Ha_measures = []
    err_log_N2_Ha_measures = []
    N2_Ha_metals = []
    err_N2_Ha_metals = []
    log_O3N2_measures = []
    err_log_O3N2_measures = []
    O3N2_metals = []
    err_O3N2_metals = []
    boot_err_O3N2_metal_lows = []
    boot_err_O3N2_metal_highs = []
    ha_fluxes = []
    hb_fluxes = []
    balmer_decs = []
    err_balmer_decs = []

    for groupID in range(n_clusters):
        groupIDs.append(groupID)
        
        if groupID in ignore_groups:
            log_03_Hbs.append(-99)
            err_log_03_Hbs.append(-99)
            log_N2_Ha_measures.append(-99)
            err_log_N2_Ha_measures.append(-99)
            log_O3N2_measures.append(-99)
            err_log_O3N2_measures.append(-99)
            O3N2_metals.append(-99)
            err_O3N2_metals.append(-99)
            boot_err_O3N2_metal_lows.append(-99)
            boot_err_O3N2_metal_highs.append(-99)
            continue
        
        fit_df = ascii.read(imd.emission_fit_csvs_dir + f'/{groupID}_emission_fits.csv').to_pandas()
        O3_5008_row = np.argmin(np.abs(fit_df['line_center_rest'] - 5008))
        O3_4960_row = np.argmin(np.abs(fit_df['line_center_rest'] - 4960))
        Hb_row = np.argmin(np.abs(fit_df['line_center_rest'] - 4862))
        Ha_row = np.argmin(np.abs(fit_df['line_center_rest'] - 6563))
        N2_6585_row = np.argmin(np.abs(fit_df['line_center_rest'] - 6585))

        ha_fluxes.append(fit_df.iloc[Ha_row]['flux'])
        hb_fluxes.append(fit_df.iloc[Hb_row]['flux'])
        balmer_decs.append(fit_df.iloc[Ha_row]['balmer_dec'])
        err_balmer_decs.append(fit_df.iloc[Ha_row]['err_balmer_dec'])

        O3_numerator = fit_df.iloc[O3_5008_row]['flux']+fit_df.iloc[O3_4960_row]['flux']
        err_O3_numerator = np.sqrt(fit_df.iloc[O3_5008_row]['err_flux']**2+fit_df.iloc[O3_4960_row]['err_flux']**2)
        O3_Hb_measure, log_O3_Hb_measure, err_O3_Hb_measure, err_log_O3_Hb_measure = compute_err_and_logerr(O3_numerator, fit_df.iloc[Hb_row]['flux'], err_O3_numerator, fit_df.iloc[Hb_row]['err_flux'])
        
        log_03_Hbs.append(log_O3_Hb_measure)
        err_log_03_Hbs.append(err_log_O3_Hb_measure)

        N2_Ha_measure, log_N2_Ha_measure, err_N2_Ha_measure, err_log_N2_Ha_measure = compute_err_and_logerr(fit_df.iloc[N2_6585_row]['flux'], fit_df.iloc[Ha_row]['flux'], fit_df.iloc[N2_6585_row]['err_flux'], fit_df.iloc[Ha_row]['err_flux'])

        log_N2_Ha_measures.append(N2_Ha_measure)
        err_log_N2_Ha_measures.append(err_N2_Ha_measure)

        O3N2_numerator, log_O3N2_numerator, err_O3N2_numerator, err_log_O3N2_numerator = compute_err_and_logerr(fit_df.iloc[O3_5008_row]['flux'], fit_df.iloc[Hb_row]['flux'], fit_df.iloc[O3_5008_row]['err_flux'], fit_df.iloc[Hb_row]['err_flux'])
        O3N2_measure, log_O3N2_measure, err_O3N2_measure, err_log_O3N2_measure = compute_err_and_logerr(O3N2_numerator, N2_Ha_measure, err_O3N2_numerator, err_N2_Ha_measure)

        O3N2_metal, err_O3N2_metal = compute_O3N2_metallicity(log_O3N2_measure, err_log_O3N2_measure)
        O3N2_metals.append(O3N2_metal)
        err_O3N2_metals.append(err_O3N2_metal)
        log_O3N2_measures.append(log_O3N2_measure)
        err_log_O3N2_measures.append(err_log_O3N2_measure)

        boot_err_O3N2_metal_lows.append(-99)
        boot_err_O3N2_metal_highs.append(-99)

    metals_df = pd.DataFrame(zip(groupIDs, log_03_Hbs, err_log_03_Hbs, log_N2_Ha_measures, err_log_N2_Ha_measures, log_O3N2_measures, err_log_O3N2_measures, O3N2_metals, err_O3N2_metals, boot_err_O3N2_metal_lows, boot_err_O3N2_metal_highs), columns=['groupID', 'log_03_Hb_measure', 'err_log_03_Hb_measure', 'log_N2_Ha_measure', 'err_log_N2_Ha_measure', 'log_O3N2_measure', 'err_log_O3N2_measure', 'O3N2_metallicity', 'err_O3N2_metallicity', 'boot_err_O3N2_metallicity_low', 'boot_err_O3N2_metallicity_high'])
    metals_df.to_csv(imd.cluster_dir + f'/cluster_metallicities.csv', index=False)

    cluster_summary_df = ascii.read(imd.loc_cluster_summary_df).to_pandas()

    cluster_summary_df = cluster_summary_df.merge(metals_df, on='groupID')

    breakpoint()
    cluster_summary_df.to_csv(imd.loc_cluster_summary_df, index=False)

def add_sanders_metallicity(ax):
    # Relation for O3N2 Computed metallicities form Sanders
    masses = np.arange(8,12,0.01)
    O3N2s = -0.94 * masses + 10.72
    O3N2_metallicities, errs_O3N2_metallicities = compute_O3N2_metallicity(O3N2s, -99*np.ones(len(O3N2s)))
    ax.plot(masses, O3N2_metallicities, ls='--', color='black', label='Sanders+ 2018')


# ignore_groups = [19]
# compute_metals(23, ignore_groups)