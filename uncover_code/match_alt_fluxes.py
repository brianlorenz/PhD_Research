from uncover_read_data import read_supercat, make_pd_table_from_fits, get_id_msa_list
from astropy.io import ascii
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_vals import *

def alt_cross_match_sources():
    supercat_df = read_supercat()
    alt_cat = make_pd_table_from_fits('/Users/brianlorenz/uncover/Catalogs/alt_dr1.fits')
    alt_ra = alt_cat['ra']
    alt_dec = alt_cat['dec']
    # Ha columns are Ha_flux, Ha_flux_err, Ha_snr
    # Pab columns are PaB_flux, PaB_flux_err, PaB_snr
    
    ha_fluxes = []
    err_ha_fluxes = []
    ha_snrs = []
    pab_fluxes = []
    err_pab_fluxes = []
    pab_snrs = []
    matched_with_alt = []


    id_msa_list = get_id_msa_list(full_sample=False)
    for id_msa in id_msa_list:
        supercat_row = supercat_df[supercat_df['id_msa']==id_msa]

        target_ra = supercat_row['ra'].iloc[0]  
        target_dec = supercat_row['dec'].iloc[0]
        target_coord = SkyCoord(ra=target_ra * u.deg, dec=target_dec * u.deg)

        ra_array = alt_ra.to_numpy()
        dec_array = alt_dec.to_numpy()
        sky_coords = SkyCoord(ra=ra_array * u.deg, dec=dec_array * u.deg)
        separations = target_coord.separation(sky_coords)
        closest_index = np.argmin(separations)
        indices_sorted_near_to_far = np.argsort(separations) # None had possible secondary sources

        closest_object = alt_cat.iloc[closest_index]

        print(f"Separation {id_msa} closest:", separations[closest_index])
        threshold = 0.000833333*u.deg # 0.3 arcsec
        if separations[closest_index] > threshold:
            print(f'No match for {id_msa}')
            ha_fluxes.append(-99)
            err_ha_fluxes.append(-99)
            ha_snrs.append(-99)
            pab_fluxes.append(-99)
            err_pab_fluxes.append(-99)
            pab_snrs.append(-99)
            matched_with_alt.append(0)
            continue
        
        closest_object = closest_object.fillna(-99)

        if closest_object['Ha_flux'] > 0:
            breakpoint()
        ha_fluxes.append(closest_object['Ha_flux'])
        err_ha_fluxes.append(closest_object['Ha_flux_err'])
        ha_snrs.append(closest_object['Ha_snr'])
        pab_fluxes.append(closest_object['PaB_flux'])
        err_pab_fluxes.append(closest_object['PaB_flux_err'])
        pab_snrs.append(closest_object['PaB_snr'])
        matched_with_alt.append(1)
        
    alt_match_df = pd.DataFrame(zip(id_msa_list, ha_fluxes, err_ha_fluxes, ha_snrs, pab_fluxes, err_pab_fluxes, pab_snrs, matched_with_alt), columns=['id_msa', 'alt_ha_flux', 'alt_err_ha_flux', 'alt_ha_snr', 'alt_pab_flux', 'alt_err_pab_flux', 'alt_pab_snr', 'alt_has_match'])
    alt_match_df.to_csv('/Users/brianlorenz/uncover/Data/generated_tables/alt_match_catalog.csv', index=False)

def read_alt_cat():
    alt_df = ascii.read('/Users/brianlorenz/uncover/Data/generated_tables/alt_match_catalog.csv').to_pandas()
    return alt_df

def plot_alt_vs_uncover(fluxcal=True):
    alt_df = read_alt_cat()
    lineflux_df = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/lineflux_df.csv').to_pandas()
    lineflux_df = lineflux_df.merge(alt_df, on='id_msa') # add in the ALT info

    fig, axarr = plt.subplots(2,2, figsize=(12,12))
    ax_ha_spec = axarr[0, 0]
    ax_ha_sed= axarr[0, 1]
    ax_pab_spec = axarr[1, 0]
    ax_pab_sed = axarr[1, 1]

    ax_list = [ax_ha_spec, ax_pab_spec, ax_ha_sed, ax_pab_sed]

    for i in range(len(lineflux_df)):
        id_msa=lineflux_df['id_msa'].iloc[i]
        if fluxcal == False:
            fit_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting_no_fluxcal/{id_msa}_emission_fits.csv').to_pandas()
            save_str='_no_fluxcal'
        else:
            fit_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting/{id_msa}_emission_fits.csv').to_pandas()
            save_str=''

        # Emission fit data
        fit_ha_flux = fit_df['flux'].iloc[0]
        err_fit_ha_flux_low = fit_df['err_flux_low'].iloc[0]
        err_fit_ha_flux_high = fit_df['err_flux_high'].iloc[0]
        fit_pab_flux = fit_df['flux'].iloc[1]
        err_fit_pab_flux_low = fit_df['err_flux_low'].iloc[1]
        err_fit_pab_flux_high = fit_df['err_flux_high'].iloc[1]
        fit_err_ha = [[err_fit_ha_flux_low], [err_fit_ha_flux_high]]
        fit_err_pab = [[err_fit_pab_flux_low], [err_fit_pab_flux_high]]

        # breakpoint()

        if lineflux_df['alt_ha_flux'].iloc[i] > -99:
            ax_ha_sed.errorbar(lineflux_df['alt_ha_flux'].iloc[i], lineflux_df['ha_sed_flux'].iloc[i], xerr=lineflux_df['alt_err_ha_flux'].iloc[i], marker='o', color='black', ls='None')
            ax_ha_spec.errorbar(lineflux_df['alt_ha_flux'].iloc[i], fit_ha_flux, xerr=lineflux_df['alt_err_ha_flux'].iloc[i], yerr=fit_err_ha, marker='o', color='black', ls='None')
        if lineflux_df['alt_pab_flux'].iloc[i] > -99:
            ax_pab_sed.errorbar(lineflux_df['alt_pab_flux'].iloc[i], lineflux_df['pab_sed_flux'].iloc[i], xerr=lineflux_df['alt_err_pab_flux'].iloc[i], marker='o', color='black', ls='None')
            ax_pab_spec.errorbar(lineflux_df['alt_pab_flux'].iloc[i], fit_pab_flux, xerr=lineflux_df['alt_err_pab_flux'].iloc[i], yerr=fit_err_pab, marker='o', color='black', ls='None')
        # ax_pab.plot(lineflux_df['int_spec_pab_fecor'], lineflux_df['pab_sed_flux'], marker='o', color='black', ls='None')
    
    
    ax_ha_sed.set_ylabel('Photometric Ha flux', fontsize=14)
    ax_ha_sed.set_xlabel('ALT Ha flux', fontsize=14)
    ax_ha_spec.set_ylabel('Emission Fit Ha flux', fontsize=14)
    ax_ha_spec.set_xlabel('ALT Ha flux', fontsize=14)
    ax_pab_sed.set_ylabel('Photometric PaB flux', fontsize=14)
    ax_pab_sed.set_xlabel('ALT PaB flux', fontsize=14)
    ax_pab_spec.set_ylabel('Emission Fit PaB flux', fontsize=14)
    ax_pab_spec.set_xlabel('ALT PaB flux', fontsize=14)
    if fluxcal==False:
        ax_ha_spec.set_ylabel('Emission Fit Ha flux no aper cor', fontsize=14)
        ax_pab_spec.set_ylabel('Emission Fit PaB flux no aper cor', fontsize=14)
    
    # line_p1 = np.array([-20, -20])
    # line_p2 = np.array([-15, -15])
    # def get_distance(datapoint):
    #     distance = np.cross(line_p2-line_p1,datapoint-line_p1)/np.linalg.norm(line_p2-line_p1)
    #     return distance
    # ha_distances = []
    # pab_distances = []
    # for i in range(len(lineflux_df)):
    #     log_ha_datapoint = (np.log10(lineflux_df['int_spec_ha_nocor'].iloc[i]), np.log10(lineflux_df['ha_sed_flux'].iloc[i]))
    #     log_pab_datapoint = (np.log10(lineflux_df['int_spec_pab_fecor'].iloc[i]), np.log10(lineflux_df['pab_sed_flux'].iloc[i]))
    #     ha_distances.append(get_distance(np.array(log_ha_datapoint)))
    #     pab_dist = get_distance(np.array(log_pab_datapoint))
    #     if pd.isnull(pab_dist):
    #         continue
    #     pab_distances.append(pab_dist)
    # ha_distances = np.abs(ha_distances)
    # pab_distances = np.abs(pab_distances)
    # median_ha_offset = np.median(ha_distances)
    # scatter_ha_offset = np.std(ha_distances)
    # median_pab_offset = np.median(pab_distances)
    # scatter_pab_offset = np.std(pab_distances)

    # start_scatter_text_x = 0.02
    # start_scatter_text_y = 0.94
    # scatter_text_sep = 0.07
    # ax_ha.text(start_scatter_text_x, start_scatter_text_y, f'Offset: {median_ha_offset:0.2f}', transform=ax_ha.transAxes, fontsize=12)
    # ax_ha.text(start_scatter_text_x, start_scatter_text_y-scatter_text_sep, f'Scatter: {scatter_ha_offset:0.2f}', transform=ax_ha.transAxes, fontsize=12)
    # ax_pab.text(start_scatter_text_x, start_scatter_text_y, f'Offset: {median_pab_offset:0.2f}', transform=ax_pab.transAxes, fontsize=12)
    # ax_pab.text(start_scatter_text_x, start_scatter_text_y-scatter_text_sep, f'Scatter: {scatter_pab_offset:0.2f}', transform=ax_pab.transAxes, fontsize=12)
    

    for ax in ax_list:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(labelsize=14)
        ax.plot([1e-20, 1e-14], [1e-20, 1e-14], ls='--', color='red', marker='None')
        ax.set_xlim([3e-19, 1e-15])
        ax.set_ylim([3e-19, 1e-15])
        scale_aspect(ax)
    plt.tight_layout()
        
    fig.savefig(f'/Users/brianlorenz/uncover/Figures/ALT_compare/alt_compare_flux{save_str}.pdf', bbox_inches='tight')


if __name__ == '__main__':
    alt_cross_match_sources()
    # alt_df = read_alt_cat()
    plot_alt_vs_uncover()
    plot_alt_vs_uncover(fluxcal=False)
    # breakpoint()
    pass