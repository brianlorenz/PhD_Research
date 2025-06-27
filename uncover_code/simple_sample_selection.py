from uncover_sed_filters import unconver_read_filters, get_filt_cols
from uncover_read_data import read_supercat, read_spec_cat, read_lineflux_cat, get_id_msa_list, read_SPS_cat, read_SPS_cat_all
from astropy.io import ascii
from uncover_prospector_seds import make_all_prospector
from simple_make_dustmap import make_3color, get_line_coverage, ha_trasm_thresh, pab_trasm_thresh
from sedpy import observate
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from plot_vals import stellar_mass_label, scale_aspect, sfr_label
import numpy as np
from uncover_make_sed import read_sed


def sample_select(paalpha=False, paalpha_pabeta=False):
    if paalpha:
        save_dir = '/Users/brianlorenz/uncover/Data/sample_selection_paa/'
        paa_str = '_paa'
        emfit_dir = 'emission_fitting_paalpha'
        from fit_emission_uncover_paalpha import line_list
        pab_snr_thresh = 1 # Actually Paalpha
        ha_snr_thresh = 3


    elif paalpha_pabeta:
        save_dir = '/Users/brianlorenz/uncover/Data/sample_selection_paa_pab/'
        paa_str = '_paa_pab'
        emfit_dir = 'emission_fitting_paalpha_pabeta'
        from fit_emission_uncover_paalpha_pabeta import line_list
        ha_snr_thresh = 2
        pab_snr_thresh = 2 

    else:
        from fit_emission_uncover_wave_divide import line_list
        save_dir = '/Users/brianlorenz/uncover/Data/sample_selection/'
        paa_str = ''
        emfit_dir = 'emission_fitting'
        pab_snr_thresh = 2
        ha_snr_thresh = 3


    overlap_thresh = 0.2

    zqual_df = find_good_spec()
    zqual_df_covered = select_spectra(zqual_df, line_list, paa_only=paalpha)
    line_not_covered_id_msas = []
    for id_msa in zqual_df['id_msa']:
        if len(zqual_df_covered[zqual_df_covered['id_msa'] == id_msa]) == 0:
            line_not_covered_id_msas.append(id_msa)
    line_not_covered_id_msas_df = pd.DataFrame(line_not_covered_id_msas, columns=['id_msa'])
    line_not_covered_id_msas_df.to_csv(save_dir + 'line_not_in_filt.csv', index=False)
    zqual_df_covered.to_csv(f'/Users/brianlorenz/uncover/zqual_df_simple{paa_str}.csv', index=False)
    id_msa_list = zqual_df_covered['id_msa'].to_list()
    total_id_msa_list_df = pd.DataFrame(id_msa_list, columns=['id_msa'])
    total_id_msa_list_df.to_csv(save_dir + 'total_before_cuts.csv', index=False)
    lines_df = read_lineflux_cat()
    breakpoint()

    # Run emission fits / sed+spec generation on this group, then:
    # Need to fix the few that are not working, think it's emissoin fit

    id_msa_good_list = []
    id_msa_filt_edge = []
    id_msa_ha_snr_flag = []
    id_msa_pab_snr_flag = []
    id_msa_line_notfullcover = []
    id_msa_line_overlapcont = []
    id_msa_skipped = []

    cont_overlap_flag = []
    cont_overlap_value = []
    line_notfullcover_flag = []
    line_notfullcover_value = []

    good_ha_trasms = []
    good_pab_trasms  = []

    paa_covered = []
    for id_msa in id_msa_list:
        print(f"Checking sample selection for {id_msa}")
        if id_msa in [6325, 49991]: 
            print(f'Skipping {id_msa} for other issues')
            id_msa_skipped.append(id_msa)
            continue 
            #42041 - not in supercat
            #49991 - not in supercat
        # Read in the images
        ha_filters, ha_images, wht_ha_images, obj_segmap, ha_photfnus, ha_all_filts = make_3color(id_msa, line_index=0, plot=False, paalpha=paalpha, paalpha_pabeta=paalpha_pabeta)
        pab_filters, pab_images, wht_pab_images, obj_segmap, pab_photfnus, pab_all_filts = make_3color(id_msa, line_index=1, plot=False, paalpha=paalpha, paalpha_pabeta=paalpha_pabeta)
        # paa_filters, paa_images, wht_paa_images, obj_segmap, paa_photfnus, paa_all_filts = make_3color(id_msa, line_index=2, plot=False)

        ha_sedpy_name = ha_filters[1].replace('f', 'jwst_f')
        ha_sedpy_filt = observate.load_filters([ha_sedpy_name])[0]
        ha_filter_width = ha_sedpy_filt.rectangular_width
        pab_sedpy_name = pab_filters[1].replace('f', 'jwst_f')
        pab_sedpy_filt = observate.load_filters([pab_sedpy_name])[0]
        pab_filter_width = pab_sedpy_filt.rectangular_width

        ha_red_sedpy_name = ha_filters[0].replace('f', 'jwst_f')
        ha_red_sedpy_filt = observate.load_filters([ha_red_sedpy_name])[0]
        pab_red_sedpy_name = pab_filters[0].replace('f', 'jwst_f')
        pab_red_sedpy_filt = observate.load_filters([pab_red_sedpy_name])[0]
        ha_blue_sedpy_name = ha_filters[2].replace('f', 'jwst_f')
        ha_blue_sedpy_filt = observate.load_filters([ha_blue_sedpy_name])[0]
        pab_blue_sedpy_name = pab_filters[2].replace('f', 'jwst_f')
        pab_blue_sedpy_filt = observate.load_filters([pab_blue_sedpy_name])[0]

        ha_rest_wavelength = line_list[0][1]
        pab_rest_wavelength = line_list[1][1]


        

        # Emission fit properties
        fit_df = ascii.read(f'/Users/brianlorenz/uncover/Data/{emfit_dir}/{id_msa}_emission_fits.csv').to_pandas()
        ha_flux_fit = fit_df.iloc[0]['flux']
        pab_flux_fit = fit_df.iloc[1]['flux']
        ha_sigma = fit_df.iloc[0]['sigma'] # full width of the line
        pab_sigma = fit_df.iloc[1]['sigma'] # full width of the line
        ha_snr = fit_df['signal_noise_ratio'].iloc[0]
        pab_snr = fit_df['signal_noise_ratio'].iloc[1]
        redshift = zqual_df[zqual_df['id_msa']==id_msa]['z_spec'].iloc[0]

        # Sedona's catalog emfit properties
        lines_df_row = lines_df[lines_df['id_msa'] == id_msa]
        lines_df_ha_snr = lines_df_row['f_Ha+NII'].iloc[0] / lines_df_row['e_Ha+NII'].iloc[0]
        lines_df_pab_snr = lines_df_row['f_PaB'].iloc[0] / lines_df_row['e_PaB'].iloc[0]

        # Check the coverage fraction of the lines - we want it high in the line, but 0 in the continuum filters
        ha_avg_transmission = get_line_coverage(id_msa, ha_sedpy_filt, redshift, line_name='ha', paalpha=paalpha, paalpha_pabeta=paalpha_pabeta)
        pab_avg_transmission = get_line_coverage(id_msa, pab_sedpy_filt, redshift, line_name='pab', paalpha=paalpha, paalpha_pabeta=paalpha_pabeta)
        ha_red_avg_transmission = get_line_coverage(id_msa, ha_red_sedpy_filt, redshift, line_name='ha', paalpha=paalpha, paalpha_pabeta=paalpha_pabeta)
        pab_red_avg_transmission = get_line_coverage(id_msa, pab_red_sedpy_filt, redshift, line_name='pab', paalpha=paalpha, paalpha_pabeta=paalpha_pabeta)
        ha_blue_avg_transmission = get_line_coverage(id_msa, ha_blue_sedpy_filt, redshift, line_name='ha', paalpha=paalpha, paalpha_pabeta=paalpha_pabeta)
        pab_blue_avg_transmission = get_line_coverage(id_msa, pab_blue_sedpy_filt, redshift, line_name='pab', paalpha=paalpha, paalpha_pabeta=paalpha_pabeta)
        ha_transmissions = [ha_red_avg_transmission, ha_avg_transmission, ha_blue_avg_transmission]
        pab_transmissions = [pab_red_avg_transmission, pab_avg_transmission, pab_blue_avg_transmission]

        if pab_snr < pab_snr_thresh and id_msa not in [19179, 39855]:
            print(f"PaB SNR of {pab_snr} less than thresh of {pab_snr_thresh}")
            id_msa_pab_snr_flag.append(id_msa)
            continue
        if ha_snr < ha_snr_thresh:
            print(f"Ha SNR of {ha_snr} less than thresh of {ha_snr_thresh}")
            id_msa_ha_snr_flag.append(id_msa)
            continue
        # Visual
        if id_msa in [24219, 34506, 42360, 23890, 27862, 19981]:
            print(f"Visually looks bad for PaB SNR")
            id_msa_pab_snr_flag.append(id_msa)
            continue
        
        # If either halpha or pab is detected in the end filters, decide what to do
        if ha_all_filts == False or pab_all_filts == False:
            print("One of the lines not detected in all filters")
            print("Consider different cont measurement method")
            id_msa_filt_edge.append(id_msa)
            continue

        overlap_flag = ''
        if ha_red_avg_transmission > overlap_thresh:
            overlap_flag = 'ha_red'
            cont_overlap_value.append(ha_red_avg_transmission)
        elif ha_blue_avg_transmission > overlap_thresh:
            overlap_flag = 'ha_blue'
            cont_overlap_value.append(ha_blue_avg_transmission)
        elif pab_red_avg_transmission > overlap_thresh:
            overlap_flag = 'pab_red'
            cont_overlap_value.append(pab_red_avg_transmission)
        elif pab_blue_avg_transmission > overlap_thresh:
            overlap_flag = 'pab_blue'
            cont_overlap_value.append(pab_blue_avg_transmission)
        
        if overlap_flag != '':
            print("One of the lines overlaps the cont filter")
            id_msa_line_overlapcont.append(id_msa)
            cont_overlap_flag.append(overlap_flag)
            continue
            
        line_notfullcover_check = ''
        if ha_avg_transmission < ha_trasm_thresh: 
            line_notfullcover_check = 'ha'
            line_notfullcover_value.append(ha_avg_transmission)
        elif pab_avg_transmission < pab_trasm_thresh:
            line_notfullcover_check = 'pab'
            line_notfullcover_value.append(pab_avg_transmission)
        if line_notfullcover_check != '':
            print("One of the lines not covered fully in the filters")
            id_msa_line_notfullcover.append(id_msa)
            line_notfullcover_flag.append(line_notfullcover_check)
            continue
        

        id_msa_good_list.append(id_msa)
        good_ha_trasms.append(ha_avg_transmission)
        good_pab_trasms.append(pab_avg_transmission)
    
    assert len(id_msa_list) == len(id_msa_good_list) + len(id_msa_line_notfullcover) + len(id_msa_filt_edge) + len(id_msa_ha_snr_flag) + len(id_msa_pab_snr_flag) + len(id_msa_skipped) + len(id_msa_line_overlapcont)
    good_df = pd.DataFrame(zip(id_msa_good_list, good_ha_trasms, good_pab_trasms), columns=['id_msa', 'ha_trasm', 'pab_trasm'])
    line_notfullcover_df = pd.DataFrame(zip(id_msa_line_notfullcover, line_notfullcover_flag, line_notfullcover_value), columns=['id_msa', 'flag_line_coverage', 'line_trasm_value'])
    filt_edge_df = pd.DataFrame(id_msa_filt_edge, columns=['id_msa'])
    ha_snr_flag_df = pd.DataFrame(id_msa_ha_snr_flag, columns=['id_msa'])
    pab_snr_flag_df = pd.DataFrame(id_msa_pab_snr_flag, columns=['id_msa'])
    cont_overlap_df = pd.DataFrame(zip(id_msa_line_overlapcont, cont_overlap_flag, cont_overlap_value), columns=['id_msa', 'flag_cont_overlap', 'cont_overlap_value'])
    id_msa_skipped_df = pd.DataFrame(id_msa_skipped, columns=['id_msa'])

    # Write the DataFrame to a text file using a space as a delimiter
    good_df.to_csv(save_dir + 'main_sample.csv', index=False)
    line_notfullcover_df.to_csv(save_dir + 'line_notfullcover_df.csv', index=False)
    filt_edge_df.to_csv(save_dir + 'filt_edge.csv', index=False)
    ha_snr_flag_df.to_csv(save_dir + 'ha_snr_flag.csv', index=False)
    pab_snr_flag_df.to_csv(save_dir + 'pab_snr_flag.csv', index=False)
    id_msa_skipped_df.to_csv(save_dir + 'id_msa_skipped.csv', index=False)
    cont_overlap_df.to_csv(save_dir + 'cont_overlap_line.csv', index=False)
    return

def find_good_spec():
    """ Reads in spectra catalog and makes sure quality is good"""
    zqual_df = read_spec_cat()
    zqual_df = zqual_df[zqual_df['flag_zspec_qual'] == 3]
    zqual_df = zqual_df[zqual_df['flag_spec_qual'] == 0]
    return zqual_df

def select_spectra(zqual_df, line_list, paa_only=False):
    """Checking that both target lines are covered in the photometry"""
    uncover_filt_dir, filters = unconver_read_filters()
    supercat_df = read_supercat()
    filt_cols = get_filt_cols(supercat_df, skip_wide_bands=True)
    covered_idxs = []
    line0_filts = []
    line1_filts = []
    line2_filts = []
    if paa_only:
         for i in range(len(zqual_df)):
            redshift = zqual_df['z_spec'].iloc[i]
            line2_cover, line2_filt_name = line_in_range(redshift, line_list[1][1], filt_cols, uncover_filt_dir)
            if line2_cover:
                covered_idxs.append(i)
                line0_filts.append('-99')
                line1_filts.append('-99')
                line2_filts.append(line2_filt_name)
    else:
        for i in range(len(zqual_df)):
            redshift = zqual_df['z_spec'].iloc[i]
            line0_cover, line0_filt_name = line_in_range(redshift, line_list[0][1], filt_cols, uncover_filt_dir)
            line1_cover, line1_filt_name = line_in_range(redshift, line_list[1][1], filt_cols, uncover_filt_dir)
            # line2_cover, line2_filt_name = line_in_range(redshift, line_list[2][1], filt_cols, uncover_filt_dir)

            both_corered = (line0_cover and line1_cover) #or (line0_cover and line2_cover)
            if both_corered == True:
                covered_idxs.append(i)
                line0_filts.append(line0_filt_name)
                line1_filts.append(line1_filt_name)
                # line2_filts.append(line2_filt_name)
    zqual_df_covered = zqual_df.iloc[covered_idxs]
    zqual_df_covered = zqual_df_covered.reset_index()
    zqual_df_covered['line0_filt'] = line0_filts
    zqual_df_covered['line1_filt'] = line1_filts
    if paa_only:
        zqual_df_covered['line2_filt'] = line2_filts
    return zqual_df_covered

def line_in_range(z, target_line, filt_cols, uncover_filt_dir):
    z_line = target_line * (1+z)
    covered = False
    filt_name = ''
    for filt in filt_cols:
        if z_line>uncover_filt_dir[filt+'_blue'] and z_line<uncover_filt_dir[filt+'_red']:
            covered = True
            filt_name = filt
    return covered, filt_name


def paper_figure_sample_selection(id_msa_list, color_var='None', plot_sfr_mass=False, plot_mags=False, plot_mass_mags=False):
    show_squares = True
    show_low_snr = True
    show_sample = True
    show_hexes = True


    fig = plt.figure(figsize=(8,6))
    ax = fig.add_axes([0.15,0.15,0.525,0.7])
    cb_ax = fig.add_axes([0.725,0.15,0.04,0.7])

    fontsize = 14
    normal_markersize=8
    small_markersize = 4
    gray_markersize = 4
    background_markersize = 2

    zqual_df = read_spec_cat()
    sps_df = read_SPS_cat()
    sps_all_df = read_SPS_cat_all()
    supercat_df = read_supercat()
    lineratio_data_df = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/lineratio_av_df_all.csv').to_pandas()


    line_notfullcover_df = ascii.read('/Users/brianlorenz/uncover/Data/sample_selection/line_notfullcover_df.csv').to_pandas()
    filt_edge_df = ascii.read('/Users/brianlorenz/uncover/Data/sample_selection/filt_edge.csv').to_pandas()
    ha_snr_flag_df = ascii.read('/Users/brianlorenz/uncover/Data/sample_selection/ha_snr_flag.csv').to_pandas()
    pab_snr_flag_df = ascii.read('/Users/brianlorenz/uncover/Data/sample_selection/pab_snr_flag.csv').to_pandas()
    id_msa_skipped_df = ascii.read('/Users/brianlorenz/uncover/Data/sample_selection/id_msa_skipped.csv').to_pandas()
    id_msa_cont_overlap_line_df = ascii.read('/Users/brianlorenz/uncover/Data/sample_selection/cont_overlap_line.csv').to_pandas()
    id_msa_not_in_filt_df = ascii.read('/Users/brianlorenz/uncover/Data/sample_selection/line_not_in_filt.csv').to_pandas()

    id_redshift_issue_list = filt_edge_df['id_msa'].append(id_msa_not_in_filt_df['id_msa']).append(id_msa_cont_overlap_line_df['id_msa']).append(line_notfullcover_df['id_msa']).to_list()
    id_pab_snr_list = pab_snr_flag_df['id_msa'].to_list()
    id_skip_list = id_msa_skipped_df['id_msa'].to_list()

    id_msa_list = id_msa_list + id_pab_snr_list

    # Gray background
    all_masses = sps_all_df['mstar_50']
    all_sfr100s = sps_all_df['sfr100_50']
    all_log_sfr100s = np.log10(all_sfr100s)
    all_redshifts = sps_all_df['z_50']
    # ax.plot(all_redshifts, all_masses, marker='o', ls='None', markersize=background_markersize, color='gray')
    cmap = plt.get_cmap('gray_r')
    new_cmap = truncate_colormap(cmap, 0, 0.7)
    good_redshift_idx = np.logical_and(all_redshifts > 1.3, all_redshifts < 2.5)
    good_mass_idx = np.logical_and(all_masses > 5, all_masses < 11)
    good_sfr_idx = np.logical_and(all_log_sfr100s > -2.5, all_log_sfr100s < 2)

    
    
    if show_hexes:
        if plot_sfr_mass == False and plot_mags == False and plot_mass_mags == False:
            hexbin_norm = mpl.colors.Normalize(vmin=1, vmax=200) 
            good_both_idx = np.logical_and(good_redshift_idx, good_mass_idx)    
            ax.hexbin(all_redshifts[good_both_idx], all_masses[good_both_idx], gridsize=15, cmap=new_cmap, norm=hexbin_norm, label='Photometric Sample')
        elif plot_mags:
            hexbin_norm = mpl.colors.Normalize(vmin=1, vmax=200) 
            f444w_fluxes = supercat_df['f_f444w'] * 1e-8
            f444w_mags = -2.5 * np.log10(f444w_fluxes) + 8.9
            good_mag_idx = np.logical_and(f444w_mags>18, f444w_mags<35)
            good_both_idx = np.logical_and(good_redshift_idx, good_mag_idx)
            ax.hexbin(all_redshifts[good_both_idx], f444w_mags[good_both_idx], gridsize=15, cmap=new_cmap, norm=hexbin_norm, label='Photometric Sample')
        elif plot_mass_mags:
            hexbin_norm = mpl.colors.Normalize(vmin=1, vmax=200) 
            f444w_fluxes = supercat_df['f_f444w'] * 1e-8
            f444w_mags = -2.5 * np.log10(f444w_fluxes) + 8.9
            good_mag_idx = np.logical_and(f444w_mags>18, f444w_mags<35)
            good_both_idx = np.logical_and(good_mass_idx, good_mag_idx)
            ax.hexbin(all_masses[good_both_idx], f444w_mags[good_both_idx], gridsize=15, cmap=new_cmap, norm=hexbin_norm, label='Photometric Sample')
        else:
            # hexbin_norm = mpl.colors.LogNorm(vmin=1, vmax=5000) 
            hexbin_norm = mpl.colors.Normalize(vmin=1, vmax=500) 
            good_both_idx = np.logical_and(good_sfr_idx, good_mass_idx)
            ax.hexbin(all_masses[good_both_idx], all_log_sfr100s[good_both_idx], gridsize=15, cmap=new_cmap, norm=hexbin_norm, label='Photometric Sample')

    # Gray high quality points
    for id_msa in zqual_df['id_msa']:
        if id_msa in id_skip_list or id_msa == 42041:
            continue
        if id_msa in id_msa_list:
            continue

        marker = 'o'
        zqual_row = zqual_df[zqual_df['id_msa'] == id_msa]

        if zqual_row['flag_zspec_qual'].iloc[0] != 3 or zqual_row['flag_spec_qual'].iloc[0] != 0:
            marker = 's'
            continue

        if id_msa in id_redshift_issue_list:
            marker = 's'
            if show_squares == False:
                continue
        elif id_msa in id_pab_snr_list or id_msa==32575:
            marker = 'o'
            if show_low_snr == False:
                continue
            
        redshift = zqual_row['z_spec'].iloc[0]
        if redshift < 1.3 or redshift > 2.4:
            continue
        # id_dr2 = zqual_df[zqual_df['id_msa']==id_msa]['id_DR2'].iloc[0]
        sps_row = sps_df[sps_df['id_msa']==id_msa]
        stellar_mass_50 = sps_row['mstar_50']
        sfr100_50 = sps_row['sfr100_50']
        log_sfr100_50 = np.log10(sps_row['sfr100_50'])
        if len(sps_row) == 0:
            print(f'No SPS for {id_msa}')
            continue
        
        if show_sample == False:
            if id_msa in id_msa_list and id_msa not in id_pab_snr_list:
                continue


        if plot_mags or plot_mass_mags:
            detected_apparent_mag_hafilt, err_detected_apparent_mag_hafilt_u, err_detected_apparent_mag_hafilt_d = get_mags_info(id_msa, detected='F444W')
        if plot_sfr_mass == False and plot_mags==False and plot_mass_mags==False:
            ax.plot(redshift, stellar_mass_50, marker=marker, color='gray', ls='None', ms=gray_markersize, mec='black')
            # ax.text(redshift, stellar_mass_50, f'{id_msa}')
        elif plot_mags:
            ax.plot(redshift, detected_apparent_mag_hafilt, marker=marker, color='gray', ls='None', ms=gray_markersize, mec='black')
        elif plot_mass_mags:
            ax.plot(stellar_mass_50, detected_apparent_mag_hafilt, marker=marker, color='gray', ls='None', ms=gray_markersize, mec='black')
        else:
            ax.plot(stellar_mass_50, log_sfr100_50, marker=marker, color='gray', ls='None', ms=gray_markersize, mec='black')


    # Selected Sample
    for id_msa in id_msa_list:
        

        zqual_row = zqual_df[zqual_df['id_msa'] == id_msa]
        # supercat_row = supercat_df[supercat_df['id_msa']==id_msa]
        redshift = zqual_row['z_spec'].iloc[0]
        # id_dr2 = zqual_df[zqual_df['id_msa']==id_msa]['id_DR2'].iloc[0]
        sps_row = sps_df[sps_df['id_msa']==id_msa]

        stellar_mass_50 = sps_row['mstar_50']
        err_stellar_mass_low = stellar_mass_50 - sps_row['mstar_16']
        err_stellar_mass_high = sps_row['mstar_84'] - stellar_mass_50

        sfr100_50 = sps_row['sfr100_50']
        log_sfr100_50 = np.log10(sps_row['sfr100_50'])
        err_sfr100_50_low = log_sfr100_50 - np.log10(sps_row['sfr100_16'])
        err_sfr100_50_high = np.log10(sps_row['sfr100_84']) - log_sfr100_50

        dust2_50 = sps_row['dust2_50'].iloc[0]

        # data_df_row = data_df[data_df['id_msa'] == id_msa]
        lineratio_data_row = lineratio_data_df[lineratio_data_df['id_msa'] == id_msa]
        fit_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting/{id_msa}_emission_fits.csv').to_pandas()
        # ha_snr = fit_df['signal_noise_ratio'].iloc[0]
        pab_snr = fit_df['signal_noise_ratio'].iloc[0]
        ha_eqw = fit_df['equivalent_width_aa'].iloc[0]
        pab_eqw = fit_df['equivalent_width_aa'].iloc[1]
        
        cmap = mpl.cm.viridis
        # cmap = truncate_colormap(cmap, 0.15, 1.0)

        if plot_mags or plot_mass_mags:
            detected_apparent_mag_hafilt, err_detected_apparent_mag_hafilt_u, err_detected_apparent_mag_hafilt_d = get_mags_info(id_msa, detected='F444W')
            
        
        if color_var == 'sed_av':
            norm = mpl.colors.Normalize(vmin=0, vmax=3) 
            rgba = cmap(norm(lineratio_data_row['sed_av']))  
            if len(lineratio_data_row['sed_av']) == 0:
                rgba = 'white'
            cbar_label = 'Photometry AV'
        if color_var == 'redshift':
            norm = mpl.colors.LogNorm(vmin=1.2, vmax=2.5) 
            rgba = cmap(norm(redshift))
            cbar_label = 'Redshift'
        if color_var == 'dust2':
            norm = mpl.colors.Normalize(vmin=0, vmax=2) 
            rgba = cmap(norm(dust2_50))
            cbar_label = 'Prospector dust2_50'
        if color_var == 'sfr':
            norm = mpl.colors.LogNorm(vmin=0.1, vmax=10) 
            rgba = cmap(norm(sfr100_50))
            cbar_label = 'Prospector SFR (M$_\odot$ / yr)'
        if color_var == 'ha_eqw':
            norm = mpl.colors.LogNorm(vmin=10, vmax=1000) 
            rgba = cmap(norm(np.abs(ha_eqw)))
            cbar_label = 'H$\\alpha$ Equivalent Width'
        if color_var != 'None':
            color_str = f'_{color_var}'
        else:
            rgba = 'black'
            color_str = ''

        if id_msa in id_pab_snr_list:
            markersize = small_markersize
            if show_low_snr == False:
                continue
        else:
            markersize = normal_markersize
            if show_sample == False:
                continue
        
        if id_msa in id_msa_list:
            print(f'{id_msa}, mass {stellar_mass_50.iloc[0]}, ha_eqw {ha_eqw}, pab_eqw {pab_eqw}')

            # SFMS location:
            measured_log_sfr = np.log10(sfr100_50.iloc[0])
            sfms_log_sfr = whitaker_sfms(stellar_mass_50.iloc[0])
            offset_below_sfms = sfms_log_sfr - measured_log_sfr # positive means below sfms, negative means above
            # print(f'id_msa: {id_msa}, sfms_offset = {offset_below_sfms}')
        

        if plot_sfr_mass == False and plot_mags == False and plot_mass_mags==False:
            ax.errorbar(redshift, stellar_mass_50, yerr=[err_stellar_mass_low, err_stellar_mass_high], marker='o', color=rgba, ls='None', mec='black', ms=markersize)
            # ax.text(redshift, stellar_mass_50.iloc[0], f'{id_msa}')
        elif plot_mags:
            ax.errorbar(redshift, detected_apparent_mag_hafilt, yerr=np.array([[err_detected_apparent_mag_hafilt_d, err_detected_apparent_mag_hafilt_u]]).T, marker='o', color=rgba, ls='None', mec='black', ms=markersize)
        elif plot_mass_mags:
            ax.errorbar(stellar_mass_50, detected_apparent_mag_hafilt, xerr=[err_stellar_mass_low, err_stellar_mass_high], yerr=np.array([[err_detected_apparent_mag_hafilt_d, err_detected_apparent_mag_hafilt_u]]).T, marker='o', color=rgba, ls='None', mec='black', ms=markersize)
        else:
            ax.errorbar(stellar_mass_50, log_sfr100_50, xerr=[err_stellar_mass_low, err_stellar_mass_high], yerr=[err_sfr100_50_low, err_sfr100_50_high], marker='o', color=rgba, ls='None', mec='black', ms=markersize)
    
    if plot_sfr_mass == False and plot_mags == False and plot_mass_mags==False:
        ax.set_ylabel('Prospector '+stellar_mass_label, fontsize=fontsize)
        ax.set_xlabel('Redshift', fontsize=fontsize) 
        ax.set_xlim([1.3, 2.5])
        ax.set_ylim([6, 11])
        scale_aspect(ax)
    elif plot_mags:
        ax.set_ylabel('F444W Apparent Magnitude', fontsize=fontsize)
        ax.set_xlabel('Redshift', fontsize=fontsize) 
        ax.set_xlim([1.3, 2.5])
        ax.set_ylim([18, 35])
        ax.invert_yaxis()
        scale_aspect(ax)
    elif plot_mass_mags:
        ax.set_ylabel('F444W Apparent Magnitude', fontsize=fontsize)
        ax.set_xlabel(stellar_mass_label, fontsize=fontsize) 
        ax.set_xlim([6, 11])
        ax.set_ylim([18, 35])
        ax.invert_yaxis()
        scale_aspect(ax)
    else:
        ax.set_xlabel('Prospector '+ stellar_mass_label, fontsize=fontsize)
        ax.set_ylabel('Prospector '+ sfr_label, fontsize=fontsize) 
        ax.set_xlim([6, 11])
        ax.set_ylim([-2.5, 2])
        # ax.set_yscale('log')
    
    ax.tick_params(labelsize=fontsize)
    

    if show_sample == False and show_low_snr == False:
        norm = mpl.colors.LogNorm(vmin=0.1, vmax=50) 
        cbar_label = 'Prospector SFR'
        color_str = '_sfr'
    if color_var != 'None':
        if color_var == 'sfr':
            cbar_ticks = [0.1, 1, 10]
            cbar_ticklabels = [str(tick) for tick in cbar_ticks]
        if color_var == 'ha_eqw':
            cbar_ticks = [10, 100, 1000]
            cbar_ticklabels = [str(tick) for tick in cbar_ticks]
        if color_var == 'redshift':
            cbar_ticks = [1.5, 2, 2.5]
            cbar_ticklabels = [str(tick) for tick in cbar_ticks]
        sm =  mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, orientation='vertical', cax=cb_ax, ticks=cbar_ticks)
        cbar.ax.set_yticklabels(cbar_ticklabels)  
        cbar.set_label(cbar_label, fontsize=16)
        cbar.ax.tick_params(labelsize=16)

    # SFMS
    if plot_sfr_mass == True:
        masses = np.arange(6,11,0.1)
        # predicted_log_sfrs = check_sfms(masses, 0.5)
        predicted_log_sfrs = whitaker_sfms(masses)
        # predicted_log_sfrs = speagle_sfms(masses, )
        ax.plot(masses, predicted_log_sfrs, color='red', ls='--', marker='None', label='SFMS, z=2')
        smfs_offsets = [0.1020446252898145,0.18283711116867318, -0.2375098376696847, 0.7051415256536799,0.6706993379653309, -0.18537210751204647, 0.22890526287538182, -0.2798992695970548, -0.773697743192055,0.3767266122566961,0.6676142678869769,-0.3323495130454126,0.308414353592555,-0.876565275930646]   
        # breakpoint()
    # Legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import RegularPolygon

    line_sample = Line2D([0], [0], color=cmap(norm(3)), marker='o', markersize=8, ls='None', mec='black')
    line_snr = Line2D([0], [0], color=cmap(norm(3)), marker='o', markersize=4, ls='None', mec='black')
    line_squares = Line2D([0], [0], color='grey', marker='s', markersize=4, ls='None', mec='black')
    line_hexes = Line2D([0], [0], color='grey', marker='h', markersize=12, ls='None')
    custom_lines = [line_sample, line_snr, line_squares, line_hexes]
    custom_labels = ['Selected Sample', 'Pa$\\beta$ SNR < 5', 'Line not in Filter', 'Photometric Sample']
    if show_hexes == False:
        custom_lines = [line for line in custom_lines if line != line_hexes]
        custom_labels = [lab for lab in custom_labels if lab != 'Photometric Sample']
    if show_sample == False:
        custom_lines = [line for line in custom_lines if line != line_sample]
        custom_labels = [lab for lab in custom_labels if lab != 'Selected Sample']
    if show_low_snr == False:
        custom_lines = [line for line in custom_lines if line != line_snr]
        custom_labels = [lab for lab in custom_labels if lab != 'Pa$\\beta$ SNR < 5']
    if show_squares == False:
        custom_lines = [line for line in custom_lines if line != line_squares]
        custom_labels = [lab for lab in custom_labels if lab != 'Line not in Filter']
    if plot_mass_mags:
        legend_loc = 4
    else:
        legend_loc = 3
    ax.legend(custom_lines, custom_labels, loc=legend_loc, fontsize=fontsize-2)

    add_str=''
    if plot_mass_mags:
        add_str='mass_mag'

    if plot_sfr_mass == False:
        save_loc = f'/Users/brianlorenz/uncover/Figures/paper_figures/sample_selection{color_str}.pdf'
    else:
        save_loc = f'/Users/brianlorenz/uncover/Figures/paper_figures/sample_selection_sfrmass_{color_str}{add_str}.pdf'

    fig.savefig(save_loc, bbox_inches='tight')

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
        return new_cmap

def get_mags_info(id_msa, detected='True'):
    if detected == 'True':
        ha_filters, ha_images, wht_ha_images, obj_segmap, ha_photfnus, ha_all_filts = make_3color(id_msa, line_index=0, plot=False)
    elif detected == 'F444W':
        ha_filters = ['_', 'f444w']
    else:
        ha_filters = ['_', 'f150w']
    sed_df = read_sed(id_msa)
    detected_hafilt_flux_jy = sed_df[sed_df['filter']==f'f_{ha_filters[1]}']['flux'].iloc[0]
    err_flux_jy = sed_df[sed_df['filter']==f'f_{ha_filters[1]}']['err_flux'].iloc[0]
    detected_apparent_mag_hafilt = -2.5 * np.log10(detected_hafilt_flux_jy) + 8.9
    detected_apparent_mag_hafilt_u = -2.5 * np.log10(detected_hafilt_flux_jy-err_flux_jy) + 8.9
    detected_apparent_mag_hafilt_d = -2.5 * np.log10(detected_hafilt_flux_jy+err_flux_jy) + 8.9
    err_detected_apparent_mag_hafilt_u = detected_apparent_mag_hafilt_u - detected_apparent_mag_hafilt
    err_detected_apparent_mag_hafilt_d = detected_apparent_mag_hafilt - detected_apparent_mag_hafilt_d
    return detected_apparent_mag_hafilt, err_detected_apparent_mag_hafilt_u, err_detected_apparent_mag_hafilt_d


def check_sfms(log_mass, redshift):
    a_coeff = 0.7-0.13*redshift
    b_coeff = 0.38+1.14*redshift-0.19*redshift**2
    log_sfr = a_coeff*(log_mass-10.5) + b_coeff
    return log_sfr


def whitaker_sfms(mass):
    # a = -24.0415
    # b = 4.1693
    # c = -0.1638

    a = -19.99
    b = 3.44
    c = -0.13
    sfms = a + b*mass + c*mass**2
    return sfms

def speagle_sfms(mass, redshift):
    from astropy.cosmology import WMAP9 as cosmo
    age_of_universe = cosmo.age(redshift).value

    sfms = (0.84-0.026*age_of_universe) * mass - (6.51-0.11*age_of_universe)
    return sfms

def pop_sfms(mass, redshift): # https://academic.oup.com/mnras/article/519/1/1526/6815739#equ10
    from astropy.cosmology import WMAP9 as cosmo
    age_of_universe = cosmo.age(redshift).value

    a0 = 0.20
    a1 = -0.034
    b0 = -26.134
    b1 = 4.722
    b2 = -0.1925

    sfms = (a1*age_of_universe+b1) * mass + b2*mass**2 + (b0+a0*age_of_universe)
    return sfms

def paper_figure_sample_selection_twopanel(id_msa_list):
    show_squares = True
    show_low_snr = True
    show_sample = True
    show_hexes = True


    fig = plt.figure(figsize=(12,6))
    height = 0.72
    width = height/2
    start_level = 0.12
    ax_mass = fig.add_axes([0.10, start_level, width, height])
    ax_mag = fig.add_axes([0.55, start_level, width, height])
    cb_ax = fig.add_axes([0.935, start_level,0.03,height])


    fontsize = 14
    normal_markersize=8
    small_markersize = 4
    gray_markersize = 4
    background_markersize = 2

    zqual_df = read_spec_cat()
    sps_df = read_SPS_cat()
    sps_all_df = read_SPS_cat_all()
    supercat_df = read_supercat()
    lineratio_data_df = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/lineratio_av_df_all.csv').to_pandas()


    line_notfullcover_df = ascii.read('/Users/brianlorenz/uncover/Data/sample_selection/line_notfullcover_df.csv').to_pandas()
    filt_edge_df = ascii.read('/Users/brianlorenz/uncover/Data/sample_selection/filt_edge.csv').to_pandas()
    ha_snr_flag_df = ascii.read('/Users/brianlorenz/uncover/Data/sample_selection/ha_snr_flag.csv').to_pandas()
    pab_snr_flag_df = ascii.read('/Users/brianlorenz/uncover/Data/sample_selection/pab_snr_flag.csv').to_pandas()
    id_msa_skipped_df = ascii.read('/Users/brianlorenz/uncover/Data/sample_selection/id_msa_skipped.csv').to_pandas()
    id_msa_cont_overlap_line_df = ascii.read('/Users/brianlorenz/uncover/Data/sample_selection/cont_overlap_line.csv').to_pandas()
    id_msa_not_in_filt_df = ascii.read('/Users/brianlorenz/uncover/Data/sample_selection/line_not_in_filt.csv').to_pandas()

    id_redshift_issue_list = filt_edge_df['id_msa'].append(id_msa_not_in_filt_df['id_msa']).append(id_msa_cont_overlap_line_df['id_msa']).append(line_notfullcover_df['id_msa']).to_list()
    id_pab_snr_list = pab_snr_flag_df['id_msa'].to_list()
    id_skip_list = id_msa_skipped_df['id_msa'].to_list()

    id_msa_list = id_msa_list + id_pab_snr_list

    # Gray background
    all_masses = sps_all_df['mstar_50']
    all_sfr100s = sps_all_df['sfr100_50']
    all_log_sfr100s = np.log10(all_sfr100s)
    all_redshifts = sps_all_df['z_50']
    # ax.plot(all_redshifts, all_masses, marker='o', ls='None', markersize=background_markersize, color='gray')
    cmap = plt.get_cmap('gray_r')
    new_cmap = truncate_colormap(cmap, 0, 0.7)
    good_redshift_idx = np.logical_and(all_redshifts > 1.3, all_redshifts < 2.5)
    good_mass_idx = np.logical_and(all_masses > 5, all_masses < 11)
    good_sfr_idx = np.logical_and(all_log_sfr100s > -2.5, all_log_sfr100s < 2)

    # Plot SFMS
    masses = np.arange(6,11,0.1)
    # predicted_log_sfrs = check_sfms(masses, 0.5)
    # predicted_log_sfrs = whitaker_sfms(masses)
    # predicted_log_sfrs = speagle_sfms(masses, 1.96)
    predicted_log_sfrs = pop_sfms(masses, 1.96)
    ax_mass.plot(masses, predicted_log_sfrs, color='red', ls='--', marker='None')
    
    if show_hexes:
        # hexbin_norm = mpl.colors.Normalize(vmin=1, vmax=200) 
        # good_both_idx = np.logical_and(good_redshift_idx, good_mass_idx)    
        # ax_mass.hexbin(all_redshifts[good_both_idx], all_masses[good_both_idx], gridsize=15, cmap=new_cmap, norm=hexbin_norm, label='Photometric Sample')
    
        hexbin_norm = mpl.colors.Normalize(vmin=1, vmax=500) 
        good_both_idx = np.logical_and(good_sfr_idx, good_mass_idx)
        ax_mass.hexbin(all_masses[good_both_idx], all_log_sfr100s[good_both_idx], gridsize=15, cmap=new_cmap, norm=hexbin_norm, label='Photometric Sample')

        mag_lower_lim = 32
        hexbin_norm = mpl.colors.Normalize(vmin=1, vmax=200) 
        f444w_fluxes = supercat_df['f_f444w'] * 1e-8
        f444w_mags = -2.5 * np.log10(f444w_fluxes) + 8.9
        good_mag_idx = np.logical_and(f444w_mags>18, f444w_mags<mag_lower_lim)
        good_both_idx = np.logical_and(good_redshift_idx, good_mag_idx)
        ax_mag.hexbin(all_redshifts[good_both_idx], f444w_mags[good_both_idx], gridsize=15, cmap=new_cmap, norm=hexbin_norm, label='Photometric Sample')
        
    # Gray high quality points
    for id_msa in zqual_df['id_msa']:
        if id_msa in id_skip_list or id_msa == 42041:
            continue
        if id_msa in id_msa_list:
            continue

        marker = 'o'
        zqual_row = zqual_df[zqual_df['id_msa'] == id_msa]

        if zqual_row['flag_zspec_qual'].iloc[0] != 3 or zqual_row['flag_spec_qual'].iloc[0] != 0:
            marker = 's'
            continue

        if id_msa in id_redshift_issue_list:
            marker = 's'
            if show_squares == False:
                continue
        elif id_msa in id_pab_snr_list or id_msa==32575:
            marker = 'o'
            if show_low_snr == False:
                continue
            
        redshift = zqual_row['z_spec'].iloc[0]
        if redshift < 1.3 or redshift > 2.4:
            continue
        # id_dr2 = zqual_df[zqual_df['id_msa']==id_msa]['id_DR2'].iloc[0]
        sps_row = sps_df[sps_df['id_msa']==id_msa]
        stellar_mass_50 = sps_row['mstar_50']
        sfr100_50 = sps_row['sfr100_50']
        log_sfr100_50 = np.log10(sps_row['sfr100_50'])
        if len(sps_row) == 0:
            print(f'No SPS for {id_msa}')
            continue
        
        if show_sample == False:
            if id_msa in id_msa_list and id_msa not in id_pab_snr_list:
                continue


        detected_apparent_mag_hafilt, err_detected_apparent_mag_hafilt_u, err_detected_apparent_mag_hafilt_d = get_mags_info(id_msa, detected='F444W')
        
        ax_mass.plot(stellar_mass_50, log_sfr100_50, marker=marker, color='gray', ls='None', ms=gray_markersize, mec='black')
        # ax_mass.plot(redshift, stellar_mass_50, marker=marker, color='gray', ls='None', ms=gray_markersize, mec='black')
            # ax.text(redshift, stellar_mass_50, f'{id_msa}')
        ax_mag.plot(redshift, detected_apparent_mag_hafilt, marker=marker, color='gray', ls='None', ms=gray_markersize, mec='black')
        
    # Selected Sample
    for id_msa in id_msa_list:
        zqual_row = zqual_df[zqual_df['id_msa'] == id_msa]
        # supercat_row = supercat_df[supercat_df['id_msa']==id_msa]
        redshift = zqual_row['z_spec'].iloc[0]
        id_msa_sample = get_id_msa_list(full_sample=False)
        if id_msa in id_msa_sample:
            print(redshift)
        # id_dr2 = zqual_df[zqual_df['id_msa']==id_msa]['id_DR2'].iloc[0]
        sps_row = sps_df[sps_df['id_msa']==id_msa]

        stellar_mass_50 = sps_row['mstar_50']
        err_stellar_mass_low = stellar_mass_50 - sps_row['mstar_16']
        err_stellar_mass_high = sps_row['mstar_84'] - stellar_mass_50

        sfr100_50 = sps_row['sfr100_50']
        log_sfr100_50 = np.log10(sps_row['sfr100_50'])
        err_sfr100_50_low = log_sfr100_50 - np.log10(sps_row['sfr100_16'])
        err_sfr100_50_high = np.log10(sps_row['sfr100_84']) - log_sfr100_50

        dust2_50 = sps_row['dust2_50'].iloc[0]

        # data_df_row = data_df[data_df['id_msa'] == id_msa]
        lineratio_data_row = lineratio_data_df[lineratio_data_df['id_msa'] == id_msa]
        fit_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting/{id_msa}_emission_fits.csv').to_pandas()
        # ha_snr = fit_df['signal_noise_ratio'].iloc[0]
        pab_snr = fit_df['signal_noise_ratio'].iloc[0]
        ha_eqw = fit_df['equivalent_width_aa'].iloc[0]
        pab_eqw = fit_df['equivalent_width_aa'].iloc[1]
        
        cmap = mpl.cm.viridis
        # cmap = truncate_colormap(cmap, 0.15, 1.0)

        detected_apparent_mag_hafilt, err_detected_apparent_mag_hafilt_u, err_detected_apparent_mag_hafilt_d = get_mags_info(id_msa, detected='F444W')
            
        
        norm_mass = mpl.colors.LogNorm(vmin=0.1, vmax=10) 
        rgba_mass = cmap(norm_mass(sfr100_50))
        cbar_label_mass = 'Prospector SFR (M$_\odot$ / yr)'

        norm = mpl.colors.LogNorm(vmin=10, vmax=1000) 
        rgba_mag = cmap(norm(np.abs(ha_eqw)))
        cbar_label_mag = 'H$\\alpha$ Equivalent Width'
        
        if id_msa in id_pab_snr_list:
            markersize = small_markersize
            if show_low_snr == False:
                continue
        else:
            markersize = normal_markersize
            if show_sample == False:
                continue
        
        if id_msa in id_msa_list:
            # print(f'{id_msa}, mass {stellar_mass_50.iloc[0]}, ha_eqw {ha_eqw}, pab_eqw {pab_eqw}')

            # SFMS location:
            measured_log_sfr = np.log10(sfr100_50.iloc[0])
            sfms_log_sfr = whitaker_sfms(stellar_mass_50.iloc[0])
            offset_below_sfms = sfms_log_sfr - measured_log_sfr # positive means below sfms, negative means above
            # print(f'id_msa: {id_msa}, sfms_offset = {offset_below_sfms}')
        
        ax_mass.errorbar(stellar_mass_50, log_sfr100_50, xerr=[err_stellar_mass_low, err_stellar_mass_high], yerr=[err_sfr100_50_low, err_sfr100_50_high], marker='o', color=rgba_mag, ls='None', mec='black', ms=markersize)
        # ax_mass.errorbar(redshift, stellar_mass_50, yerr=[err_stellar_mass_low, err_stellar_mass_high], marker='o', color=rgba_mass, ls='None', mec='black', ms=markersize)
            # ax.text(redshift, stellar_mass_50.iloc[0], f'{id_msa}')
        ax_mag.errorbar(redshift, detected_apparent_mag_hafilt, yerr=np.array([[err_detected_apparent_mag_hafilt_d, err_detected_apparent_mag_hafilt_u]]).T, marker='o', color=rgba_mag, ls='None', mec='black', ms=markersize)
        
    # ax_mass.set_ylabel('Prospector '+stellar_mass_label, fontsize=fontsize)
    # ax_mass.set_xlabel('Redshift', fontsize=fontsize) 
    # ax_mass.set_xlim([1.3, 2.5])
    # ax_mass.set_ylim([5, 11])
    # scale_aspect(ax_mass)
    # ax_mass.tick_params(labelsize=fontsize)
    ax_mass.set_xlabel('Prospector '+ stellar_mass_label, fontsize=fontsize)
    ax_mass.set_ylabel('Prospector '+ sfr_label, fontsize=fontsize) 
    ax_mass.set_xlim([6, 11])
    ax_mass.set_ylim([-2.5, 2])
    scale_aspect(ax_mass)
    ax_mass.tick_params(labelsize=fontsize)


    ax_mag.set_ylabel('F444W Apparent Magnitude', fontsize=fontsize)
    ax_mag.set_xlabel('Redshift', fontsize=fontsize) 
    ax_mag.set_xlim([1.3, 2.5])
    ax_mag.set_ylim([18, mag_lower_lim])
    ax_mag.invert_yaxis()
    scale_aspect(ax_mag)
    ax_mag.tick_params(labelsize=fontsize)
    

   
    cbar_sfr_ticks = [0.1, 1, 10]
    cbar_sfr_ticklabels = [str(tick) for tick in cbar_sfr_ticks]

    cbar_ew_ticks = [10, 100, 1000]
    cbar_ew_ticklabels = [str(tick) for tick in cbar_ew_ticks]
        
    sm =  mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, orientation='vertical', cax=cb_ax, ticks=cbar_ew_ticks)
    cbar.ax.set_yticklabels(cbar_ew_ticklabels)  
    cbar.set_label(cbar_label_mag, fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    # # SFMS
    # if plot_sfr_mass == True:
    #     masses = np.arange(6,11,0.1)
    #     # predicted_log_sfrs = check_sfms(masses, 0.5)
    #     predicted_log_sfrs = whitaker_sfms(masses)
    #     ax.plot(masses, predicted_log_sfrs, color='red', ls='--', marker='None', label='SFMS, z=2')
    #     smfs_offsets = [0.1020446252898145,0.18283711116867318, -0.2375098376696847, 0.7051415256536799,0.6706993379653309, -0.18537210751204647, 0.22890526287538182, -0.2798992695970548, -0.773697743192055,0.3767266122566961,0.6676142678869769,-0.3323495130454126,0.308414353592555,-0.876565275930646]   
    #     # breakpoint()
    # Legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import RegularPolygon

    line_sample = Line2D([0], [0], color=cmap(norm_mass(3)), marker='o', markersize=8, ls='None', mec='black')
    line_snr = Line2D([0], [0], color=cmap(norm_mass(3)), marker='o', markersize=4, ls='None', mec='black')
    line_squares = Line2D([0], [0], color='grey', marker='s', markersize=4, ls='None', mec='black')
    line_hexes = Line2D([0], [0], color='grey', marker='h', markersize=12, ls='None')
    line_sfms = Line2D([0], [0], color='red', marker='None', ls='--')
    custom_lines = [line_sample, line_snr, line_squares, line_hexes, line_sfms]
    custom_labels = ['Selected Sample', 'Pa$\\beta$ SNR < 5', 'Line not in Filter', 'Photometric Sample', 'Popesso+22 SFMS']
    if show_hexes == False:
        custom_lines = [line for line in custom_lines if line != line_hexes]
        custom_labels = [lab for lab in custom_labels if lab != 'Photometric Sample']
    if show_sample == False:
        custom_lines = [line for line in custom_lines if line != line_sample]
        custom_labels = [lab for lab in custom_labels if lab != 'Selected Sample']
    if show_low_snr == False:
        custom_lines = [line for line in custom_lines if line != line_snr]
        custom_labels = [lab for lab in custom_labels if lab != 'Pa$\\beta$ SNR < 5']
    if show_squares == False:
        custom_lines = [line for line in custom_lines if line != line_squares]
        custom_labels = [lab for lab in custom_labels if lab != 'Line not in Filter']
    ax_mass.legend(custom_lines, custom_labels, loc=2, fontsize=fontsize-2)
    # ax_mass.legend(custom_lines, custom_labels, loc=3, fontsize=fontsize-2)


    save_loc = f'/Users/brianlorenz/uncover/Figures/paper_figures/sample_selection_twopanel.pdf'
    
    fig.savefig(save_loc, bbox_inches='tight')

if __name__ == "__main__":
    # z_list = [2.2107,
    #             1.3688,
    #             2.3257,
    #             1.8854,
    #             1.4121,
    #             2.33,
    #             1.5602,
    #             2.2972,
    #             1.8584,
    #             2.1854,
    #             2.1857,
    #             1.8622,
    #             1.8659,
    #             2.1217]
    # breakpoint()
    # sample_select()
    # sample_select(paalpha=True)
    sample_select(paalpha_pabeta=True)
    
    # id_msa_list = get_id_msa_list(full_sample=False)
    # paper_figure_sample_selection(id_msa_list, color_var='sfr')
    # paper_figure_sample_selection(id_msa_list, color_var='sfr', plot_mass_mags=True)
    # paper_figure_sample_selection(id_msa_list, color_var='ha_eqw', plot_mags=True)
    # paper_figure_sample_selection_twopanel(id_msa_list)


    # paper_figure_sample_selection(id_msa_list, color_var='redshift', plot_sfr_mass=True)

    pass






""" PaAlpha analysis"""
# paschen alpha candidate ids:
# [15350, 17089, 18045, 18708, 19283, 25774, 27621, 29398, 33157, 38987, 42203, 42238, 43497, 48463, 60032]
# After visual SNR filtering:
# [15350, 17089, 18045, 19283, 25774, 27621, 29398, 33157, 38987, 42203, 42238, 43497, 48463]
# Here are their id_dr3s
# [26618, 28495, 29574, 30915, 37776, 39748, 41581, 45334, 51405, 54614, 54643, 56018, 61218]

# Due to phot_sample_df, had to drop 3rd index:
# msa = [15350, 17089, 19283, 25774, 27621, 29398, 33157, 38987, 42203, 42238, 43497, 48463]
# dr3 = [26618, 28495, 30915, 37776, 39748, 41581, 45334, 51405, 54614, 54643, 56018, 61218]


""" PaAlpha with PaBeta"""
# both lines in SED
# [15350, 17089, 18045, 18708, 19283, 25774, 42203, 42238, 48463, 60032]
# dropping same idxs as above for snr and 3rd index
# [15350, 17089, 19283, 25774, 42203, 42238, 48463]
# [26618, 28495, 30915, 37776, 54614, 54643, 61218]