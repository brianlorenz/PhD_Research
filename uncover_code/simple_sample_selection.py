from uncover_sed_filters import unconver_read_filters, get_filt_cols
from uncover_read_data import read_supercat, read_spec_cat, read_lineflux_cat
from astropy.io import ascii
from uncover_prospector_seds import make_all_prospector
from simple_make_dustmap import make_3color, get_line_coverage
from sedpy import observate
from fit_emission_uncover_wave_divide import line_list
import sys
import pandas as pd

def sample_select():
    ha_snr_thresh = 2
    pab_snr_thresh = 2

    zqual_df = find_good_spec()
    zqual_df_covered = select_spectra(zqual_df)
    zqual_df_covered.to_csv('/Users/brianlorenz/uncover/zqual_df_simple.csv', index=False)
    id_msa_list = zqual_df_covered['id_msa'].to_list()
    total_id_msa_list_df = pd.DataFrame(id_msa_list, columns=['id_msa'])
    total_id_msa_list_df.to_csv('/Users/brianlorenz/uncover/Data/sample_selection/total_before_cuts.csv', index=False)
    lines_df = read_lineflux_cat()

    # Run emission fits / sed+spec generation on this group, then:
    # Need to fix the few that are not working, think it's emissoin fit

    id_msa_good_list = []
    id_msa_filt_edge = []
    id_msa_ha_snr_flag = []
    id_msa_pab_snr_flag = []
    id_msa_line_notfullcover = []
    id_msa_skipped = []
    for id_msa in id_msa_list:
        print(f"Checking sample selection for {id_msa}")
        if id_msa in [42041, 49991]: 
            print(f'Skipping {id_msa} for other issues')
            id_msa_skipped.append(id_msa)
            continue 
            #42041 - not in supercat
            #49991 - not in supercat
        # Read in the images
        ha_filters, ha_images, wht_ha_images, obj_segmap, ha_photfnus, ha_all_filts = make_3color(id_msa, line_index=0, plot=False)
        pab_filters, pab_images, wht_pab_images, obj_segmap, pab_photfnus, pab_all_filts = make_3color(id_msa, line_index=1, plot=False)
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


        # If either halpha or pab is detected in the end filters, decide what to do
        if ha_all_filts == False or pab_all_filts == False:
            print("One of the lines not detected in all filters")
            print("Consider different cont measurement method")
            id_msa_filt_edge.append(id_msa)
            continue

        # Emission fit properties
        fit_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting/{id_msa}_emission_fits.csv').to_pandas()
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

        # Check the coverage fraction of the lines - we want it high in the line, but 0 int he continuum filters
        ha_avg_transmission = get_line_coverage(ha_sedpy_filt, line_list[0][1] * (1+redshift), ha_sigma * (1+redshift))
        pab_avg_transmission = get_line_coverage(pab_sedpy_filt, line_list[1][1] * (1+redshift), pab_sigma * (1+redshift))
        ha_red_avg_transmission = get_line_coverage(ha_red_sedpy_filt, line_list[0][1] * (1+redshift), ha_sigma * (1+redshift))
        pab_red_avg_transmission = get_line_coverage(pab_red_sedpy_filt, line_list[1][1] * (1+redshift), pab_sigma * (1+redshift))
        ha_blue_avg_transmission = get_line_coverage(ha_blue_sedpy_filt, line_list[0][1] * (1+redshift), ha_sigma * (1+redshift))
        pab_blue_avg_transmission = get_line_coverage(pab_blue_sedpy_filt, line_list[1][1] * (1+redshift), pab_sigma * (1+redshift))
        ha_transmissions = [ha_red_avg_transmission, ha_avg_transmission, ha_blue_avg_transmission]
        pab_transmissions = [pab_red_avg_transmission, pab_avg_transmission, pab_blue_avg_transmission]

        if ha_avg_transmission < 0.9 or pab_avg_transmission < 0.9:
            print("One of the lines not covered fully in the filters")
            id_msa_line_notfullcover.append(id_msa)
            continue

        if ha_snr < ha_snr_thresh:
            print(f"Ha SNR of {ha_snr} less than thresh of {ha_snr_thresh}")
            id_msa_ha_snr_flag.append(id_msa)
            continue
        if pab_snr < pab_snr_thresh:
            print(f"PaB SNR of {pab_snr} less than thresh of {pab_snr_thresh}")
            id_msa_pab_snr_flag.append(id_msa)
            continue

        id_msa_good_list.append(id_msa)
    
    assert len(id_msa_list) == len(id_msa_good_list) + len(id_msa_line_notfullcover) + len(id_msa_filt_edge) + len(id_msa_ha_snr_flag) + len(id_msa_pab_snr_flag) + len(id_msa_skipped)
    good_df = pd.DataFrame(id_msa_good_list, columns=['id_msa'])
    line_notfullcover_df = pd.DataFrame(id_msa_line_notfullcover, columns=['id_msa'])
    filt_edge_df = pd.DataFrame(id_msa_filt_edge, columns=['id_msa'])
    ha_snr_flag_df = pd.DataFrame(id_msa_ha_snr_flag, columns=['id_msa'])
    pab_snr_flag_df = pd.DataFrame(id_msa_pab_snr_flag, columns=['id_msa'])
    id_msa_skipped_df = pd.DataFrame(id_msa_skipped, columns=['id_msa'])

    # Write the DataFrame to a text file using a space as a delimiter
    good_df.to_csv('/Users/brianlorenz/uncover/Data/sample_selection/main_sample.csv', index=False)
    line_notfullcover_df.to_csv('/Users/brianlorenz/uncover/Data/sample_selection/line_notfullcover_df.csv', index=False)
    filt_edge_df.to_csv('/Users/brianlorenz/uncover/Data/sample_selection/filt_edge.csv', index=False)
    ha_snr_flag_df.to_csv('/Users/brianlorenz/uncover/Data/sample_selection/ha_snr_flag.csv', index=False)
    pab_snr_flag_df.to_csv('/Users/brianlorenz/uncover/Data/sample_selection/pab_snr_flag.csv', index=False)
    id_msa_skipped_df.to_csv('/Users/brianlorenz/uncover/Data/sample_selection/id_msa_skipped.csv', index=False)

    return

def find_good_spec():
    """ Reads in spectra catalog and makes sure quality is good"""
    zqual_df = read_spec_cat()
    zqual_df = zqual_df[zqual_df['flag_zspec_qual'] == 3]
    zqual_df = zqual_df[zqual_df['flag_spec_qual'] == 0]
    return zqual_df

def select_spectra(zqual_df):
    """Checking that both target lines are covered in the photometry"""
    uncover_filt_dir, filters = unconver_read_filters()
    supercat_df = read_supercat()
    filt_cols = get_filt_cols(supercat_df, skip_wide_bands=True)
    covered_idxs = []
    line0_filts = []
    line1_filts = []
    for i in range(len(zqual_df)):
        redshift = zqual_df['z_spec'].iloc[i]
        line1_cover, line0_filt_name = line_in_range(redshift, line_list[0][1], filt_cols, uncover_filt_dir)
        line2_cover, line1_filt_name = line_in_range(redshift, line_list[1][1], filt_cols, uncover_filt_dir)
        both_corered = line1_cover and line2_cover
        if both_corered == True:
            covered_idxs.append(i)
            line0_filts.append(line0_filt_name)
            line1_filts.append(line1_filt_name)
    zqual_df_covered = zqual_df.iloc[covered_idxs]
    zqual_df_covered = zqual_df_covered.reset_index()
    zqual_df_covered['line0_filt'] = line0_filts
    zqual_df_covered['line1_filt'] = line1_filts
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

sample_select()