from uncover_sed_filters import unconver_read_filters, get_filt_cols
from uncover_read_data import read_supercat, read_spec_cat, read_lineflux_cat, get_id_msa_list, read_SPS_cat
from astropy.io import ascii
from uncover_prospector_seds import make_all_prospector
from simple_make_dustmap import make_3color, get_line_coverage, ha_trasm_thresh, pab_trasm_thresh
from sedpy import observate
from fit_emission_uncover_wave_divide import line_list
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from plot_vals import stellar_mass_label, scale_aspect

def sample_select():
    ha_snr_thresh = 0
    pab_snr_thresh = 5
    overlap_thresh = 0.2

    zqual_df = find_good_spec()
    zqual_df_covered = select_spectra(zqual_df)
    line_not_covered_id_msas = []
    for id_msa in zqual_df['id_msa']:
        if len(zqual_df_covered[zqual_df_covered['id_msa'] == id_msa]) == 0:
            line_not_covered_id_msas.append(id_msa)
    line_not_covered_id_msas_df = pd.DataFrame(line_not_covered_id_msas, columns=['id_msa'])
    line_not_covered_id_msas_df.to_csv('/Users/brianlorenz/uncover/Data/sample_selection/line_not_in_filt.csv', index=False)
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
    id_msa_line_overlapcont = []
    id_msa_skipped = []

    cont_overlap_flag = []
    cont_overlap_value = []
    line_notfullcover_flag = []
    line_notfullcover_value = []

    for id_msa in id_msa_list:
        print(f"Checking sample selection for {id_msa}")
        if id_msa in [6325, 42041, 49991]: 
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
            

        if ha_snr < ha_snr_thresh:
            print(f"Ha SNR of {ha_snr} less than thresh of {ha_snr_thresh}")
            id_msa_ha_snr_flag.append(id_msa)
            continue
        if pab_snr < pab_snr_thresh:
            print(f"PaB SNR of {pab_snr} less than thresh of {pab_snr_thresh}")
            id_msa_pab_snr_flag.append(id_msa)
            continue

        id_msa_good_list.append(id_msa)
    
    assert len(id_msa_list) == len(id_msa_good_list) + len(id_msa_line_notfullcover) + len(id_msa_filt_edge) + len(id_msa_ha_snr_flag) + len(id_msa_pab_snr_flag) + len(id_msa_skipped) + len(id_msa_line_overlapcont)
    good_df = pd.DataFrame(id_msa_good_list, columns=['id_msa'])
    line_notfullcover_df = pd.DataFrame(zip(id_msa_line_notfullcover, line_notfullcover_flag, line_notfullcover_value), columns=['id_msa', 'flag_line_coverage', 'line_trasm_value'])
    filt_edge_df = pd.DataFrame(id_msa_filt_edge, columns=['id_msa'])
    ha_snr_flag_df = pd.DataFrame(id_msa_ha_snr_flag, columns=['id_msa'])
    pab_snr_flag_df = pd.DataFrame(id_msa_pab_snr_flag, columns=['id_msa'])
    cont_overlap_df = pd.DataFrame(zip(id_msa_line_overlapcont, cont_overlap_flag, cont_overlap_value), columns=['id_msa', 'flag_cont_overlap', 'cont_overlap_value'])
    id_msa_skipped_df = pd.DataFrame(id_msa_skipped, columns=['id_msa'])

    # Write the DataFrame to a text file using a space as a delimiter
    good_df.to_csv('/Users/brianlorenz/uncover/Data/sample_selection/main_sample.csv', index=False)
    line_notfullcover_df.to_csv('/Users/brianlorenz/uncover/Data/sample_selection/line_notfullcover_df.csv', index=False)
    filt_edge_df.to_csv('/Users/brianlorenz/uncover/Data/sample_selection/filt_edge.csv', index=False)
    ha_snr_flag_df.to_csv('/Users/brianlorenz/uncover/Data/sample_selection/ha_snr_flag.csv', index=False)
    pab_snr_flag_df.to_csv('/Users/brianlorenz/uncover/Data/sample_selection/pab_snr_flag.csv', index=False)
    id_msa_skipped_df.to_csv('/Users/brianlorenz/uncover/Data/sample_selection/id_msa_skipped.csv', index=False)
    cont_overlap_df.to_csv('/Users/brianlorenz/uncover/Data/sample_selection/cont_overlap_line.csv', index=False)
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


def paper_figure_sample_selection(id_msa_list, color_var='None'):
    fig, ax = plt.subplots(figsize=(6,6))

    fontsize = 14
    markersize=8
    gray_markersize = 4

    zqual_df = read_spec_cat()
    sps_df = read_SPS_cat()
    # supercat_df = read_supercat()
    lineratio_data_df = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/lineratio_av_df.csv').to_pandas()

    line_notfullcover_df = ascii.read('/Users/brianlorenz/uncover/Data/sample_selection/line_notfullcover_df.csv').to_pandas()
    filt_edge_df = ascii.read('/Users/brianlorenz/uncover/Data/sample_selection/filt_edge.csv').to_pandas()
    ha_snr_flag_df = ascii.read('/Users/brianlorenz/uncover/Data/sample_selection/ha_snr_flag.csv').to_pandas()
    pab_snr_flag_df = ascii.read('/Users/brianlorenz/uncover/Data/sample_selection/pab_snr_flag.csv').to_pandas()
    id_msa_skipped_df = ascii.read('/Users/brianlorenz/uncover/Data/sample_selection/id_msa_skipped.csv').to_pandas()
    id_msa_cont_overlap_line_df = ascii.read('/Users/brianlorenz/uncover/Data/sample_selection/cont_overlap_line.csv').to_pandas()
    id_msa_not_in_filt_df = ascii.read('/Users/brianlorenz/uncover/Data/sample_selection/line_not_in_filt.csv').to_pandas()

    id_redshift_issue_list = line_notfullcover_df['id_msa'].append(filt_edge_df['id_msa']).append(id_msa_skipped_df['id_msa']).append(id_msa_not_in_filt_df['id_msa']).append(id_msa_cont_overlap_line_df['id_msa']).to_list()

    # Gray background
    for id_msa in zqual_df['id_msa']:
        marker = 'o'
        zqual_row = zqual_df[zqual_df['id_msa'] == id_msa]

        if zqual_row['flag_zspec_qual'].iloc[0] != 3 or zqual_row['flag_spec_qual'].iloc[0] != 0:
            marker = 's'
        if id_msa in id_redshift_issue_list:
            marker = '^'

        redshift = zqual_row['z_spec'].iloc[0]
        if redshift < 1.3 or redshift > 2.4:
            continue
        id_dr2 = zqual_df[zqual_df['id_msa']==id_msa]['id_DR2'].iloc[0]
        sps_row = sps_df[sps_df['id']==id_dr2]
        stellar_mass_50 = sps_row['mstar_50']
        if len(sps_row) == 0:
            print(f'No SPS for {id_msa}')
            continue
        if marker=='o':
            print(id_msa)

        ax.plot(redshift, stellar_mass_50, marker=marker, color='gray', ls='None', ms=gray_markersize)

    # Selected Sample
    for id_msa in id_msa_list:
        zqual_row = zqual_df[zqual_df['id_msa'] == id_msa]
        # supercat_row = supercat_df[supercat_df['id_msa']==id_msa]
        redshift = zqual_row['z_spec'].iloc[0]
        id_dr2 = zqual_df[zqual_df['id_msa']==id_msa]['id_DR2'].iloc[0]
        sps_row = sps_df[sps_df['id']==id_dr2]

        stellar_mass_50 = sps_row['mstar_50']
        err_stellar_mass_low = stellar_mass_50 - sps_row['mstar_16']
        err_stellar_mass_high = sps_row['mstar_84'] - stellar_mass_50
        dust2_50 = sps_row['dust2_50'].iloc[0]

        # data_df_row = data_df[data_df['id_msa'] == id_msa]
        lineratio_data_row = lineratio_data_df[lineratio_data_df['id_msa'] == id_msa]
        fit_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting/{id_msa}_emission_fits.csv').to_pandas()
        # ha_snr = fit_df['signal_noise_ratio'].iloc[0]
        pab_snr = fit_df['signal_noise_ratio'].iloc[0]
        
        cmap = mpl.cm.inferno
        
        if color_var == 'sed_av':
            norm = mpl.colors.Normalize(vmin=0, vmax=3) 
            rgba = cmap(norm(lineratio_data_row['sed_av']))  
            cbar_label = 'Photometry AV'
        # if color_var == 'ha_snr':
        #     norm = mpl.colors.LogNorm(vmin=2, vmax=100) 
        #     rgba = cmap(norm(ha_snr))
        #     cbar_label = 'H$\\alpha$ SNR'
        # if color_var == 'pab_snr':
        #     norm = mpl.colors.LogNorm(vmin=2, vmax=50) 
        #     rgba = cmap(norm(pab_snr))
        #     cbar_label = 'Pa$\\beta$ SNR'
        if color_var == 'dust2':
            norm = mpl.colors.Normalize(vmin=0, vmax=2) 
            rgba = cmap(norm(dust2_50))
            cbar_label = 'Prospector dust2_50'
        if color_var != 'None':
            color_str = f'_{color_var}'
        else:
            rgba = 'black'
            color_str = ''

        ax.errorbar(redshift, stellar_mass_50, yerr=[err_stellar_mass_low, err_stellar_mass_high], marker='o', color=rgba, ls='None', mec='black', ms=markersize)
    ax.set_ylabel('Prospector '+stellar_mass_label, fontsize=fontsize)
    ax.set_xlabel('Redshift', fontsize=fontsize)


    
     
    ax.tick_params(labelsize=fontsize)
    # ax.set_xlim([1e-20, 1e-15])
    # ax.set_ylim([1e-20, 1e-15])
    scale_aspect(ax)

    # cb_ax = fig.add_axes([.91,.124,.02,.734])
    if color_var != 'None':
        sm =  mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm,orientation='vertical', ax=ax)
        cbar.set_label(cbar_label, fontsize=16)
        cbar.ax.tick_params(labelsize=16)

    save_loc = f'/Users/brianlorenz/uncover/Figures/paper_figures/sample_selection{color_str}.pdf'
    fig.savefig(save_loc)


if __name__ == "__main__":
    # sample_select()
    
    id_msa_list = get_id_msa_list(full_sample=False)
    paper_figure_sample_selection(id_msa_list, color_var='sed_av')
    