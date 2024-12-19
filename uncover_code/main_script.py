from uncover_sed_filters import unconver_read_filters, get_filt_cols
from uncover_read_data import read_supercat, read_spec_cat, read_lineflux_cat
from compare_sed_spec_flux import compare_sed_flux, compare_all_sed_flux
from fit_emission_uncover_old import fit_all_emission_uncover
from astropy.io import ascii
from make_dust_maps_old import make_all_3color, make_all_dustmap
from make_dust_maps_old import find_filters_around_line
from uncover_prospector_seds import make_all_prospector
target_lines = 6563, 12820

def main(redo_fit=False):
    zqual_df = find_good_spec()
    zqual_df_covered = select_spectra(zqual_df)
    id_msa_list = zqual_df_covered['id_msa'].to_list()
    
    # Ensure that all ids are in the catalog:
    supercat = read_supercat()
    id_msa_list = [id_msa for id_msa in id_msa_list if len(supercat[supercat['id_msa'] == id_msa]) == 1]
    if redo_fit == True:
        make_all_prospector(id_msa_list)
        compare_all_sed_flux(id_msa_list) 
        fit_all_emission_uncover(id_msa_list)
    
    detected_list, ha_detected_list, detected_snrs, ha_detected_snrs, all_ha_snrs = select_detected_lines(id_msa_list)
    breakpoint()
    zqual_df_detected = zqual_df_covered[zqual_df_covered['id_msa'].isin(detected_list)]
    zqual_df_detected.to_csv('/Users/brianlorenz/uncover/zqual_detected.csv', index=False)

    zqual_df_ha_detected = zqual_df_covered[zqual_df_covered['id_msa'].isin(ha_detected_list)]
    zqual_df_ha_detected.to_csv('/Users/brianlorenz/uncover/zqual_df_ha_detected.csv', index=False)
    
    # Check if there are filters on both sides for continuum:
    def check_cont_coverage(det_list):
        cont_covered = []
        for id_msa in det_list:
            _,_,_, all_good0 = find_filters_around_line(id_msa, 0)
            _,_,_, all_good1 = find_filters_around_line(id_msa, 1)
            if all_good0 == True and all_good1 == True:
                cont_covered.append(id_msa)
        return cont_covered
    
    detected_cont_covered = check_cont_coverage(detected_list)
    ha_detected_cont_covered = check_cont_coverage(ha_detected_list)

    zqual_df_cont_covered = zqual_df_detected[zqual_df_detected['id_msa'].isin(detected_cont_covered)]
    zqual_df_cont_covered.to_csv('/Users/brianlorenz/uncover/zqual_df_cont_covered.csv', index=False)

    zqual_df_ha_cont_covered = zqual_df_ha_detected[zqual_df_ha_detected['id_msa'].isin(ha_detected_cont_covered)]
    zqual_df_ha_cont_covered.to_csv('/Users/brianlorenz/uncover/zqual_df_ha_cont_covered.csv', index=False)
    
    make_all_3color(detected_list)
    make_all_dustmap()
    print(detected_list)
    
   
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
        line1_cover, line0_filt_name = line_in_range(redshift, target_lines[0], filt_cols, uncover_filt_dir)
        line2_cover, line1_filt_name = line_in_range(redshift, target_lines[1], filt_cols, uncover_filt_dir)
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

def select_detected_lines(id_msa_list, thresh = 2):
    """Selects the ids that have detected emission lines above SNR of thresh
    
    Parameters:
    thresh (float): SNR required to count as a dtection
    """
    detected_list = []
    detected_snrs = []
    ha_detected_list = []
    ha_detected_snrs = []
    all_ha_snrs = []

    lines_df = read_lineflux_cat()
    for id_msa in id_msa_list:
        lines_df_row = lines_df[lines_df['id_msa'] == id_msa]
        emission_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting/{id_msa}_emission_fits.csv').to_pandas()
        ha_snr = emission_df[emission_df['line_name']=='Halpha']['signal_noise_ratio'].iloc[0]
        pab_snr = emission_df[emission_df['line_name']=='PaBeta']['signal_noise_ratio'].iloc[0]

        ha_flux_cat = lines_df_row['f_Ha+NII'].iloc[0]
        ha_err_cat = lines_df_row['e_Ha+NII'].iloc[0]
        pab_flux_cat = lines_df_row['f_PaB'].iloc[0]
        pab_err_cat = lines_df_row['e_PaB'].iloc[0]
        ha_snr = ha_flux_cat / ha_err_cat
        pab_snr = pab_flux_cat / pab_err_cat

        if ha_snr > thresh and pab_snr > thresh:
            detected_list.append(id_msa)
            detected_snrs.append((ha_snr, pab_snr))
        if ha_snr > 1:
            ha_detected_list.append(id_msa)
            ha_detected_snrs.append((ha_snr, pab_snr))
        all_ha_snrs.append(ha_snr)
        
    return detected_list, ha_detected_list, detected_snrs, ha_detected_snrs, all_ha_snrs

def line_in_range(z, target_line, filt_cols, uncover_filt_dir):
    z_line = target_line * (1+z)
    covered = False
    filt_name = ''
    for filt in filt_cols:
        if z_line>uncover_filt_dir[filt+'_blue'] and z_line<uncover_filt_dir[filt+'_red']:
            covered = True
            filt_name = filt
    return covered, filt_name

main(redo_fit=False)