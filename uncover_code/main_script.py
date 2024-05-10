from uncover_sed_filters import unconver_read_filters, get_filt_cols
from uncover_read_data import read_supercat, read_spec_cat
from compare_sed_spec_flux import compare_sed_flux, compare_all_sed_flux
from fit_emission_uncover import fit_all_emission_uncover
from astropy.io import ascii
from make_color_images import make_all_3color

target_lines = 6563, 12820

def main(redo_fit=False):
    zqual_df = find_good_spec()
    zqual_df_covered = select_spectra(zqual_df)
    id_msa_list = zqual_df_covered['id_msa'].to_list()
    
    #Ensure that all ids are in the catalog:
    supercat = read_supercat()
    id_msa_list = [id_msa for id_msa in id_msa_list if len(supercat[supercat['id_msa'] == id_msa]) == 1]
    
    if redo_fit == True:
        compare_all_sed_flux(id_msa_list) 
        fit_all_emission_uncover(id_msa_list)
    detected_list = select_detected_lines(id_msa_list)
    zqual_df_detected = zqual_df_covered[zqual_df_covered['id_msa'].isin(detected_list)]
    zqual_df_detected.to_csv('/Users/brianlorenz/uncover/zqual_detected.csv', index=False)
    make_all_3color(detected_list)
    print(detected_list)
    
   
def find_good_spec():
    """ Reads in spectra catalog and makes sure quality is good"""
    zqual_df = read_spec_cat()
    zqual_df = zqual_df[zqual_df['flag_zspec_qual'] == 3]
    zqual_df = zqual_df[zqual_df['flag_spec_qual'] == 0]
    return zqual_df

def select_spectra(zqual_df):
    """Checking that both target lines are covered int he photometry"""
    uncover_filt_dir, filters = unconver_read_filters()
    supercat_df = read_supercat()
    filt_cols = get_filt_cols(supercat_df)
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

def select_detected_lines(id_msa_list, thresh = 5):
    """Selects the ids that have detected emission lines above SNR of thresh
    
    Parameters:
    thresh (float): SNR required to count as a dtection
    """
    detected_list = []
    for id_msa in id_msa_list:
        emission_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting/{id_msa}_emission_fits.csv').to_pandas()
        ha_snr = emission_df[emission_df['line_name']=='Halpha']['signal_noise_ratio'].iloc[0]
        pab_snr = emission_df[emission_df['line_name']=='PaBeta']['signal_noise_ratio'].iloc[0]
        if ha_snr > thresh and pab_snr > thresh:
            detected_list.append(id_msa)
    return detected_list

def line_in_range(z, target_line, filt_cols, uncover_filt_dir):
    z_line = target_line * (1+z)
    covered = False
    filt_name = ''
    for filt in filt_cols:
        if z_line>uncover_filt_dir[filt+'_blue'] and z_line<uncover_filt_dir[filt+'_red']:
            covered = True
            filt_name = filt
    return covered, filt_name

main()