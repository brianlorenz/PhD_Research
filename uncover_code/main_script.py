from uncover_sed_filters import unconver_read_filters, get_filt_cols
from uncover_read_data import read_supercat, read_spec_cat
from compare_sed_spec_flux import compare_sed_flux, compare_all_sed_flux
from fit_emission_uncover import fit_all_emission_uncover

target_lines = 6563, 12820

def main():
    zqual_df = find_good_spec()
    zqual_df_covered = select_spectra(zqual_df)
    id_msa_list = zqual_df_covered['id_msa'].to_list()
    id_msa_list = id_msa_list[0:20]
    # compare_all_sed_flux(id_msa_list)
    fit_all_emission_uncover(id_msa_list)
   
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
    for i in range(len(zqual_df)):
        redshift = zqual_df['z_spec'].iloc[i]
        line1_cover = line_in_range(redshift, target_lines[0], filt_cols, uncover_filt_dir)
        line2_cover = line_in_range(redshift, target_lines[1], filt_cols, uncover_filt_dir)
        both_corered = line1_cover and line2_cover
        if both_corered == True:
            covered_idxs.append(i)
    zqual_df_covered = zqual_df.iloc[covered_idxs]
    zqual_df_covered = zqual_df_covered.reset_index()
    return zqual_df_covered

def line_in_range(z, target_line, filt_cols, uncover_filt_dir):
    z_line = target_line * (1+z)
    covered = False
    for filt in filt_cols:
        if z_line>uncover_filt_dir[filt+'_blue'] and z_line<uncover_filt_dir[filt+'_red']:
            covered = True
    return covered

main()