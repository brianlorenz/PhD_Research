from uncover_read_data import read_supercat, read_SPS_cat_all
from uncover_sed_filters import unconver_read_filters, get_filt_cols
import pandas as pd
import numpy as np
import time

save_loc = '/Users/brianlorenz/uncover/Data/generated_tables/phot_linecoverage.csv'

line_list = [
    ('Hbeta', 4861),
    ('OIII', 5008),
    ('Halpha', 6564.6),
    ('PaBeta', 12821.7),
    ('PaAlpha', 18750)
]

# line_list = [
#     ('Halpha', 6564.6),
#     ('PaBeta', 12821.7)
# ]

def full_phot_sample_select():
    supercat_df = read_supercat()
    sps_df = read_SPS_cat_all()
    uncover_filt_dict, filters = unconver_read_filters()
    filt_colnames = get_filt_cols(supercat_df, skip_wide_bands=True)

    id_DR3_list = supercat_df['id'].to_list()
    phot_sample_df = pd.DataFrame(zip(id_DR3_list), columns=['id'])
    merged_df = phot_sample_df.merge(sps_df, on='id')
    phot_sample_df['z_16'] = merged_df['z_16']
    phot_sample_df['z_50'] = merged_df['z_50']
    phot_sample_df['z_84'] = merged_df['z_84']
    phot_sample_df = phot_sample_df.fillna(-99)

    # for line in line_list:
    #     add_line_columns(phot_sample_df, line[0])
   
    for line in line_list:
        check_line_in_filters(phot_sample_df, line[0], line[1], uncover_filt_dict, filters, filt_colnames, sps_df)

    phot_sample_df.to_csv(save_loc, index=False)


def check_line_in_filters(dataframe, line_name, line_wave, uncover_filt_dict, filters, filt_colnames, sps_df):
    """Checks if the listed line is within a filter. If so, record that filter, the nearby filters for continuum, and the redshift sigma to shift it out of the filter
    
    Parameters:
    dataframe (pd.Dataframe): df to save the information to
    line_name (str): line name, matching column names in dataframe
    line_wave (float): wavelength in angstroms
    uncover_filt_dict (dict): from uncover_read_filters
    filters (list): list of sedpy filters, from uncover_read_filters
    filt_colnames (list): list of filter columnnames, from get_filter_colnames
    sps_df (dataframe): the sps catalog from UNCOVER
    """

    # This was much faster for computing to append to lists, then join the list to the dataframe at the end
    obs_filts = []
    blue_filts = []
    red_filts = []
    redshift_sigs = []

    for i in range(len(dataframe)):
        id_dr3 = dataframe['id'].iloc[i]
        sps_row = sps_df[sps_df['id'] == id_dr3]
        
        z50 = sps_row['z_50'].iloc[0]
        z16 = sps_row['z_16'].iloc[0]
        z84 = sps_row['z_84'].iloc[0] 

        # Check if the z50 puts the target line within any medium band, but not near the edges
        line_covered, detected_filt_name, redshift_sigma = line_in_range(z50, z16, z84, line_wave, filt_colnames, uncover_filt_dict)

        # Skip the object if the line is not within any filters
        if line_covered == False:
            obs_filts.append(-99)
            blue_filts.append(-99)
            red_filts.append(-99)
            redshift_sigs.append(-99)
            continue
        # Otherwise, continue and fill in the columns
        
        # Find the continuum filters
        filt_cont_blue_name, filt_cont_red_name = find_nearby_filters(detected_filt_name, filt_colnames)

        
        # dataframe.loc[i, f'{line_name}_filter_obs'] = detected_filt_name
        # dataframe.loc[i, f'{line_name}_filter_bluecont'] = filt_cont_blue_name
        # dataframe.loc[i, f'{line_name}_filter_redcont'] = filt_cont_red_name
        # dataframe.loc[i, f'{line_name}_redshift_sigma'] = redshift_sigma
        obs_filts.append(detected_filt_name)
        blue_filts.append(filt_cont_blue_name)
        red_filts.append(filt_cont_red_name)
        redshift_sigs.append(redshift_sigma)

        if i%100==0:
            print(i)
    dataframe[f'{line_name}_filter_obs'] = obs_filts
    dataframe[f'{line_name}_filter_bluecont'] = blue_filts
    dataframe[f'{line_name}_filter_redcont'] = red_filts
    dataframe[f'{line_name}_redshift_sigma'] = redshift_sigs
    return dataframe       


# def add_line_columns(dataframe, line_name):
#     """Adds the columns for recording the filter containing the line, surrounding filters, and redshift uncertainy

#     Parameters:
#     dataframe (pd.DataFrame): df to append the columns to
#     line_name (str): line name for header of all columns
#     """
#     append_99s = -99*np.ones(len(dataframe))

#     dataframe[f'{line_name}_filter_obs'] = append_99s
#     dataframe[f'{line_name}_filter_bluecont'] = append_99s
#     dataframe[f'{line_name}_filter_redcont'] = append_99s
#     dataframe[f'{line_name}_redshift_sigma'] = append_99s

#     return dataframe


def line_in_range(z50, z16, z84, line_wave, filt_cols, uncover_filt_dict, wavelength_pad=10):
    """Checks if the target emission line falls within any of the listed filt_cols

    z50 (float): z_50 from prospector SPS catalog
    z16 (float): z_16 from prospector
    z84 (float): z_84 from prospector
    line_wave (float): wavelength in angstroms
    filt_cols (list): list of names of the filters to check
    uncover_filt_dict (dict): from uncover_read_filters
    wavelength_pad (float): How far the line needs to be away from the filter edge, in angstroms
    
    """
    sigma = -99
    z_line = line_wave * (1+z50)
    lower_z_1sigma = z50-z16
    upper_z_1sigma = z84-z50
    covered = False
    filt_name = ''
    for filt in filt_cols:
        filt_lower_edge = uncover_filt_dict[filt+'_blue']+wavelength_pad
        filt_upper_edge = uncover_filt_dict[filt+'_red']-wavelength_pad
        if z_line>filt_lower_edge and z_line<filt_upper_edge:
            covered = True
            filt_name = filt

            # Assess how many sigma in redshift we have to go to move the line out of the filt
            lower_edge_redshift = (filt_lower_edge/line_wave)-1
            upper_edge_redshift = (filt_upper_edge/line_wave)-1
            lower_edge_offset = z50 - lower_edge_redshift
            upper_edge_offset = upper_edge_redshift - z50

            lower_z_sigma_measure = lower_edge_offset/lower_z_1sigma
            upper_z_sigma_measure = upper_edge_offset/upper_z_1sigma

            # Take the lesser of the answers     
            sigma = np.min([lower_z_sigma_measure, upper_z_sigma_measure])

    return covered, filt_name, sigma
    
def find_nearby_filters(detected_filt, filt_names):
    """ Finds the continuum filters surrounding the line

    Parameters:
    detected_filt (str): Name of the filter within filt_names that the line is detected in
    filt_names (str): All filter names, sorted by increasing wavelength
    """
    detected_index = [i for i in range(len(filt_names)) if filt_names[i] == detected_filt][0]
    if detected_index == 0: # If it's at the bluest edge, then you don't get a blue filter
        filt_red = filt_names[detected_index+1]
        filt_blue = -99
        return filt_blue, filt_red
    if detected_index == len(filt_names)-1: # opposite for reddest edge
        filt_red = -99
        filt_blue = filt_names[detected_index-1]
        return filt_blue, filt_red
    
    # Otherwise, just grab the nearest filters, avoiding overlaps
    add_index = 1
    subtract_index = 1

    if detected_filt in ['f_f335m', 'f_f410m']: # These have overlaps with the next reddest filter
        add_filt = 2
    filt_red = filt_names[detected_index+add_index]
    if detected_filt in ['f_f360m', 'f_f430m', 'f_f480m']: # These overlap with next bluest filter
        subtract_filt = 2
    filt_blue = filt_names[detected_index-subtract_index]
    return filt_blue, filt_red


def get_filt_cols(df, skip_wide_bands=False):
    filt_cols = [col for col in df.columns if 'f_' in col]
    filt_cols = [col for col in filt_cols if 'alma' not in col]
    if skip_wide_bands ==  True:
        filt_cols = [col for col in filt_cols if 'w' not in col]
    return filt_cols

if __name__ == "__main__":
    full_phot_sample_select()