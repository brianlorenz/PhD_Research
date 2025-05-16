# Code to create sample_selection - determines which galaxies match the selection criteria

from uncover_input_data import read_supercat, read_SPS_cat, halpha_name, halpha_wave, line_coverage_path
from uncover_filters import unconver_read_filters, get_filt_cols
import pandas as pd
import numpy as np


def make_line_coverage(line_name, line_wave, save_path):
    """Generates a line coverage table for all objects the given emission line

    Parameters:
    line_name (str): Name for the emission line - used for column headers
    line_wave (float): Wavelength in angstroms
    save_path (str): File path for where to save the line coverage data

    Returns:
    Nothing - saves the line coverage table
    """
    supercat_df = read_supercat()
    sps_df = read_SPS_cat()
    uncover_filt_dict, sedpy_filters = unconver_read_filters() # Builds a useful dictionary and list of sedpy filters - see inside that file
    filt_names = get_filt_cols(supercat_df, skip_wide_bands=True) # Grabs the names of only the medium bands

    # Builds the dataframe to save - will have the object ID, then all the redshift info and flags
    id_DR3_list = supercat_df['id'].to_list()
    line_coverage_df = pd.DataFrame(zip(id_DR3_list), columns=['id'])
    merged_df_sps = line_coverage_df.merge(sps_df, on='id')
    merged_df_super = line_coverage_df.merge(supercat_df, on='id')
    line_coverage_df['z_16'] = merged_df_sps['z_16']
    line_coverage_df['z_50'] = merged_df_sps['z_50']
    line_coverage_df['z_84'] = merged_df_sps['z_84']
    line_coverage_df['use_phot'] = merged_df_super['use_phot']
    line_coverage_df['flag_nearbcg'] = merged_df_super['flag_nearbcg']
    line_coverage_df = line_coverage_df.fillna(-99)
    
    # Next, determine if each object has emission lines in the filters
    # This will add the results directly to the DataFrame
    line_coverage_df = check_line_in_filters(line_coverage_df, line_name, line_wave, uncover_filt_dict, filt_names, supercat_df)
    
    # Save the dataframe
    line_coverage_df.to_csv(f'{save_path}_{line_name}.csv', index=False)


def check_line_in_filters(dataframe, line_name, line_wave, uncover_filt_dict, filt_colnames, supercat_df):
    """Checks if the listed line is within a filter. If so, record that filter, the nearby filters for continuum, and the redshift sigma to shift it out of the filter
    
    Parameters:
    dataframe (pd.Dataframe): DataFrame to save the information to
    line_name (str): line name, will be used for column names in dataframe
    line_wave (float): wavelength in angstroms
    uncover_filt_dict (dict): from uncover_filters
    filt_colnames (list): list of filter column names, from uncover_filters
    supercat_df (pd.Dataframe): the SUPER catalog from UNCOVER
    """

    # This was much faster for computing to append to lists, then join the list to the dataframe at the end
    # I'm also sure there is a better way to do this :)
    obs_filts = [] # Observed filter that has the line
    blue_filts = [] # Blue side of continuum filters
    red_filts = [] # Red side of continuum filters
    redshift_sigs = [] # See note where this is computed - number of sigma in redshift uncertainty to fall out of the filter
    all_detected = [] # 1 if all of obs,blue,red have detected filters. 0 if any of them do not 

    # Loop through the DataFrame
    for i in range(len(dataframe)):
        if i%100==0:
            print(f'{i} / {len(dataframe)}') # To show progress, print a statement 100 objects

        id_dr3 = dataframe['id'].iloc[i]
        
        z16 = dataframe['z_16'].iloc[i]
        z50 = dataframe['z_50'].iloc[i]
        z84 = dataframe['z_84'].iloc[i] 

        # Check if the z50 puts the target line within any medium band, but not near the edges
        line_covered, detected_filt_name, redshift_sigma = line_in_range(z50, z16, z84, line_wave, filt_colnames, uncover_filt_dict)

        # Skip the object if the line is not within any filters
        if line_covered == False:
            obs_filts.append(-99)
            blue_filts.append(-99)
            red_filts.append(-99)
            redshift_sigs.append(-99)
            all_detected.append(0)
            continue
        # Otherwise, continue and fill in the columns
        
        # Find the continuum filters that surround the line. Not necessarily the closes two, since sometimes they can overlap
        filt_cont_blue_name, filt_cont_red_name = find_nearby_filters(detected_filt_name, filt_colnames)

        # Store the results
        obs_filts.append(detected_filt_name)
        blue_filts.append(filt_cont_blue_name)
        red_filts.append(filt_cont_red_name)
        redshift_sigs.append(redshift_sigma)
        # Need to check if all_detected is True of False
        
        # If any of the filters are not named, then they are not all detected
        if detected_filt_name == -99 or filt_cont_red_name == -99 or filt_cont_blue_name == -99:
            all_detected.append(0)
            continue
        
        # If the flux of the object in any of the filters is null (empty), then it is not detected
        supercat_row = supercat_df[supercat_df['id']==id_dr3]
        null_obs = pd.isnull(supercat_row[detected_filt_name].iloc[0])
        null_red = pd.isnull(supercat_row[filt_cont_red_name].iloc[0])
        null_blue = pd.isnull(supercat_row[filt_cont_blue_name].iloc[0])
        
        if null_obs + null_red + null_blue == 0:
            all_detected.append(1) # If none are null, then it is detected!
        else:
            all_detected.append(0)

    # After looping through all the objects, we add the results to the dataframe
    dataframe[f'{line_name}_filter_obs'] = obs_filts
    dataframe[f'{line_name}_filter_bluecont'] = blue_filts
    dataframe[f'{line_name}_filter_redcont'] = red_filts
    dataframe[f'{line_name}_redshift_sigma'] = redshift_sigs
    dataframe[f'{line_name}_all_detected'] = all_detected
    return dataframe       


def line_in_range(z50, z16, z84, line_wave, filt_cols, uncover_filt_dict):
    """Checks if the target emission line falls within any of the listed filt_cols

    Parameters:
    z50 (float): z_50 from prospector SPS catalog
    z16 (float): z_16 from prospector SPS catalog
    z84 (float): z_84 from prospector SPS catalog
    line_wave (float): wavelength in angstroms
    filt_cols (list): list of names of the filters to check 
    uncover_filt_dict (dict): from uncover_filters
    
    Returns:
    line_covered (boolean): True/False for whether or not the line falls in a filter
    detected_filt_name (str): The name of the filter the line is in (or empty if not covered)
    sigma (float): How many sigma in redshift uncertanty until the line falls out of the filter (see note in code below)
    """
    line_covered = False # Will return False if nothing changes this value
    detected_filt_name = '' # Will return an empty filter if nothing changes
    redshift_sigma = -99 # Will return -99 if nothing changes this value


    redshifted_line_wave = line_wave * (1+z50) # Location of the line at the object redshift
    lower_z_1sigma = z50-z16 # 1 sigma redshift uncertainties in either direction
    upper_z_1sigma = z84-z50
    
    # Loop through each of the possible filter names
    for filt in filt_cols:
        filt_lower_edge = uncover_filt_dict[filt+'_lower_20pct_wave'] # Grab the 20% transmission cutoffs computed in uncover_filters.py
        filt_upper_edge = uncover_filt_dict[filt+'_upper_20pct_wave']
        if redshifted_line_wave>filt_lower_edge and redshifted_line_wave<filt_upper_edge: # Check if the redshifted line is in the filter at all
            
            # If yes, great! Then the line is covered by something, and this will store the name of the filter
            line_covered = True
            detected_filt_name = filt

            # Assess how many sigma in redshift we have to go to move the line out of the filt
            # Essentially asks, "given the redshift uncertainty, how many sigma in redshift do we have to move before the line falls out of the filter?"
            # e.g. assume z=3, uncertainty=0.5 : If the filter edges are at z=1.5 and z=4, then it would take 3sigma to fall out of the left edge, and 2 sigma to fall out of the right edge
            # In that case, the code would return "2" as the value for sigma. And we would then be confident to 2sigma that the line is in the filter (95%)
            # There are many other ways to do this I'm sure, but this is where I started
            lower_edge_redshift = (filt_lower_edge/line_wave)-1
            upper_edge_redshift = (filt_upper_edge/line_wave)-1
            lower_edge_offset = z50 - lower_edge_redshift
            upper_edge_offset = upper_edge_redshift - z50

            lower_z_sigma_measure = lower_edge_offset/lower_z_1sigma
            upper_z_sigma_measure = upper_edge_offset/upper_z_1sigma

            # Take the lesser of the answers     
            redshift_sigma = np.min([lower_z_sigma_measure, upper_z_sigma_measure])

    return line_covered, detected_filt_name, redshift_sigma 
    
def find_nearby_filters(detected_filt, filt_names):
    """ Finds the continuum filters surrounding the line

    Parameters:
    detected_filt (str): Name of the filter within filt_names that the line is detected in
    filt_names (str): All filter names, sorted by increasing wavelength (the default)

    Returns:
    filt_blue (str): Name of the blue filter for continuum (or -99 if none)
    filt_red (str): Name of the red filter for continuum (or -99 if none)
    """
    detected_index = [i for i in range(len(filt_names)) if filt_names[i] == detected_filt][0]
    if detected_index == 0: # If it's at the bluest edge, then you don't get a blue filter. Return -99
        filt_red = filt_names[detected_index+1]
        filt_blue = -99
        return filt_blue, filt_red
    if detected_index == len(filt_names)-1: # If it's at the reddest edge, then you don't get a red filter. Return -99
        filt_red = -99
        filt_blue = filt_names[detected_index-1]
        return filt_blue, filt_red
    
    # Otherwise, get the nearest filters, avoiding overlaps
    # These are saying "take the index of the filter, than add 1 or subtract 1 to get the next closest filters"
    add_index = 1
    subtract_index = 1

    if detected_filt in ['f_f335m', 'f_f410m']: # These have overlaps with the next reddest filter
        add_index = 2 # Instead of adding 1, we have to now add 2 to skip the filter with overlap
    filt_red = filt_names[detected_index+add_index]
    if detected_filt in ['f_f360m', 'f_f430m', 'f_f480m']: # These overlap with next bluest filter
        subtract_index = 2
    filt_blue = filt_names[detected_index-subtract_index]
    return filt_blue, filt_red


def get_filt_cols(df, skip_wide_bands=False):
    filt_cols = [col for col in df.columns if 'f_' in col]
    filt_cols = [col for col in filt_cols if 'alma' not in col]
    if skip_wide_bands ==  True:
        filt_cols = [col for col in filt_cols if 'w' not in col]
    return filt_cols

if __name__ == "__main__":
    make_line_coverage(halpha_name, halpha_wave, line_coverage_path)