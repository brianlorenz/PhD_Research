from sedpy import observate 
from uncover_input_data import read_supercat
import numpy as np
import pandas as pd

def unconver_read_filters():
    """Pulls the filters from supercat, then returns a dict with lots of useful filter info
    """
    supercat_df = read_supercat() # This is the fits file that has all of the medium band flux info
    filt_cols = get_filt_cols(supercat_df) # Grabs a list of the filter names from the columns of supercat
    
    # Setup list and dictionary to store the info... I'm sure there are better ways to do this, but this is what I know :) 
    sedpy_filts = [] 
    uncover_filt_dict = {}

    for filt in filt_cols: # Loops through each filter to read the info, then store it in the list and dict above
        filtname = filt
        filt = filt.replace('f_', 'jwst_') 
        
        # These try...excepts are also super sloppy - sedpy needs the filter to have the right name
        # Most names are jwst_f000m. But a few of the bands are from other telescopes
        # So this essentially says "try the jwst name. If it throws and error, instead try these other names"
        # It does eventually work and capture all of the filters 
        try: 
            sedpy_filt = observate.load_filters([filt])
        except:
            try:
                filt = filt.replace('jwst_', 'wfc3_ir_')
                sedpy_filt = observate.load_filters([filt])
            except:
                filt = filt.replace('wfc3_ir_', 'acs_wfc_')
                sedpy_filt = observate.load_filters([filt])

        # Saves a bunch of useful information to the dictionary
        uncover_filt_dict[filtname+'_blue'] = sedpy_filt[0].blue_edge 
        uncover_filt_dict[filtname+'_red'] = sedpy_filt[0].red_edge
        uncover_filt_dict[filtname+'_wave_eff'] = sedpy_filt[0].wave_effective # This is the wavelength of the filter, very useful
        uncover_filt_dict[filtname+'_width_eff'] = sedpy_filt[0].effective_width
        uncover_filt_dict[filtname+'_width_rect'] = sedpy_filt[0].rectangular_width

        # This whole chunk of codes notes where the transmission of the filter is less than 20% of its maximum, then saves those values as well
        # We may not want to include anything where the emission line falls in this very low-transmission region of the filter
        scaled_transmission = sedpy_filt[0].transmission / np.max(sedpy_filt[0].transmission)
        trasm_low = scaled_transmission<0.2
        idx_lows = [i for i, x in enumerate(trasm_low) if x]
        idx_lows = np.array(idx_lows)
        max_idx = np.argmax(sedpy_filt[0].transmission)
        lower_cutoff_idx = np.max(idx_lows[idx_lows<max_idx])
        upper_cutoff_idx = np.min(idx_lows[idx_lows>max_idx])
        uncover_filt_dict[filtname+'_lower_20pct_wave'] = sedpy_filt[0].wavelength[lower_cutoff_idx]
        uncover_filt_dict[filtname+'_upper_20pct_wave'] = sedpy_filt[0].wavelength[upper_cutoff_idx]

        # Saves the sedpy option into the list - in case we need to reference other properties directly from sedpy later
        sedpy_filts.append(sedpy_filt[0])

    return uncover_filt_dict, sedpy_filts # Returns the dictionary and list with all the info saved

def get_filt_cols(supercat_df, skip_wide_bands=False):
    """Grabs the names of the filter columns we need
    Filters are of the format 'f_f000m' for medium bands and 'f_f000w' for wide bands. 

    Parameters:
    supercat_df (pd.DataFrame) - the supercat_df, where we pull the names from

    Returns:
    filter_cols (list) - the names of the filter columns
    """
    filt_cols = [col for col in supercat_df.columns if 'f_' in col] # Selects all m and w bands, as well as f_alma
    filt_cols = [col for col in filt_cols if 'alma' not in col] # Removes the alma band
    if skip_wide_bands ==  True:
        filt_cols = [col for col in filt_cols if 'w' not in col] # Removes the wide bands, if selected
    return filt_cols

