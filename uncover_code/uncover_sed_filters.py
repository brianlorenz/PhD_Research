from sedpy import observate 
from uncover_read_data import read_supercat
import numpy as np
import pandas as pd
# jwst_filters = ['f070w', 'f090w', 'f115w', 'f150w', 'f200w', 'f277w', 'f356w', 'f444w', 'f140m', 'f162m', 'f182m', 'f210m', 'f250m', 'f300m', 'f335m']

filter_save_dir = '/Users/brianlorenz/uncover/Data/filter_curves/'

def unconver_read_filters(save_filts=False):
    supercat_df = read_supercat()
    filt_cols = get_filt_cols(supercat_df)
    sedpy_filts = []
    uncover_filt_dir = {}
    for filt in filt_cols:
        filtname = filt
        filt = filt.replace('f_', 'jwst_')
        try: 
            sedpy_filt = observate.load_filters([filt])
        except:
            try:
                filt = filt.replace('jwst_', 'wfc3_ir_')
                sedpy_filt = observate.load_filters([filt])
            except:
                filt = filt.replace('wfc3_ir_', 'acs_wfc_')
                sedpy_filt = observate.load_filters([filt])
        uncover_filt_dir[filtname+'_blue'] = sedpy_filt[0].blue_edge
        uncover_filt_dir[filtname+'_red'] = sedpy_filt[0].red_edge
        uncover_filt_dir[filtname+'_wave_eff'] = sedpy_filt[0].wave_effective
        uncover_filt_dir[filtname+'_width_eff'] = sedpy_filt[0].effective_width
        uncover_filt_dir[filtname+'_width_rect'] = sedpy_filt[0].rectangular_width

        scaled_trasm = sedpy_filt[0].transmission / np.max(sedpy_filt[0].transmission)
        trasm_low = scaled_trasm<0.2
        idx_lows = [i for i, x in enumerate(trasm_low) if x]
        idx_lows = np.array(idx_lows)
        max_idx = np.argmax(sedpy_filt[0].transmission)
        lower_cutoff_idx = np.max(idx_lows[idx_lows<max_idx])
        upper_cutoff_idx = np.min(idx_lows[idx_lows>max_idx])
        uncover_filt_dir[filtname+'_lower_20pct_wave'] = sedpy_filt[0].wavelength[lower_cutoff_idx]
        uncover_filt_dir[filtname+'_upper_20pct_wave'] = sedpy_filt[0].wavelength[upper_cutoff_idx]

        sedpy_filts.append(sedpy_filt[0])

        if save_filts:
            filt_df = pd.DataFrame(zip(sedpy_filt[0].wavelength, sedpy_filt[0].transmission, scaled_trasm), columns=['wavelength', 'transmission', 'scaled_transmission'])
            filt_df.to_csv(filter_save_dir+f'{filtname}_filter_curve.csv', index=False)
    
    return uncover_filt_dir, sedpy_filts

def get_filt_cols(df, skip_wide_bands=False):
    filt_cols = [col for col in df.columns if 'f_' in col]
    filt_cols = [col for col in filt_cols if 'alma' not in col]
    if skip_wide_bands ==  True:
        filt_cols = [col for col in filt_cols if 'w' not in col]
    return filt_cols

# uncover_filt_dir, sedpy_filts = unconver_read_filters(save_filts=True)
# breakpoint()