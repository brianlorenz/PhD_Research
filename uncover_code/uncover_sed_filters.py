from sedpy import observate 
from uncover_read_data import read_supercat
# jwst_filters = ['f070w', 'f090w', 'f115w', 'f150w', 'f200w', 'f277w', 'f356w', 'f444w', 'f140m', 'f162m', 'f182m', 'f210m', 'f250m', 'f300m', 'f335m']

def unconver_read_filters():
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
        sedpy_filts.append(sedpy_filt[0])
    
    return uncover_filt_dir, sedpy_filts

def get_filt_cols(df):
    filt_cols = [col for col in df.columns if 'f_' in col]
    filt_cols = [col for col in filt_cols if 'alma' not in col]
    return filt_cols
