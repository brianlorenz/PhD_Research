from sedpy import observate 
from uncover_read_data import read_supercat
# jwst_filters = ['f070w', 'f090w', 'f115w', 'f150w', 'f200w', 'f277w', 'f356w', 'f444w', 'f140m', 'f162m', 'f182m', 'f210m', 'f250m', 'f300m', 'f335m']

def unconver_read_filters():
    supercat_df = read_supercat()
    filt_cols = [col for col in supercat_df.columns if 'f_' in col]
    filt_cols = [col for col in filt_cols if 'alma' not in col]
    sedpy_filts = []
    for filt in filt_cols:
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
        print(f'Read filter {filt}')
        sedpy_filts.append(sedpy_filt)
    breakpoint()

unconver_read_filters()