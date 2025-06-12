from uncover_read_data import get_id_msa_list, read_supercat
from astropy.io import ascii
from uncover_sed_filters import get_filt_cols

id_msa_list = get_id_msa_list(full_sample=False)
supercat_df = read_supercat()
filt_cols = get_filt_cols(supercat_df)
for id_msa in id_msa_list:
    supercat_row = supercat_df[supercat_df['id_msa']==id_msa]
    id_dr3 = int(supercat_row.iloc[0]['id'])

    int_spec_df = ascii.read(f'/Users/brianlorenz/uncover/Data/integrated_specs/{id_msa}_integrated_spec.csv').to_pandas()
    
    int_spec_df['filt_name'] = filt_cols
    w_idxs=[]
    for i in range(len(int_spec_df)):
        if 'w' in int_spec_df['filt_name'].iloc[i]:
            w_idxs.append(i) 
    int_spec_df = int_spec_df.drop(w_idxs)
    print(int_spec_df)
    print(id_msa)
    breakpoint()
    
ha_skips = []
pab_skips = []

"""
250 300 = 2 = 4908
360 410 = 4 = 4598
300 335 = 1 = 3659
335 360 = 4 = 2607
[4908, 4908, 4598, 4598, 4598, 4598, 3659, 2607, 2607, 2607, 2607]
average separation = 3845.0

360 430 = 3 = 6606
compared to 6606
"""
