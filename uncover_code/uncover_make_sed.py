from uncover_read_data import read_supercat
from uncover_sed_filters import unconver_read_filters

def get_sed(id_msa):
    supercat_df = read_supercat()
    row = supercat_df[supercat_df['id_msa'] == id_msa]
    # Ideally make a new row for wavelength
    sed_filts = unconver_read_filters(row)
    sedpy_test = observate.load_filters(['jwst_f410m', 'jwst_f41230m'])
    breakpoint()
    obs["filters"] = observate.load_filters(filters_list)
    return row