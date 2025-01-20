from fit_emission_uncover_wave_divide import line_list, line_centers_rest

def get_line_flux_coverage(id_msa, line_name, redshift, sedpy_filt):
    fit_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting/{id_msa}_emission_fits.csv').to_pandas()
    if line_name == 'ha':
        line_wave = line_centers_rest[0] #angstrom
        line_width = fit_df['sigma'].iloc[0]
    if line_name == 'pab':
        line_wave = line_centers_rest[1]
        line_width = fit_df['sigma'].iloc[1]
    obs_line_wave = line_wave * (1+redshift)
    breakpoint()
