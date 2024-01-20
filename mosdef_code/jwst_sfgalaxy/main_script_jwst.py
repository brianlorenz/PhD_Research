from astropy.io import ascii
from fit_emission_jwst import fit_emission
from read_jwst_spectrum import main_read_spec, flux_columns
from read_catalog import spec_loc, sed_loc
from scale_spec import scale_spec
from compute_sfr_jwst import compute_sfr

def main():
    spec_df = main_read_spec()
    sed_df = ascii.read(sed_loc).to_pandas()
    scale, spec_df_scaled = scale_spec(sed_df, spec_df)
    spec_df_scaled.to_csv(spec_loc, index=False)
    fit_emission(spec_df_scaled, 'combined')
    compute_sfr()


main()