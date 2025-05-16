from uncover_input_data import read_supercat
from uncover_filters import unconver_read_filters, get_filt_cols
import pandas as pd

def get_sed(id_dr3):
    """Given an id_dr3, turn the row from the supercat into a more readable SED with flux in Jansky

    Parameters:
    id_dr3 (int): DR3 form UNCOVER

    Returns
    sed_df (pd.DataFrame): dataframe of the sed
    """
    supercat_df = read_supercat()
    row = supercat_df[supercat_df['id'] == id_dr3]
    filt_dir, filters = unconver_read_filters()
    filt_cols = get_filt_cols(row)   
    fluxes = []
    e_fluxes = []
    eff_waves = []
    filt_names = []
    eff_widths = []
    rect_widths = []
    for col in filt_cols:
        filt_names.append(col)
        flux = row[col].iloc[0]
        eff_wave = filt_dir[col+'_wave_eff']
        ecol = col.replace('f_', 'e_')
        e_flux = row[ecol].iloc[0]
        eff_width = filt_dir[col+'_width_eff']
        rect_width = filt_dir[col+'_width_rect']

        fluxes.append(flux*1e-8) # Jy, originally 10 nJy
        e_fluxes.append(e_flux*1e-8) # Jy
        eff_waves.append(eff_wave/10000) # microns
        eff_widths.append(eff_width) # microns
        rect_widths.append(rect_width) # microns
    sed_df = pd.DataFrame(zip(filt_names, eff_waves, fluxes, e_fluxes, eff_widths, rect_widths), columns=['filter', 'eff_wavelength', 'flux', 'err_flux', 'eff_width', 'rectangular_width'])
    return sed_df