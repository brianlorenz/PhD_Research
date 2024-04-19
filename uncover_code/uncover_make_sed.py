from uncover_read_data import read_supercat, read_raw_spec
from uncover_sed_filters import unconver_read_filters, get_filt_cols
import pandas as pd
import matplotlib.pyplot as plt

def main(id_msa):
    sed_df = get_sed(id_msa)
    spec_df = read_raw_spec(id_msa)
    plot_sed(sed_df, spec_df)

def get_sed(id_msa):
    supercat_df = read_supercat()
    row = supercat_df[supercat_df['id_msa'] == id_msa]
    filt_dir = unconver_read_filters()
    filt_cols = get_filt_cols(row)   
    fluxes = []
    e_fluxes = []
    eff_waves = []
    filt_names = []
    for col in filt_cols:
        filt_names.append(col)
        flux = row[col].iloc[0]
        eff_wave = filt_dir[col+'_wave_eff']
        ecol = col.replace('f_', 'e_')
        e_flux = row[ecol].iloc[0]

        fluxes.append(flux*1e-8) # Jy, originally 10 nJy
        e_fluxes.append(e_flux*1e-8) # Jy
        eff_waves.append(eff_wave/10000) # mincrons
    sed_df = pd.DataFrame(zip(filt_names, eff_waves, fluxes, e_fluxes), columns=['filter', 'eff_wavelength', 'flux', 'err_flux'])
    return sed_df

def plot_sed(sed_df, spec_df):
    font = 14
    fig, ax = plt.subplots(figsize=(6,6))
    ax.errorbar(spec_df['wave'], spec_df['flux'], yerr=spec_df['err'], marker='.', ls='-', color='orange')
    ax.errorbar(sed_df['eff_wavelength'], sed_df['flux'], yerr=sed_df['err_flux'], marker='o', ls='None', color='black')
    ax.set_xlabel('Effective Wavelength (um)', fontsize=font)
    ax.set_ylabel('Flux (Jy)', fontsize=font)
    plt.show()

# main(6226)
# main(7487)