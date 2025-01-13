from uncover_read_data import read_supercat, read_raw_spec, read_aper_cat
from uncover_sed_filters import unconver_read_filters, get_filt_cols
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import ascii

def main(id_msa):
    sed_df = get_sed(id_msa)
    spec_df = read_raw_spec(id_msa)
    plot_sed(sed_df, spec_df)

def read_sed(id_msa, aper_size='None'):
    sed_loc = f'/Users/brianlorenz/uncover/Data/seds/{id_msa}_sed.csv'
    if aper_size != 'None':
        sed_loc = f'/Users/brianlorenz/uncover/Data/seds/aper{aper_size}/{id_msa}_sed_aper{aper_size}.csv'
    sed_df = ascii.read(sed_loc).to_pandas()
    return sed_df

def make_aper_seds(id_msa_list, aper_size='048'):
    for id_msa in id_msa_list:
        sed_df = get_sed(id_msa, aper_size=aper_size)
        sed_df.to_csv(f'/Users/brianlorenz/uncover/Data/seds/aper{aper_size}/{id_msa}_sed_aper{aper_size}.csv', index=False)

def get_sed(id_msa, aper_size = 'None'):
    supercat_df = read_supercat()
    if aper_size != 'None':
        supercat_df = read_aper_cat(aper_size=aper_size)
    row = supercat_df[supercat_df['id_msa'] == id_msa]
    if id_msa == 42041:
        row = supercat_df[supercat_df['id'] == 54635]
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
    # breakpoint()
    return sed_df

def plot_sed(sed_df, spec_df):
    font = 14
    fig, ax = plt.subplots(figsize=(6,6))
    ax.errorbar(spec_df['wave'], spec_df['flux'], yerr=spec_df['err'], marker='.', ls='-', color='orange')
    ax.errorbar(sed_df['eff_wavelength'], sed_df['flux'], yerr=sed_df['err_flux'], marker='o', ls='None', color='black')
    ax.set_xlabel('Effective Wavelength (um)', fontsize=font)
    ax.set_ylabel('Flux (Jy)', fontsize=font)
    plt.show()

if __name__ == "__main__":
    zqual_detected_df = ascii.read('/Users/brianlorenz/uncover/zqual_detected.csv').to_pandas()
    id_msa_list = zqual_detected_df['id_msa'].to_list()
    make_aper_seds(id_msa_list, aper_size='140')
    # main(6291)
    # main(6325)
    # main(7887)
    # main(9457)
    pass