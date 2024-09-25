import matplotlib.pyplot as plt
from astropy.io import ascii
import matplotlib as mpl
import numpy as np

def plot_helium_vs_pab(id_msa_list, add_str, color_var='he_snr'):
    fig, axarr = plt.subplots(1, 2, figsize = (12, 6))
    fontsize = 12

    ax_ha_pab = axarr[0]
    ax_pab_he = axarr[1]

    ax_ha_pab.set_xlabel('Halpha flux', fontsize=fontsize)
    ax_ha_pab.set_ylabel('PaB flux', fontsize=fontsize)
    ax_pab_he.set_xlabel('He flux', fontsize=fontsize)
    ax_pab_he.set_ylabel('PaB flux', fontsize=fontsize)

    cmap = mpl.cm.inferno

    for id_msa in id_msa_list:
        helium_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting/helium/{id_msa}_emission_fits_helium.csv').to_pandas()
        emission_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting/{id_msa}_emission_fits.csv').to_pandas()

        ha_flux = emission_df.iloc[0]['flux']
        pab_flux = emission_df.iloc[1]['flux']
        he_flux = helium_df.iloc[1]['flux']

        he_sigma = helium_df.iloc[1]['sigma']
        he_snr = helium_df.iloc[1]['signal_noise_ratio']

        if color_var == 'he_snr':
            norm = mpl.colors.Normalize(vmin=0, vmax=5) 
            rgba = cmap(norm(he_snr))
        else:
            rgba = 'black'
        print(he_snr)

        ax_ha_pab.plot(ha_flux, pab_flux, marker='o', color='black', ls='None')
        ax_ha_pab.text(ha_flux, pab_flux, f'{id_msa}')

        ax_pab_he.plot(he_flux, pab_flux, marker='o', color=rgba, ls='None', mec='black')
        ax_pab_he.text(he_flux, pab_flux, f'{id_msa}')

    ax_pab_he.set_xlim(-0.1e-18, 1.5e-18)
    ax_pab_he.set_ylim(0, 6.5e-18)
    ax_pab_he.plot([-10e-18, 10e-18], [-10e-18, 10e-18], ls='--', color='red', marker='None', label='one-to-one')
    ax_pab_he.plot([-5e-18, 5e-18], [-10e-18, 10e-18], ls='--', color='orange', marker='None', label='pab = 2x he')
    ax_pab_he.plot([-2.5e-18, 2.5e-18], [-10e-18, 10e-18], ls='--', color='green', marker='None', label='pab = 4x he')

    ax_pab_he.legend()

    sm =  mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm)
    cbar.set_label('Helium SNR', fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    for ax in axarr:
        ax.tick_params(labelsize=fontsize)
        scale_aspect(ax)



    fig.savefig(f'/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/helium_strength_{add_str}.pdf')
    # plt.show()


        

def scale_aspect(ax):
    ylims = ax.get_ylim()
    xlims = ax.get_xlim()
    ydiff = np.abs(ylims[1]-ylims[0])
    xdiff = np.abs(xlims[1]-xlims[0])
    ax.set_aspect(xdiff/ydiff)


zqual_df_cont_covered = ascii.read('/Users/brianlorenz/uncover/zqual_df_cont_covered.csv').to_pandas()
id_msa_list = zqual_df_cont_covered['id_msa']
plot_helium_vs_pab(id_msa_list, 'full_sample')

filtered_lineratio_df = ascii.read(f'/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/filtered_lineratio_df.csv').to_pandas()
id_msa_list = filtered_lineratio_df['id_msa']
plot_helium_vs_pab(id_msa_list, 'good_sample')