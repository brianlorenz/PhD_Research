import matplotlib.pyplot as plt
from astropy.io import ascii
import matplotlib as mpl
import numpy as np
from uncover_read_data import read_lineflux_cat, get_id_msa_list, read_SPS_cat

def plot_helium_vs_pab(id_msa_list, add_str, color_var='he_snr'):
    lineratio_df = ascii.read(f'/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/lineratio_df.csv', data_start=1).to_pandas()
    zqual_df_cont_covered = ascii.read('/Users/brianlorenz/uncover/zqual_df_cont_covered.csv').to_pandas()
    sps_df = read_SPS_cat()

    fig, axarr = plt.subplots(1, 2, figsize = (12, 6))
    fontsize = 12

    ax_ha_pab = axarr[0]
    ax_pab_he = axarr[1]

    ax_ha_pab.set_xlabel('Halpha flux', fontsize=fontsize)
    ax_ha_pab.set_ylabel('PaB flux', fontsize=fontsize)
    ax_pab_he.set_xlabel('Fe flux', fontsize=fontsize)
    ax_pab_he.set_ylabel('PaB flux', fontsize=fontsize)

    cmap = mpl.cm.inferno

    for id_msa in id_msa_list:
        helium_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting/helium/{id_msa}_emission_fits_helium.csv').to_pandas()
        emission_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting/{id_msa}_emission_fits.csv').to_pandas()
        lineratio_row = lineratio_df[lineratio_df['id_msa'] == id_msa]
        zqual_df_cont_covered_row = zqual_df_cont_covered[zqual_df_cont_covered['id_msa']==id_msa]
        sps_row = sps_df[sps_df['id_msa'] == id_msa]

        ha_flux = emission_df.iloc[0]['flux']
        pab_flux = emission_df.iloc[1]['flux']
        he_flux = helium_df.iloc[1]['flux']

        he_sigma = helium_df.iloc[1]['sigma']
        he_snr = helium_df.iloc[1]['signal_noise_ratio']

        if color_var == 'he_snr':
            norm = mpl.colors.Normalize(vmin=0, vmax=5) 
            rgba = cmap(norm(he_snr))
            color_str = '_he_snr'
        elif color_var == 'sed_ratio':
            norm = mpl.colors.Normalize(vmin=0, vmax=18) 
            print(lineratio_row['sed_lineratio'])
            rgba = cmap(norm(lineratio_row['sed_lineratio']))
            color_str = '_sed_ratio'
        elif color_var == 'redshift':
            norm = mpl.colors.Normalize(vmin=1.3, vmax=2.3) 
            rgba = cmap(norm(zqual_df_cont_covered_row['z_spec']))
            color_str = '_redshift'
        elif color_var == 'metallicity':
            norm = mpl.colors.Normalize(vmin=-1.2, vmax=0.1) 
            rgba = cmap(norm(sps_row['met_50']))
            print(sps_row['met_50'])
            color_str = '_metallicity'
        
        else:
            rgba = 'black'
            color_str = ''
        # print(he_snr)

        ax_ha_pab.plot(ha_flux, pab_flux, marker='o', color='black', ls='None')
        ax_ha_pab.text(ha_flux, pab_flux, f'{id_msa}')

        ax_pab_he.plot(he_flux, pab_flux, marker='o', color=rgba, ls='None', mec='black')
        ax_pab_he.text(he_flux, pab_flux, f'{id_msa}')

    ax_pab_he.set_xlim(-0.1e-18, 2.0e-18)
    ax_pab_he.set_ylim(0, 10e-18)
    ax_pab_he.plot([-10e-18, 10e-18], [-10e-18, 10e-18], ls='--', color='red', marker='None', label='one-to-one')
    ax_pab_he.plot([-5e-18, 5e-18], [-10e-18, 10e-18], ls='--', color='orange', marker='None', label='pab = 2x Fe')
    ax_pab_he.plot([-2.5e-18, 2.5e-18], [-10e-18, 10e-18], ls='--', color='green', marker='None', label='pab = 4x Fe')

    ax_pab_he.legend(loc=2)

    sm =  mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm)
    cbar.set_label('Fe SNR', fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    for ax in axarr:
        ax.tick_params(labelsize=fontsize)
        scale_aspect(ax)



    fig.savefig(f'/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/he_plots/helium_strength_{add_str}{color_str}.pdf')
    # plt.show()


        

def scale_aspect(ax):
    ylims = ax.get_ylim()
    xlims = ax.get_xlim()
    ydiff = np.abs(ylims[1]-ylims[0])
    xdiff = np.abs(xlims[1]-xlims[0])
    ax.set_aspect(xdiff/ydiff)

if __name__ == "__main__":
    # zqual_df_cont_covered = ascii.read('/Users/brianlorenz/uncover/zqual_df_cont_covered.csv').to_pandas()
    # id_msa_list = zqual_df_cont_covered['id_msa']
    # plot_helium_vs_pab(id_msa_list, 'full_sample')
    # plot_helium_vs_pab(id_msa_list, 'full_sample', color_var='sed_ratio')
    # plot_helium_vs_pab(id_msa_list, 'full_sample', color_var='redshift')


    # filtered_lineratio_df = ascii.read(f'/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/filtered_lineratio_df.csv').to_pandas()
    # id_msa_list = filtered_lineratio_df['id_msa']
    # plot_helium_vs_pab(id_msa_list, 'good_sample')
    # plot_helium_vs_pab(id_msa_list, 'good_sample', color_var='sed_ratio')
    # plot_helium_vs_pab(id_msa_list, 'good_sample', color_var='redshift')
    id_msa_list = get_id_msa_list(full_sample=False)
    # plot_helium_vs_pab(id_msa_list, '')
    id_msa_list = [25147, 39855, 42213, 47875]
    plot_helium_vs_pab(id_msa_list, '', color_var='metallicity')
    pass