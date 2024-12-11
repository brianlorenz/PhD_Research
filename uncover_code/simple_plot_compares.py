import matplotlib.pyplot as plt
from astropy.io import ascii
from plot_vals import scale_aspect

def plot_simpletests(all=False):
    data_df = ascii.read(f'/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/simpletest_offset_df.csv')
    if all:
        data_df = ascii.read(f'/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/simpletest_offset_df_all.csv')

    fig, axarr = plt.subplots(2,3,figsize=(16,10))
    ax_ha_sed_vs_cat = axarr[0,0]
    ax_ha_sed_vs_emfit = axarr[0,1]
    ax_ha_cat_vs_emfit = axarr[0,2]
    ax_pab_sed_vs_cat = axarr[1,0]
    ax_pab_sed_vs_emfit = axarr[1,1]
    ax_pab_cat_vs_emfit = axarr[1,2]
    ax_list = [ax_ha_sed_vs_cat, ax_ha_sed_vs_emfit, ax_ha_cat_vs_emfit, ax_pab_sed_vs_cat, ax_pab_sed_vs_emfit, ax_pab_cat_vs_emfit]

    sed_label = 'SED method'
    cat_label = 'UNCOVER Catalog'
    emfit_label = 'emission fit'

    ax_ha_sed_vs_cat.plot(data_df['ha_cat_flux'], data_df['ha_sed_flux'], marker='o', color='black', ls='None')
    ax_ha_sed_vs_cat.set_xlabel(cat_label)
    ax_ha_sed_vs_cat.set_ylabel(sed_label)

    ax_ha_sed_vs_emfit.plot(data_df['ha_emfit_flux'], data_df['ha_sed_flux'], marker='o', color='black', ls='None')
    ax_ha_sed_vs_emfit.set_xlabel(emfit_label)
    ax_ha_sed_vs_emfit.set_ylabel(sed_label)

    ax_ha_cat_vs_emfit.plot(data_df['ha_cat_flux'], data_df['ha_emfit_flux'], marker='o', color='black', ls='None')
    ax_ha_cat_vs_emfit.set_xlabel(cat_label)
    ax_ha_cat_vs_emfit.set_ylabel(emfit_label)

    ax_pab_sed_vs_cat.plot(data_df['pab_cat_flux'], data_df['pab_sed_flux'], marker='o', color='black', ls='None')
    ax_pab_sed_vs_cat.set_xlabel(cat_label)
    ax_pab_sed_vs_cat.set_ylabel(sed_label)

    ax_pab_sed_vs_emfit.plot(data_df['pab_emfit_flux'], data_df['pab_sed_flux'], marker='o', color='black', ls='None')
    ax_pab_sed_vs_emfit.set_xlabel(emfit_label)
    ax_pab_sed_vs_emfit.set_ylabel(sed_label)

    ax_pab_cat_vs_emfit.plot(data_df['pab_cat_flux'], data_df['pab_emfit_flux'], marker='o', color='black', ls='None')
    ax_pab_cat_vs_emfit.set_xlabel(cat_label)
    ax_pab_cat_vs_emfit.set_ylabel(emfit_label)


    for ax in ax_list:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(labelsize=12)
        ax.plot([1e-20, 1e-15], [1e-20, 1e-15], ls='--', color='red', marker='None')
        ax.set_xlim([1e-20, 1e-15])
        ax.set_ylim([1e-20, 1e-15])
        # scale_aspect(ax)
        
    plt.tight_layout()
    save_loc = '/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/simpletest_flux_compare.pdf'
    if all:
        save_loc = '/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/simpletest_flux_compare_all.pdf'
    fig.savefig(save_loc)

def plot_offsets(all=False):
    data_df = ascii.read(f'/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/simpletest_offset_df.csv')
    if all:
        data_df = ascii.read(f'/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/simpletest_offset_df_all.csv')

    fig, ax = plt.subplots(1,1,figsize=(6,6))
   
    # sed_label = 'SED method'
    # cat_label = 'UNCOVER Catalog'
    # emfit_label = 'emission fit'

    ax.plot(data_df['ha_sed_div_emfit'], data_df['pab_sed_div_emfit'], marker='o', color='black', ls='None')
    ax.set_xlabel('Ha offset sed/emfit')
    ax.set_ylabel('PaB offset sed/emfit')

    


    # for ax in ax_list:
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(labelsize=12)
    ax.plot([-10, 100], [-10, 100], ls='--', color='red', marker='None')
    ax.set_xlim([0.5, 40])
    ax.set_ylim([0.5, 40])
    
        
    plt.tight_layout()
    save_loc = '/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/simpletest_offset_compare.pdf'
    if all:
        save_loc = '/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/simpletest_offset_compare_all.pdf'
    fig.savefig(save_loc)

plot_simpletests(all=True)
plot_offsets(all=True)